"""
Задача: INT8 MatMul with Fused Dequant (Triton)
(дедлайн МСК: 10.11.2025 23:59 MSK)

Задача: INT8 матмул с фьюзом деквантизации и биаса

Даны тензоры:
- X_q формы (B, IN) — int8 входные активации (row-major, contiguous);
- W_q формы (IN, OUT) — int8 веса (row-major, contiguous);
- s_x — fp32 скаляр масштаба для X_q (per-tensor);
- s_w — fp32 масштабы для W_q: либо скаляр (per-tensor), либо вектор формы (OUT,) для per-OUT-channel;
- bias — опционально fp16 вектор формы (OUT,).

Нужно вычислить выход Y формы (B, OUT) в fp16.
Математика (словами):
1) накапливайте произведения int8*int8 в int32: ACC[b, o] = sum_k X_q[b, k] * W_q[k, o];
2) вычислите масштаб на столбец o: alpha[o] = s_x * (s_w[o] при per-channel, иначе s_w);
3) примените масштаб после окончания цикла по k: Y[b, o] = fp16( fp32(ACC[b, o]) * alpha[o] + bias[o] (если задан) ).

Требования:
- Реализовать Triton-ядро, которое загружает тайлы int8 из X_q и W_q, аккумулирует в int32, применяет масштабы один раз после K-цикла (fused dequant), добавляет bias внутри ядра (если он есть) и пишет Y как fp16 (row-major).
- Поддержать per-channel и per-tensor варианты масштабов весов (constexpr-флаг PER_CHANNEL).
- Обязательно маскировать хвосты по всем измерениям.
- Использовать tl.load и tl.store.
- Обязательно использовать @triton.autotune с не менее чем 3 различными конфигурациями (BLOCK_M/N/K, num_warps, num_stages).

--------------------------------

These are **tiling parameters** for matrix multiplication optimization. Let me explain:

For the matrix multiplication `Y[B, OUT] = X_q[B, IN] @ W_q[IN, OUT]`:

- **BLOCK_M**: Tile size along the **M (row) dimension** = how many rows of X (and Y) each program block processes at once
  - M corresponds to **B** (batch dimension)

- **BLOCK_N**: Tile size along the **N (column) dimension** = how many columns of W (and Y) each program block processes at once
  - N corresponds to **OUT** (output dimension)

- **BLOCK_K**: Tile size along the **K (reduction) dimension** = chunk size for the inner loop over the reduction dimension
  - K corresponds to **IN** (input dimension, the shared dimension being summed over)

**Visual example:**

```
X_q [B=256, IN=1024]  @  W_q [IN=1024, OUT=512]  =  Y [B=256, OUT=512]
     ↑       ↑              ↑         ↑                ↑        ↑
   BLOCK_M BLOCK_K      BLOCK_K   BLOCK_N          BLOCK_M  BLOCK_N
```

**How it works:**

Each program block computes a `BLOCK_M x BLOCK_N` tile of the output by:
1. Looping over the K dimension in chunks of `BLOCK_K`
2. Loading `BLOCK_M x BLOCK_K` from X_q
3. Loading `BLOCK_K x BLOCK_N` from W_q
4. Accumulating the products in int32
5. After all K chunks, apply scaling and write the `BLOCK_M x BLOCK_N` result

**Autotune configs:**
```python
triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, ...)  # balanced
triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, ...) # wider tiles in M
triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, ...) # wider tiles in N
```

Different tile sizes perform better depending on the matrix shapes and hardware!
"""

import torch
import triton
import triton.language as tl


# fmt: off
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    ],
    key=["B", "IN", "OUT"],
)
# fmt: on
@triton.jit
def _forward_int8_fused_kernel(
    x_q_ptr,
    x_scale_ptr,
    w_q_ptr,
    w_scale_ptr,
    b_ptr,
    y_ptr,
    B,
    IN,
    OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Each program computes one BLOCK_M x BLOCK_N tile of output
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator in int32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Loop over K dimension in chunks of BLOCK_K
    for k in range(0, IN, BLOCK_K):
        k_offs = k + offs_k

        # Load X_q tile: [BLOCK_M, BLOCK_K] (row-major: X[b, k] = b*IN + k)
        x_offsets = offs_m[:, None] * IN + k_offs[None, :]
        x_mask = (offs_m[:, None] < B) & (k_offs[None, :] < IN)
        x = tl.load(x_q_ptr + x_offsets, mask=x_mask, other=0)

        # Load W_q tile: [BLOCK_K, BLOCK_N] (row-major: W[k, o] = k*OUT + o)
        w_offsets = k_offs[:, None] * OUT + offs_n[None, :]
        w_mask = (k_offs[:, None] < IN) & (offs_n[None, :] < OUT)
        w = tl.load(w_q_ptr + w_offsets, mask=w_mask, other=0)

        # Accumulate: acc += X @ W (int8 x int8 -> int32 accumulation)
        acc += tl.dot(x, w, out_dtype=tl.int32)

    # Now apply fused dequantization
    # Load x_scale (scalar)
    x_scale = tl.load(x_scale_ptr)

    # Load w_scale (per-channel or scalar)
    if PER_CHANNEL:
        w_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < OUT, other=0.0)
    else:
        w_scale = tl.load(w_scale_ptr)

    # Compute alpha = x_scale * w_scale
    alpha = x_scale * w_scale

    # Convert accumulator to fp32 and apply scale
    y = acc.to(tl.float32) * alpha[None, :]

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < OUT, other=0.0)
        y += bias[None, :]

    # Convert to fp16 and store
    y = y.to(tl.float16)

    # Store output tile (row-major: Y[b, o] = b*OUT + o)
    y_offsets = offs_m[:, None] * OUT + offs_n[None, :]
    y_mask = (offs_m[:, None] < B) & (offs_n[None, :] < OUT)
    tl.store(y_ptr + y_offsets, y, mask=y_mask)


def matmul_int8_fused(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    per_channel: bool = True,
) -> torch.Tensor:
    """Вернуть Y = dequant(X_q) @ dequant(W_q) + bias, dtype fp16, shape (B, OUT)."""
    # Validate inputs
    assert x_q.is_cuda and w_q.is_cuda, "Tensors must be on CUDA"
    assert x_q.dtype == torch.int8 and w_q.dtype == torch.int8, "Inputs must be int8"
    assert x_q.dim() == 2 and w_q.dim() == 2, "Inputs must be 2D"
    assert x_scale.dtype == torch.float32 and w_scale.dtype == torch.float32, "Scales must be fp32"

    B, IN = x_q.shape
    IN_w, OUT = w_q.shape
    assert IN == IN_w, "Inner dimensions must match"

    # Validate scales
    assert x_scale.numel() == 1, "x_scale must be scalar"
    if per_channel:
        assert w_scale.shape == (OUT,), f"w_scale must have shape ({OUT},) for per-channel"
    else:
        assert w_scale.numel() == 1, "w_scale must be scalar for per-tensor"

    # Validate bias
    if bias is not None:
        assert bias.shape == (OUT,), f"bias must have shape ({OUT},)"
        assert bias.dtype == torch.float16, "bias must be fp16"
        bias = bias.cuda()

    # Allocate output
    y = torch.empty((B, OUT), dtype=torch.float16, device=x_q.device)

    # Launch kernel with 2D grid
    def grid(meta):
        return (
            triton.cdiv(B, meta['BLOCK_M']),
            triton.cdiv(OUT, meta['BLOCK_N']),
        )

    _forward_int8_fused_kernel[grid](
        x_q,
        x_scale,
        w_q,
        w_scale,
        bias if bias is not None else x_q,  # dummy pointer if no bias
        y,
        B,
        IN,
        OUT,
        PER_CHANNEL=per_channel,
        HAS_BIAS=bias is not None,
    )

    return y


if __name__ == "__main__":
    B, IN, OUT = 128, 256, 512

    # Create int8 tensors
    x_q = torch.randint(-127, 128, (B, IN), dtype=torch.int8).cuda()
    w_q = torch.randint(-127, 128, (IN, OUT), dtype=torch.int8).cuda()

    # Create scales
    x_scale = torch.tensor(0.01, dtype=torch.float32).cuda()
    w_scale = torch.rand(OUT, dtype=torch.float32).cuda() * 0.02  # per-channel

    # Create bias
    bias = torch.randn(OUT, dtype=torch.float16).cuda()

    # Test with per-channel
    y = matmul_int8_fused(x_q, x_scale, w_q, w_scale, bias, per_channel=True)
    print(f"Output shape: {y.shape}, dtype: {y.dtype}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")

    # Test without bias
    y_no_bias = matmul_int8_fused(x_q, x_scale, w_q, w_scale, None, per_channel=True)
    print(f"No bias output shape: {y_no_bias.shape}")

    # Verify correctness (dequantize and compute in fp32)
    x_fp = x_q.float() * x_scale.item()
    w_fp = w_q.float() * w_scale.unsqueeze(0)
    y_ref = (x_fp @ w_fp + bias).half()
    error = (y - y_ref).abs().max().item()
    print(f"Max error vs reference: {error:.6f}")
