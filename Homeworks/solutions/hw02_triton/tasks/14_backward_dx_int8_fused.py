"""
Задача: INT8 Backward dX with Fused Dequant (Triton)
(дедлайн МСК: 10.11.2025 23:59 MSK)

Задача: беквард по входу с фьюзом деквантизации весов (INT8)

Дано:
- dY формы (B, OUT) — fp16 (row-major, contiguous) — апстрим-градиент;
- W_q формы (IN, OUT) — int8 (row-major, contiguous) — квантованные веса;
- s_w — fp32 масштабы для W_q: либо скаляр (per-tensor), либо вектор формы (OUT,) (per-OUT-channel).

Нужно вычислить dX формы (B, IN) в fp16:
- накапливайте dY @ (W_q_deq)^T в fp32, где W_q_deq = W_q * s_w (пер-столбец или скаляр);
- применяйте масштабы один раз после цикла по K (фьюз деквантизации на выходе тайла);
- запишите результат как fp16.

Математика (словами):
- W_q_deq[:, o] = W_q[:, o] * s_w[o] при per-channel, иначе * s_w_scalar;
- dX[b, i] = sum_k dY[b, k] * W_q_deq[i, k];
- хранение и доступ — row-major.

Требования:
- Реализовать Triton-ядро, которое грузит тайлы dY (fp16) и W_q (int8), аккумулирует в fp32, умножает на s_w (вектор по OUT или скаляр) внутри K-цикла или сразу после него один раз (разрешено умножать загружаемый тайл весов; важно, чтобы масштабы не применялись к каждому элементу повторно после суммирования);
- Маскировать хвосты по всем измерениям;
- Использовать tl.load и tl.store;
- Обязательно использовать @triton.autotune с не менее чем 3 конфигурациями (BLOCK_M/N/K, num_warps, num_stages);
- Поддержать режимы per_channel=True/False (constexpr-флаг PER_CHANNEL).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    ],
    key=["B", "IN", "OUT"],
)
@triton.jit
def _backward_dx_fused_kernel(
    dy_ptr,
    wq_ptr,
    w_scale_ptr,
    dx_ptr,
    B,
    IN,
    OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
):
    # Compute dX = dY @ W_deq^T
    # dY: (B, OUT) - fp16
    # W_q: (IN, OUT) - int8
    # dX: (B, IN) - fp16

    # Each program computes one BLOCK_M x BLOCK_N tile of dX
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this tile of dX
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # batch indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # input feature indices
    offs_k = tl.arange(0, BLOCK_K)  # output feature indices (reduction dim)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (OUT) in chunks of BLOCK_K
    for k in range(0, OUT, BLOCK_K):
        k_offs = k + offs_k

        # Load dY tile: [BLOCK_M, BLOCK_K] (row-major: dY[b, o] = b*OUT + o)
        dy_offsets = offs_m[:, None] * OUT + k_offs[None, :]
        dy_mask = (offs_m[:, None] < B) & (k_offs[None, :] < OUT)
        dy = tl.load(dy_ptr + dy_offsets, mask=dy_mask, other=0.0)

        # Load W_q tile: [BLOCK_N, BLOCK_K] (row-major: W[i, o] = i*OUT + o)
        # We need W_q[n:n+BLOCK_N, k:k+BLOCK_K]
        wq_offsets = offs_n[:, None] * OUT + k_offs[None, :]
        wq_mask = (offs_n[:, None] < IN) & (k_offs[None, :] < OUT)
        wq = tl.load(wq_ptr + wq_offsets, mask=wq_mask, other=0)

        # Convert W_q from int8 to fp32
        wq_fp = wq.to(tl.float32)

        # Load and apply w_scale (per-channel or scalar)
        if PER_CHANNEL:
            w_scale = tl.load(w_scale_ptr + k_offs, mask=k_offs < OUT, other=0.0)
        else:
            w_scale = tl.load(w_scale_ptr)

        # Dequantize W_q: W_deq = W_q * s_w
        # w_scale has shape [BLOCK_K], wq_fp has shape [BLOCK_N, BLOCK_K]
        wq_deq = wq_fp * w_scale[None, :]

        # Convert dY to fp32
        dy_fp = dy.to(tl.float32)

        # Accumulate: dX += dY @ W_deq^T
        # dY: [BLOCK_M, BLOCK_K], W_deq^T: [BLOCK_K, BLOCK_N]
        # We need to transpose W_deq from [BLOCK_N, BLOCK_K] to [BLOCK_K, BLOCK_N]
        wq_deq_t = tl.trans(wq_deq)

        # Compute dot product and accumulate
        acc += tl.dot(dy_fp, wq_deq_t)

    # Convert accumulator to fp16
    dx = acc.to(tl.float16)

    # Store output tile (row-major: dX[b, i] = b*IN + i)
    dx_offsets = offs_m[:, None] * IN + offs_n[None, :]
    dx_mask = (offs_m[:, None] < B) & (offs_n[None, :] < IN)
    tl.store(dx_ptr + dx_offsets, dx, mask=dx_mask)


def backward_dx_int8_fused(
    dy: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    per_channel: bool = True,
) -> torch.Tensor:
    """Вернуть dX = dY @ (dequant(W_q))^T, dtype fp16, shape (B, IN)."""
    assert dy.is_cuda and w_q.is_cuda, "Tensors must be on CUDA"
    assert dy.dtype == torch.float16 and w_q.dtype == torch.int8, "Inputs must be fp16 and int8"
    assert dy.dim() == 2 and w_q.dim() == 2, "Inputs must be 2D"
    assert w_scale.dtype == torch.float32, "Scales must be fp32"

    B, OUT = dy.shape
    IN, OUT_w = w_q.shape
    assert OUT == OUT_w, "Inner dimensions must match"

    # Validate scales
    if per_channel:
        assert w_scale.shape == (OUT,), f"w_scale must have shape ({OUT},) for per-channel"
    else:
        assert w_scale.numel() == 1, "w_scale must be scalar for per-tensor"

    # Allocate output
    dx = torch.empty((B, IN), dtype=torch.float16, device=dy.device)

    # Launch kernel with 2D grid
    def grid(meta):
        return (
            triton.cdiv(B, meta['BLOCK_M']),
            triton.cdiv(IN, meta['BLOCK_N']),
        )

    _backward_dx_fused_kernel[grid](
        dy,
        w_q,
        w_scale,
        dx,
        B,
        IN,
        OUT,
        PER_CHANNEL=per_channel,
    )

    return dx

if __name__ == "__main__":
    B, IN, OUT = 128, 256, 512

    # Create test tensors
    dy = torch.randn(B, OUT, dtype=torch.float16).cuda()
    w_q = torch.randint(-127, 128, (IN, OUT), dtype=torch.int8).cuda()
    w_scale_per_channel = torch.rand(OUT, dtype=torch.float32).cuda() * 0.02  # per-channel
    w_scale_scalar = torch.tensor(0.015, dtype=torch.float32).cuda()  # per-tensor

    # Test with per-channel
    print("=" * 60)
    print("Testing per-channel mode:")
    dx_pc = backward_dx_int8_fused(dy, w_q, w_scale_per_channel, per_channel=True)
    print(f"Output shape: {dx_pc.shape}, dtype: {dx_pc.dtype}")
    print(f"Output range: [{dx_pc.min().item():.4f}, {dx_pc.max().item():.4f}]")

    # Verify correctness for per-channel
    w_deq_pc = w_q.float() * w_scale_per_channel[None, :]
    dx_ref_pc = (dy.float() @ w_deq_pc.T).half()
    error_pc = (dx_pc - dx_ref_pc).abs().max().item()
    print(f"Max error vs reference (per-channel): {error_pc:.6f}")

    # Test with per-tensor (scalar)
    print("\n" + "=" * 60)
    print("Testing per-tensor mode:")
    dx_pt = backward_dx_int8_fused(dy, w_q, w_scale_scalar, per_channel=False)
    print(f"Output shape: {dx_pt.shape}, dtype: {dx_pt.dtype}")
    print(f"Output range: [{dx_pt.min().item():.4f}, {dx_pt.max().item():.4f}]")

    # Verify correctness for per-tensor
    w_deq_pt = w_q.float() * w_scale_scalar.item()
    dx_ref_pt = (dy.float() @ w_deq_pt.T).half()
    error_pt = (dx_pt - dx_ref_pt).abs().max().item()
    print(f"Max error vs reference (per-tensor): {error_pt:.6f}")

    print("\n" + "=" * 60)
    if error_pc < 0.1 and error_pt < 0.1:
        print("✓ All tests passed!")
    else:
        print("✗ Tests failed!")
