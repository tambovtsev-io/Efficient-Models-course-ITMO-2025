"""
Задача: Global INT8 Quantize + Transpose (Triton)
(дедлайн МСК: 10.11.2025 23:59 MSK)

Задача: глобальная симметричная квантизация в int8 с транспозированием

Дан 2D тензор X формы (M, N) (fp16/fp32, row-major, contiguous). Требуется:
1) вычислить absmax = max(|X|) (скаляр fp32);
2) квантизовать элементы по формуле Q = round(127 * X / denom), где denom = max(absmax, 127*1e-8);
3) записать результат в выходной тензор B формы (N, M) в int8 (то есть сразу хранить Q^T);
4) вернуть (B, absmax[1]) — обратите внимание: возвращаем именно absmax без клэмпа, a denom используется только во избежание деления на 0.

Требования:
- Реализовать одно Triton-ядро, которое загружает тайлы из A (X), квантизует и пишет в B (транспонируя индексы);
- Использовать tl.load / tl.store и маски по краям;
- Обязательно использовать @triton.autotune с не менее чем 3 конфигурациями (BLOCK_M/N, GROUP_M, num_warps/num_stages);
- Поддержать произвольные (но совместимые) страйды: у входа хотя бы одно измерение unit-stride, у выхода тоже хотя бы одно измерение unit-stride;
- Python-обёртка quantize_global_transpose(x) возвращает (q_T:int8[N,M], absmax:fp32[1]).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
    ],
    key=["M", "N"],
)
@triton.jit
def _quantize_global_transpose(
    A,
    absmax_ptr,
    B,
    stride_am,
    stride_an,
    stride_bn,
    stride_bm,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Setup pids
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Setup offsets and masks
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    offsets_m = block_start_m + tl.arange(0, BLOCK_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_N)
    mask_m = offsets_m < M
    mask_n = offsets_n < N

    # Load x tile from A[M, N]
    x = tl.load(
        A + (offsets_m * stride_am)[:, None] + (offsets_n * stride_an)[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    # Load absmax and compute denom
    absmax = tl.load(absmax_ptr)
    denom = tl.maximum(absmax, 127.0 * 1e-8)

    # Quantize: Q = round(127 * X / denom)
    scaled = 127.0 * (x / denom)
    # Manual rounding: add 0.5 for positive, subtract 0.5 for negative, then truncate
    q = tl.where(scaled >= 0, scaled + 0.5, scaled - 0.5)
    q = q.to(tl.int8)

    # Store q tile transposed to B[N, M]
    # We loaded A[offsets_m, offsets_n], now store to B[offsets_n, offsets_m]
    tl.store(
        B + (offsets_n * stride_bn)[:, None] + (offsets_m * stride_bm)[None, :],
        tl.trans(q),
        mask=mask_n[:, None] & mask_m[None, :],
    )


def quantize_global_transpose(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (q_T:int8[N,M], absmax:fp32[1])."""
    assert x.is_cuda, "Tensor must be on CUDA"
    assert x.dtype in [torch.float16, torch.float32], "Tensor must be fp16 or fp32"
    assert x.dim() == 2, "Tensor must be 2D"
    M, N = x.shape

    # Compute absmax as fp32 (return this unchanged)
    absmax = torch.max(torch.abs(x)).to(torch.float32).reshape(1)

    # Create output tensor with transposed shape (N, M)
    q_T = torch.empty((N, M), dtype=torch.int8, device=x.device)

    # Get strides
    stride_am = x.stride(0)
    stride_an = x.stride(1)
    stride_bn = q_T.stride(0)  # stride for first dimension (N)
    stride_bm = q_T.stride(1)  # stride for second dimension (M)

    # Launch kernel with 2D grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    _quantize_global_transpose[grid](
        x,
        absmax,
        q_T,
        stride_am,
        stride_an,
        stride_bn,
        stride_bm,
        M,
        N,
    )

    return q_T, absmax


if __name__ == "__main__":
    M, N = 1024, 2048
    x = torch.randn(M, N, dtype=torch.float32).cuda()
    q, absmax_result = quantize_global_transpose(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {q.shape}")
    print(f"Expected output shape: ({N}, {M})")
    print(f"absmax: {absmax_result.item():.6f}")

    # Validate shape is transposed
    assert q.shape == (N, M), f"Expected output shape ({N}, {M}), got {q.shape}"

    # Validate correctness: dequantize and compare with x.T
    denom = max(absmax_result.item(), 127 * 1e-8)
    x_T_dequantized = q.float() * (denom / 127.0)
    x_T_expected = x.T

    max_diff = (x_T_dequantized - x_T_expected).abs().max().item()
    quantization_step = denom / 127.0
    print(f"Max difference (dequantized vs expected): {max_diff:.6f}")
    print(f"Quantization step size: {quantization_step:.6f}")
    # Expected error is up to half the quantization step
    print("✓ Test passed!" if max_diff < quantization_step else "✗ Test failed!")
