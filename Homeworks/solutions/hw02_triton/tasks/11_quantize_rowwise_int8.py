"""
Задача: Rowwise INT8 Quantization (Triton)
(дедлайн МСК: 10.11.2025 23:59 MSK)

Задача: построчная (per-row) симметричная квантизация

Дан 2D тензор X формы (N_ROWS, N_COLS) (fp16/fp32, CUDA, row-major, contiguous). Требуется вернуть квантизованный тензор Q:int8 той же формы и вектор absmaxs длины N_ROWS с максимумами по модулю для каждой строки.

Математика
Для каждой строки r:
- a_r = max_c |X[r, c]| — максимум по модулю в строке (без clamp);
- Q[r, c] = round(127.0 * X[r, c] / a_r) с записью в int8.

Требования
- Одно Triton-ядро, одна строка на программный блок. Ширину строки расширяем до ближайшей степени двойки P2 для удобной редукции (маскируем хвост).
- Использовать tl.load / tl.store и маскирование по столбцам.
- Обязательно использовать @triton.autotune с несколькими конфигурациями (см. сигнатуру ниже).
- Python-обёртка quantize_rowwise(x) возвращает (q_int8, absmaxs), где absmaxs.dtype == torch.float16 и absmaxs.shape == (N_ROWS,).
"""

import torch, math
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.heuristics(
    values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['n_elements'])}
)
@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    # One program per row
    pid = tl.program_id(axis=0)

    # Calculate row offset and column offsets
    # BLOCK_SIZE is automatically set to next_power_of_2(n_elements) by heuristics
    row_start = pid * n_elements  # n_elements here is actually n_cols per row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_elements

    # Load data for this row
    offsets = row_start + col_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load the absmax for this specific row
    absmax = tl.load(output_maxs + pid)

    # Quantization
    dtype = tl.int8
    q = tl.cast(127.0 * (x / absmax), dtype=dtype)
    tl.store(output_ptr + offsets, q, mask=mask)



def quantize_rowwise(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA"
    assert x.dtype in [torch.float16, torch.float32], "Tensor must be fp16 or fp32"
    assert x.dim() == 2, "Tensor must be 2D"
    n_rows, n_cols = x.shape

    # Calculate P2: next power of 2 >= n_cols (for efficient reduction)
    P2 = 2 ** math.ceil(math.log2(n_cols))

    absmaxs = x.abs().max(dim=1).values.to(torch.float16)
    q = torch.empty_like(x, dtype=torch.int8).cuda()
    # Launch one program per row (as per requirements)
    grid = lambda meta: (n_rows,)
    _quantize_rowwise[grid](x, q, absmaxs, n_cols, P2=P2)
    return q, absmaxs


if __name__ == "__main__":
    dtype = torch.float16
    x = torch.randn(3072, 5120, dtype=dtype).cuda()
    print(f"Input shape: {x.shape}")

    # Quantize
    q, absmaxs = quantize_rowwise(x)
    print(f"absmaxs dtype: {absmaxs.dtype}, shape: {absmaxs.shape}")
    print(f"q dtype: {q.dtype}, shape: {q.shape}")

    # Dequantize: x_approx = q * absmax / 127.0 (broadcast absmaxs across columns)
    x_dequant = q.float() * (absmaxs.unsqueeze(1) / 127.0)

    # Compute error metrics
    abs_error = torch.abs(x - x_dequant)
    max_error = abs_error.max().item()
    mean_error = abs_error.mean().item()

    # Relative error (avoid division by zero)
    x_abs = torch.abs(x)
    mask = x_abs > 1e-6  # Only compute relative error for non-tiny values
    if mask.any():
        relative_error = (abs_error[mask] / x_abs[mask]).mean().item()
    else:
        relative_error = 0.0

    print(f"\nQuantization Quality:")
    print(f"  Max absolute error: {max_error:.6f}")
    print(f"  Mean absolute error: {mean_error:.6f}")
    print(f"  Mean relative error: {relative_error:.6f}")
    print(f"  Original range: [{x.min().item():.4f}, {x.max().item():.4f}]")