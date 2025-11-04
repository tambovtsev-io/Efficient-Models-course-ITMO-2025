"""
Задача: Global INT8 Quantization (Triton)

Задача: глобальная (per-tensor) симметричная квантизация тензора в int8 на GPU

Дан входной тензор X (fp16/fp32). Нужно вернуть квантизованный тензор Q (int8) и absmax = max(|X|) (скаляр той же плавающей точности, что и X).

Математика:
- absmax = max(|X|)
- absmax_inv = 1.0 / absmax
- Q = round(127.0 * (X * absmax_inv)) с записью в int8

Требования
- Одно Triton-ядро, линейный проход по массиву с маскированием хвоста.
- Использовать tl.load / tl.store.
- Использовать @triton.autotune (как минимум 2 конфига).
- Python-обёртка quantize_global(x) возвращает (q_int8, absmax).
- Вход: CUDA, torch.float16 или torch.float32. Выход: q.dtype == torch.int8, absmax.shape == (1,), absmax.dtype == x.dtype.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_global(x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Triton loading
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Quantization
    dtype = tl.int8
    absmax_inv = tl.load(absmax_inv_ptr)
    q = tl.cast(127.0 * (x * absmax_inv), dtype=dtype)
    tl.store(output_ptr + offsets, q, mask=mask)


def quantize_global(x: torch.Tensor, block_size: int = 1024):
    assert x.is_cuda, "Tensor must be on CUDA"
    assert x.dtype in [torch.float16, torch.float32], "Tensor must be fp16 or fp32"
    # assert x.dim() == 1, "Tensor must be 1D"
    absmax = torch.max(torch.abs(x)).reshape(1)
    absmax_inv = 1.0 / absmax
    q = torch.empty_like(x, dtype=torch.int8).cuda()
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    _quantize_global[grid](x, absmax_inv, q, x.numel())
    return q, absmax


if __name__ == "__main__":
    dtype = torch.float16
    x = torch.randn(1024, dtype=dtype).cuda()
    print(x.shape)
    q, absmax = quantize_global(x)
    print(absmax, q)
    print(absmax.dtype, q.dtype)