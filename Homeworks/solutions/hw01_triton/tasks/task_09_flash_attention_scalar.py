"""
Task 9: Simple FlashAttention (Scalar)

Реализуйте простую версию FlashAttention для одного запроса.

Даны тензоры:
- q формы (T,) — запрос;
- k формы (T,) — ключи;
- v формы (T,) — значения.

Нужно вычислить:

s_j = q * k[j]
soft = softmax(s)
res = sum_j soft[j] * v[j]


Требования:
- процессить последовательность тайлами B1 < T в одном ядре (один pid0 = одна голова);
- использовать стабильную форму softmax с трюком *running max / renormalization*:

  m_new = max(m, s_j)
  l = l * 2**(m - m_new) + 2**(s_j - m_new)
  acc = acc * 2**(m - m_new) + v_j * 2**(s_j - m_new)
  m = m_new

- результат = acc / l.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flashatt_kernel(q_ptr, k_ptr, v_ptr, out_ptr, T, B0: tl.constexpr, B1: tl.constexpr):
    # Streaming softmax with running max / renormalization over tiles
    LOG2E = 1.4426950408889634  # log2(e)

    # Indices for a block of queries (rows i)
    pid = tl.program_id(0)
    offs_i = pid * B0 + tl.arange(0, B0)
    mask_i = offs_i < T

    # Load queries for this block
    q_i = tl.load(q_ptr + offs_i, mask=mask_i, other=0.0)  # [B0]

    # Running stats per query in the block
    m = tl.full((B0,), -float("inf"), dtype=tl.float32)  # [B0]
    l = tl.zeros((B0,), dtype=tl.float32)                # [B0]
    acc = tl.zeros((B0,), dtype=tl.float32)              # [B0]

    # Iterate over keys/values in tiles of size B1 (columns j)
    for start in range(0, T, B1):
        offs_j = start + tl.arange(0, B1)               # [B1]
        mask_j = offs_j < T

        k_j = tl.load(k_ptr + offs_j, mask=mask_j, other=0.0)  # [B1]
        v_j = tl.load(v_ptr + offs_j, mask=mask_j, other=0.0)  # [B1]

        # Scores s[i,j] = q[i] * k[j]
        s = q_i[:, None] * k_j[None, :]                         # [B0, B1]
        valid = mask_i[:, None] & mask_j[None, :]

        # Update running max per row
        tile_max = tl.max(tl.where(valid, s, -float("inf")), axis=1)  # [B0]
        m_new = tl.maximum(m, tile_max)                                # [B0]
        alpha = tl.exp2((m - m_new) * LOG2E)                           # [B0]

        # Exponentials for this tile
        exps = tl.where(valid, tl.exp2((s - m_new[:, None]) * LOG2E), 0.0)  # [B0, B1]

        # Update denominator and numerator
        l = l * alpha + tl.sum(exps, axis=1)
        acc = acc * alpha + tl.sum(exps * v_j[None, :], axis=1)
        m = m_new

    # Final output per query: acc / l
    out_i = acc / l
    tl.store(out_ptr + offs_i, out_i, mask=mask_i)


def flashatt_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 1 and k.dim() == 1 and v.dim() == 1
    T = q.numel()
    assert k.numel() == T and v.numel() == T
    B0 = 128
    B1 = 128
    out = torch.empty(T, device=q.device, dtype=q.dtype)
    grid = (triton.cdiv(T, B0),)
    _flashatt_kernel[grid](q, k, v, out, T, B0, B1)
    return out


if __name__ == "__main__":
    # Simple correctness check vs PyTorch reference
    T = 257  # not divisible by B1 to test tail masking
    device = "cuda"
    dtype = torch.float32

    q = torch.randn(T, device=device, dtype=dtype)
    k = torch.randn(T, device=device, dtype=dtype)
    v = torch.randn(T, device=device, dtype=dtype)

    out = flashatt_triton(q, k, v)

    # PyTorch reference: out[i] = sum_j softmax(q[i]*k[j]) * v[j]
    s = q[:, None] * k[None, :]
    weights = torch.softmax(s, dim=1)
    ref_vec = (weights * v[None, :]).sum(dim=1)

    max_err = (out - ref_vec).abs().max().item()
    print("out[:8]", out[:8])
    print("ref[:8]", ref_vec[:8])
    print("max_abs_err:", max_err)
