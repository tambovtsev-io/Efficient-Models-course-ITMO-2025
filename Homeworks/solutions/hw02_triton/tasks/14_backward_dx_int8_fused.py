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
    pass


def backward_dx_int8_fused(
    dy: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    per_channel: bool = True,
) -> torch.Tensor:
    """Вернуть dX = dY @ (dequant(W_q))^T, dtype fp16, shape (B, IN)."""
    pass
