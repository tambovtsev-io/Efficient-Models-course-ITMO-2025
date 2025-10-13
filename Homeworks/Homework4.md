# HW4 — Triton INT8 (Quantization & MatMul)

This homework asks you to implement several **Triton** GPU kernels for INT8 quantization and matrix ops with fused dequantization. Submissions are validated by automated tests and scored per test.

> **Deadline (MSK):** **10 Nov 2025, 23:59 (UTC+3)**


## Task List

### 10) `10_quantize_global_int8` — Global INT8 Quantization (Triton) — **2.0 pts, up to 50 attempts**

**Goal**: Global (per-tensor) **symmetric** quantization of input tensor `X` into `int8`.

**Return**: `(Q: int8, absmax: scalar same dtype as X)`.

**Math**

* `absmax = max(|X|)`
* `Q = round(127.0 * (X / absmax))` written as `int8`

**Kernel**

* Single pass over the linear buffer with tail masking.
* Use `tl.load` / `tl.store`.
* **Autotune** with at least 2 configs.

**Allowed imports**: `torch`, `triton`, `triton.language`.

**Signature skeleton**

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_global(x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # YOUR CODE HERE


def quantize_global(x: torch.Tensor):
    # YOUR CODE HERE
```

---

### 11) `11_quantize_rowwise_int8` — Rowwise INT8 Quantization (Triton) — **3.0 pts, up to 50 attempts**

**Goal**: Per-row symmetric quantization for a 2D tensor `X[N_ROWS, N_COLS]` (row-major, contiguous; fp16/fp32 on CUDA).

**Return**: `(Q: int8[N_ROWS, N_COLS], absmaxs: fp16[N_ROWS])`.

**Math** (for each row `r`)

* `a_r = max_c |X[r, c]|`
* `Q[r, c] = round(127.0 * X[r, c] / a_r)` written as `int8`

**Kernel**

* **One program block per row**.
* Extend the row width to the next power-of-two `P2` (mask the tail).
* Use `tl.load` / `tl.store` and column masking.
* **Autotune** with multiple configs.

**Allowed imports**: `torch`, `triton`, `triton.language`, `math`.

**Signature skeleton**

```python
import torch, math
import triton
import triton.language as tl

@triton.autotune(
    # YOUR CONFIGS HERE
)
@triton.jit
def _quantize_rowwise(x_ptr, output_ptr, output_maxs, n_elements,
                      BLOCK_SIZE: tl.constexpr, P2: tl.constexpr):
    # YOUR CODE HERE


def quantize_rowwise(x: torch.Tensor):
    # YOUR CODE HERE
```

---

### 12) `12_matmul_int8_fused_dequant` — INT8 MatMul + Fused Dequant & Bias — **4.0 pts, up to 50 attempts**

**Inputs**

* `X_q: (B, IN)` — int8 activations
* `W_q: (IN, OUT)` — int8 weights
* `s_x` — fp32 scalar scale (per-tensor for inputs)
* `s_w` — fp32 scales for weights: either scalar (per-tensor) or vector `(OUT,)` (per-OUT-channel)
* `bias` — optional fp16 vector `(OUT,)`

**Output**: `Y: (B, OUT)`, `fp16`.

**Computation**

1. Accumulate `int8 * int8` products into `int32`:
   `ACC[b, o] = sum_k X_q[b, k] * W_q[k, o]`.
2. Column scale: `alpha[o] = s_x * (s_w[o] if per-channel else s_w)`.
3. After the `K` loop, produce: `Y[b, o] = fp16( fp32(ACC[b, o]) * alpha[o] + bias[o] )`.

**Kernel**

* Load int8 tiles of `X_q` and `W_q`, accumulate in `int32`.
* Apply scaling **once after** the K-loop (fused dequant), add `bias` in-kernel, store `fp16`.
* Support both per-channel and per-tensor weight scales via `PER_CHANNEL` constexpr flag.
* Mask tails across `B`, `IN`, `OUT`.
* **Autotune** with ≥3 configs (vary `BLOCK_M/N/K`, `num_warps`, `num_stages`).

**Allowed imports**: `torch`, `triton`, `triton.language`.

**Signature skeleton**

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=2)
    ],
    key=["B", "IN", "OUT"],
)
@triton.jit
def _forward_int8_fused_kernel(x_q_ptr, x_scale_ptr,
                               w_q_ptr, w_scale_ptr,
                               b_ptr, y_ptr,
                               B, IN, OUT,
                               BLOCK_M: tl.constexpr,
                               BLOCK_N: tl.constexpr,
                               BLOCK_K: tl.constexpr,
                               PER_CHANNEL: tl.constexpr):
    # YOUR CODE HERE


def matmul_int8_fused(x_q: torch.Tensor,
                      x_scale: torch.Tensor,
                      w_q: torch.Tensor,
                      w_scale: torch.Tensor,
                      bias: torch.Tensor | None = None,
                      *, per_channel: bool = True) -> torch.Tensor:
    """Return Y = dequant(X_q) @ dequant(W_q) + bias, dtype fp16, shape (B, OUT)."""
    # YOUR CODE HERE
```

---

### 13) `13_quantize_global_transpose_int8` — Global INT8 Quantize **+ Transpose** (Triton) — **2.0 pts, up to 50 attempts**

**Goal**: For 2D `X[M, N]` (fp16/fp32), compute `absmax = max(|X|)`, quantize `Q = round(127*X/denom)`, and **store transposed** into `B[N, M]` (`int8`). Return `(B, absmax_fp32[1])`. Use `denom = max(absmax, 127*1e-8)` to avoid division by zero; return **original** `absmax`.

**Kernel**

* One kernel that loads tiles from `A (X)`, quantizes, and writes into `B` with transposed indices.
* Use `tl.load` / `tl.store` and edge masking.
* Support compatible strides: each tensor must have at least one unit-stride dimension.
* **Autotune** with ≥3 configs (`BLOCK_M/N`, `GROUP_M`, and different warps/stages).

**Allowed imports**: `torch`, `triton`, `triton.language`.

**Signature skeleton**

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "GROUP_M": 8}, num_warps=4, num_stages=2)
    ],
    key=["M", "N"],
)
@triton.jit
def _quantize_global_transpose(A, absmax_inv_ptr, B,
                              stride_am, stride_an,
                              stride_bn, stride_bm,
                              M, N,
                              BLOCK_M: tl.constexpr,
                              BLOCK_N: tl.constexpr,
                              GROUP_M: tl.constexpr):
    # YOUR CODE HERE


def quantize_global_transpose(x: torch.Tensor):
    """Return (q_T:int8[N,M], absmax:fp32[1])."""
    # YOUR CODE HERE
```

---

### 14) `14_backward_dx_int8_fused` — INT8 Backward dX with Fused Dequant (Triton) — **3.5 pts, up to 50 attempts**

**Goal**: Compute `dX = dY @ (dequant(W_q))^T` with accumulation in `fp32` and output in `fp16`.

**Inputs**

* `dY: (B, OUT)` — `fp16`
* `W_q: (IN, OUT)` — `int8`
* `s_w` — weight scales: scalar (per-tensor) or vector `(OUT,)` (per-OUT-channel)

**Computation**

* `W_q_deq[:, o] = W_q[:, o] * s_w[o]` (or `* s_w_scalar`)
* `dX[b, i] = sum_k dY[b, k] * W_q_deq[i, k]`

**Kernel**

* Load `dY` tiles (`fp16`) and `W_q` tiles (`int8`), accumulate in `fp32`.
* Apply weight scales once per tile (either inside the K-loop or right after, but not redundantly per partial sum).
* Mask tails in all dimensions.
* **Autotune** with ≥3 configs; support `PER_CHANNEL` constexpr flag.

**Allowed imports**: `torch`, `triton`, `triton.language`.

**Signature skeleton**

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=2)
    ],
    key=["B", "IN", "OUT"],
)
@triton.jit
def _backward_dx_fused_kernel(dy_ptr, wq_ptr, w_scale_ptr, dx_ptr,
                              B, IN, OUT,
                              BLOCK_M: tl.constexpr,
                              BLOCK_N: tl.constexpr,
                              BLOCK_K: tl.constexpr,
                              PER_CHANNEL: tl.constexpr):
    # YOUR CODE HERE


def backward_dx_int8_fused(dy: torch.Tensor,
                           w_q: torch.Tensor,
                           w_scale: torch.Tensor,
                           *, per_channel: bool = True) -> torch.Tensor:
    """Return dX = dY @ (dequant(W_q))^T, dtype fp16, shape (B, IN)."""
    # YOUR CODE HERE
```

---


