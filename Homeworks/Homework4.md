# Homework 4 — Quantized Linear in Triton (Oct 20 → Nov 10)

**Timeline:** three weeks — **Oct 20** to **Nov 10**  
**Goal:** implement an INT8 *quantized* linear layer with Triton:
- symmetric linear quantization to int8 with a Triton,
- **forward** in int8 with a Triton matmul that **fuses dequantization**,
- **backward** (grad w.r.t. input, weight, bias) implemented in Triton,

---

## What you will build

Implement a custom autograd op and module:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_global_triton(
    fp_ptr,               # *const* pointer to input FP tensor data, row-major
                          #   dtype: fp16/fp32 supported by tl.load
                          #   shape: (N_ROWS, N_COLS)
    q_ptr,                # pointer to output INT8 tensor (same shape as fp_ptr)
                          #   dtype: int8
    scale_out_ptr,        # pointer to single fp32 scale value to be written:
                          #   scale_out_ptr[0] = max(abs(fp_ptr)) / 127  (clamped >= 1e-8)
    N_ROWS,               # int: number of rows in the 2D matrix
    N_COLS: tl.constexpr  # constexpr int: number of columns in the 2D matrix
):
    """
    Symmetric per-tensor quantization (zero-point = 0).

    Contract:
      - Let s = max(|X|)/127 (X is the full tensor), clamped to >= 1e-8.
      - For every element: Q = clamp(round(X / s), -127, 127).to(int8).
      - Write s to scale_out_ptr[0] (fp32).
      - X layout is row-major (index = row * N_COLS + col).
    """

    raise NotImplementedError


@triton.jit
def _quantize_rowwise_triton(
    fp_ptr,            # *const* pointer to input FP tensor data, row-major
                       #   dtype: fp16/fp32 supported by tl.load
                       #   shape: (N_ROWS, N_COLS)
    q_ptr,             # pointer to output INT8 tensor (same shape as fp_ptr)
                       #   dtype: int8
    scale_row_ptr,     # pointer to row-wise scales (fp32) of shape (N_ROWS,)
                       #   scale_row_ptr[row] = max(abs(fp_ptr[row, :])) / 127
    N_ROWS,            # int: number of rows
    N_COLS: tl.constexpr  # constexpr int: number of columns
):
    """
    Symmetric per-row quantization (zero-point = 0).

    Typical usage for weights:
      - To get per-OUT-channel scales for (IN, OUT) weights, quantize W^T as (OUT, IN)
        so each OUT-channel becomes a row.

    Contract for each row r:
      - s_r = max(|X[r, :]|)/127 (clamped to >= 1e-8)
      - Q[r, c] = clamp(round(X[r, c]/s_r), -127, 127).to(int8)
      - Write s_r to scale_row_ptr[r] (fp32).
    """

    raise NotImplementedError


@triton.jit
def _forward_triton(
    x_q_ptr,           # pointer to int8 input activations X_q, shape (B, IN), row-major
                       #   dtype: int8
    x_scale_ptr,       # pointer to scalar fp32 scale for X (per-tensor):
                       #   x_scale_ptr[0] = s_x
    w_q_ptr,           # pointer to int8 weights W_q, shape (IN, OUT), row-major
                       #   dtype: int8
    w_scale_ptr,       # pointer to fp32 scales for W:
                       #   if PER_CHANNEL==1: shape (OUT,) per-output-channel scales s_w[j]
                       #   else (PER_CHANNEL==0): scalar s_w at w_scale_ptr[0]
    b_ptr,             # pointer to optional bias (fp16) shape (OUT,), or 0 if no bias
    y_ptr,             # pointer to output activations Y (fp16), shape (B, OUT), row-major
    B,                 # int: batch size (rows of X / Y)
    IN,                # int: input features (K dimension)
    OUT,               # int: output features (N dimension)
    BLOCK_M: tl.constexpr,  # tile size along M=B
    BLOCK_N: tl.constexpr,  # tile size along N=OUT
    BLOCK_K: tl.constexpr,  # tile size along K=IN
    PER_CHANNEL: tl.constexpr  # 1 if per-OUT-channel scales for W, else 0 (per-tensor)
):
    """
    Computes:
      Y = dequant(X_q) @ dequant(W_q) + bias, with fusion of dequantization into GEMM.

    Math:
      - int32 accumulator: ACC = Σ_k int(X_q) * int(W_q)
      - scale per output column j: scale_j = s_x * (s_w[j] if PER_CHANNEL else s_w)
      - Y[:, j] = (ACC[:, j].to(fp32) * scale_j).to(fp16) + bias[j] (if present)

    Layouts:
      - X_q: row-major (B, IN)
      - W_q: row-major (IN, OUT)
      - Y:   row-major (B, OUT)

    Notes:
      - Use masking for edge tiles.
      - Load int8 tiles, cast to int32 for dot.
      - Apply scales only once after K-loop (fused dequant).
      - Add bias inside the kernel for best fusion.
    """

    raise NotImplementedError

@triton.jit
def _backward_dx_triton(
    dy_ptr,            # pointer to upstream grad dY (fp16), shape (B, OUT), row-major
    w_q_ptr,           # pointer to int8 weights W_q, shape (IN, OUT), row-major
    w_scale_ptr,       # pointer to scales for W (fp32):
                       #   if PER_CHANNEL==1: (OUT,) per-output-channel
                       #   else (scalar at w_scale_ptr[0])
    dx_ptr,            # pointer to output grad dX (fp16), shape (B, IN), row-major
    B,                 # int: batch size
    IN,                # int: input features
    OUT,               # int: output features
    BLOCK_M: tl.constexpr,  # tile size along M=B
    BLOCK_N: tl.constexpr,  # tile size along N=IN
    BLOCK_K: tl.constexpr,  # tile size along K=OUT
    PER_CHANNEL: tl.constexpr  # 1: per-OUT scales, 0: scalar scale
):
    """
    Computes:
      dX = dY @ (dequant(W_q))^T

    Math:
      - W_deq = W_q.to(fp32) * (s_w[j] per column j OR scalar s_w)
      - dX = dY (B, OUT) @ W_deq^T (OUT, IN) → (B, IN)

    Requirements:
      - Stream tiles of dY and W_q^T.
      - On-the-fly dequant of W_q tiles (multiply by s_w or s_w[j]).
      - Accumulate in fp32, store fp16 to dx_ptr.
    """

    raise NotImplementedError


@triton.jit
def _backward_dw_triton(
    x_q_ptr,           # pointer to int8 inputs X_q, shape (B, IN), row-major
    x_scale_ptr,       # pointer to scalar fp32 scale: x_scale_ptr[0] = s_x
    dy_ptr,            # pointer to upstream grad dY (fp16), shape (B, OUT), row-major
    dw_ptr,            # pointer to output grad dW (fp16), shape (IN, OUT), row-major
    B,                 # int: batch size
    IN,                # int: input features
    OUT,               # int: output features
    BLOCK_M: tl.constexpr,  # tile size along M=IN
    BLOCK_N: tl.constexpr,  # tile size along N=OUT
    BLOCK_K: tl.constexpr   # tile size along K=B
):
    """
    Computes:
      dW = (dequant(X_q))^T @ dY

    Math:
      - X_deq = X_q.to(fp32) * s_x (scalar per-tensor scale)
      - dW = X_deq^T (IN, B) @ dY (B, OUT) → (IN, OUT)

    Requirements:
      - Stream tiles of X_q^T and dY.
      - On-the-fly dequant of X_q tiles (multiply by s_x).
      - Accumulate in fp32, store fp16 to dw_ptr.
    """

    raise NotImplementedError


@triton.jit
def _backward_db_triton(
    dy_ptr,            # pointer to upstream grad dY (fp16), shape (B, OUT), row-major
    db_ptr,            # pointer to output grad dB (fp16), shape (OUT,)
    B,                 # int: batch size
    OUT,               # int: output features
    BLOCK_N: tl.constexpr,  # tile size along N=OUT
    BLOCK_M: tl.constexpr   # tile size along M=B (reduction chunk)
):
    """
    Computes:
      dB = sum(dY, dim=0)  → shape (OUT,)

    Requirements:
      - Reduce along batch dimension in chunks of BLOCK_M.
      - Accumulate in fp32, store fp16 to db_ptr.
      - Use masking for tail tiles along both dimensions.
    """

    raise NotImplementedError

class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_fp16,          # torch.Tensor, CUDA, dtype=torch.float16, shape (B, IN)
        w_fp16,          # torch.Tensor, CUDA, dtype=torch.float16, shape (IN, OUT)
        b_fp16=None,     # Optional[torch.Tensor], CUDA, dtype=torch.float16, shape (OUT,)
        *,
        per_channel=True # bool: if True — per-OUT-channel quant for W; else per-tensor
    ):
        """
        Returns:
          y_fp16: torch.Tensor, CUDA, dtype=torch.float16, shape (B, OUT)

        Must:
          - Quantize X per-tensor in Triton → (x_q:int8, s_x:fp32 scalar)
          - Quantize W per-OUT-channel (if per_channel) or per-tensor (if not) in Triton
            → (w_q:int8, s_w: fp32 scalar or fp32 (OUT,))
          - Call _forward_triton to compute Y with fused dequant & optional bias
          - Save (x_q, x_scale, w_q, w_scale, b or None) + shape/per_channel in ctx for backward
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx, dy):
        """
        Args:
          dy: torch.Tensor, CUDA, dtype=torch.float16, shape (B, OUT)

        Returns:
          dx: torch.Tensor, CUDA, dtype=torch.float16, shape (B, IN)
          dw: torch.Tensor, CUDA, dtype=torch.float16, shape (IN, OUT)
          db: Optional[torch.Tensor], CUDA, dtype=torch.float16, shape (OUT,) or None
          None (for per_channel kwarg)

        Must:
          - Use _backward_dx_triton (with on-the-fly dequant of W_q)
          - Use _backward_dw_triton (with on-the-fly dequant of X_q)
          - Use _backward_db_triton for bias reduction if bias was present
        """
        raise NotImplementedError


class QuantizedLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,     # int: number of input features (IN)
        out_features,    # int: number of output features (OUT)
        bias=True,       # bool: whether to include bias parameter (shape OUT)
        *,
        per_channel=True # bool: whether to use per-OUT-channel quantization for W
    ):
        """
        Buffers/Parameters to create:
          - self.weight: torch.nn.Parameter, shape (IN, OUT), dtype fp16, CUDA
          - self.bias:   Optional[torch.nn.Parameter], shape (OUT,), dtype fp16, CUDA
          - store flags: self.in_features, self.out_features, self.per_channel

        Init tips:
          - torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
          - if bias: uniform_(-1/sqrt(IN), 1/sqrt(IN))
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        """
        Args:
          x: torch.Tensor, CUDA, dtype=torch.float16, shape (B, IN)

        Returns:
          y: torch.Tensor, CUDA, dtype=torch.float16, shape (B, OUT)

        Must:
          return CustomLinearFunction.apply(x, self.weight, self.bias, per_channel=self.per_channel)
        """
        raise NotImplementedError


        
```
