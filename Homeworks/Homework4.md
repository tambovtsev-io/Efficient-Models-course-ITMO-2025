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

class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_fp16, w_fp16, b_fp16=None, *, per_channel=True):
        ...

    @staticmethod
    def backward(ctx, dy):
        ...

class QuantizedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, *, per_channel=True):
        super().__init__()
        ...

    def forward(self, x):
        return CustomLinearFunction.apply(x, self.weight, self.bias, per_channel=self.per_channel)
        
```
