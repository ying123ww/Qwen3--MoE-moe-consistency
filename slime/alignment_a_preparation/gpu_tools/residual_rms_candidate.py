"""Standalone exact-candidate prototype; not registered in the model runtime."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _fused_residual_rms_kernel(X, R, W, Y, RES, EPS: tl.constexpr,
                               H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # The original FP32 mean reads four contiguous values per thread.
    # BF16 loads otherwise select eight and change the reduction tree.
    offsets = tl.max_contiguous(tl.arange(0, BLOCK), 4)
    acc = tl.full((), 0.0, tl.float32)
    # Match mean_dim's two 1024-element sums and their ordered accumulation.
    for start in range(0, H, BLOCK):
        index = row * H + start + offsets
        value = tl.load(X + index).to(tl.float32) + tl.load(R + index).to(tl.float32)
        squared = value * value
        acc = acc + tl.sum(squared, 0)
    mean = acc / H
    inverse = libdevice.rsqrt(mean + EPS)
    for start in range(0, H, BLOCK):
        index = row * H + start + offsets
        value = tl.load(X + index).to(tl.float32) + tl.load(R + index).to(tl.float32)
        weight = tl.load(W + start + offsets).to(tl.float32)
        normalized = value * inverse
        weighted = normalized * weight
        tl.store(Y + index, weighted.to(Y.dtype.element_ty))
        tl.store(RES + index, value.to(RES.dtype.element_ty))


def candidate(x, residual, weight, eps=1e-6):
    if not (x.is_cuda and x.device == residual.device == weight.device
            and x.dtype == residual.dtype == torch.bfloat16
            and weight.dtype in (torch.bfloat16, torch.float32)
            and x.ndim == 2 and 1 <= x.shape[0] <= 4 and x.shape[1] == 2048
            and residual.shape == x.shape and weight.shape == (2048,)
            and x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
        raise ValueError('Prototype requires contiguous BF16 M=1..4/H2048 input/residual and BF16/FP32 weights')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Prototype has no CUDA graph gate')
    output = torch.empty_like(x)
    residual_output = torch.empty_like(residual)
    _fused_residual_rms_kernel[(x.shape[0],)](
        x, residual, weight, output, residual_output, EPS=eps, H=2048, BLOCK=1024,
        num_warps=4, num_stages=3, enable_fp_fusion=False)
    return output, residual_output


def reference(x, residual, weight, eps=1e-6):
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import mean_dim
    value = x.float() + residual.float()
    visible_residual = value.to(x.dtype)
    variance = mean_dim(value.pow(2), dim=-1, keepdim=True)
    normalized = value * torch.rsqrt(variance + eps)
    return (normalized * weight).to(x.dtype), visible_residual

