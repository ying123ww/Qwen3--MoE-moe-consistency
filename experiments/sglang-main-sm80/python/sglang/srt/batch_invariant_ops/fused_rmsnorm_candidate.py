import json
import os
from pathlib import Path

_MODE = os.environ.get("SLIME_QWEN3_RESIDUAL_RMS_MODE", "off")
if _MODE not in ("off", "shadow", "candidate"):
    raise ValueError(f"Invalid residual RMSNorm mode: {_MODE!r}")
_AUDIT = os.environ.get("SLIME_QWEN3_RESIDUAL_RMS_AUDIT", "1")
if _AUDIT not in ("0", "1"):
    raise ValueError(f"Invalid residual RMSNorm audit setting: {_AUDIT!r}")
_AUDIT_ENABLED = _AUDIT == "1"
if _MODE == "shadow" and not _AUDIT_ENABLED:
    raise ValueError("Residual RMSNorm shadow requires audit")
_AUDIT_DIR = os.environ.get("SLIME_QWEN3_RESIDUAL_RMS_AUDIT_DIR")
if _MODE != "off" and _AUDIT_ENABLED and not _AUDIT_DIR:
    raise ValueError("Residual RMSNorm experiments require an audit directory")
ENABLED = _MODE != "off"
_CALLS = {}
_GRAPH_SETTING = os.environ.get("SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH", "0")
if _GRAPH_SETTING not in ("0", "1"):
    raise ValueError("Invalid residual RMSNorm CUDA Graph setting")
_GRAPH_ENABLED = _GRAPH_SETTING == "1"
if _GRAPH_ENABLED and (_MODE != "candidate" or _AUDIT_ENABLED):
    raise ValueError("Residual RMSNorm CUDA Graph requires unaudited candidate mode")
_GRAPH_CAPTURE_CALLS = 0

"""Opt-in Qwen3 residual RMSNorm fusion; default runtime is unchanged."""
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


def _candidate(x, residual, weight, eps=1e-6, *, allow_cuda_graph=False):
    global _GRAPH_CAPTURE_CALLS
    if not (x.is_cuda and x.device == residual.device == weight.device
            and x.dtype == residual.dtype == torch.bfloat16
            and weight.dtype in (torch.bfloat16, torch.float32)
            and x.ndim == 2 and 1 <= x.shape[0] <= 4 and x.shape[1] == 2048
            and residual.shape == x.shape and weight.shape == (2048,)
            and x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
        raise ValueError('Prototype requires contiguous BF16 M=1..4/H2048 input/residual and BF16/FP32 weights')
    if torch.cuda.is_current_stream_capturing():
        if not allow_cuda_graph:
            raise RuntimeError('Residual RMSNorm CUDA Graph requires explicit opt-in')
        _GRAPH_CAPTURE_CALLS += 1
    output = torch.empty_like(x)
    residual_output = torch.empty_like(residual)
    _fused_residual_rms_kernel[(x.shape[0],)](
        x, residual, weight, output, residual_output, EPS=eps, H=2048, BLOCK=1024,
        num_warps=4, num_stages=3, enable_fp_fusion=False)
    return output, residual_output


def eligible(module, x, residual, post_residual_addition):
    return (getattr(module, "_slime_qwen3_rms_kind", None) in ("input", "pre_mlp")
            and residual is not None and post_residual_addition is None
            and module.hidden_size == 2048 and module.variance_size_override is None
            and not module.cast_x_before_out_mul and not module.fp32_residual
            and module.override_orig_dtype is None
            and x.is_cuda and x.device == residual.device == module.weight.device
            and x.dtype == residual.dtype == torch.bfloat16
            and module.weight.dtype in (torch.bfloat16, torch.float32)
            and x.ndim == 2 and 1 <= x.shape[0] <= 4 and x.shape[1] == 2048
            and residual.shape == x.shape and module.weight.shape == (2048,)
            and x.is_contiguous() and residual.is_contiguous() and module.weight.is_contiguous())


def forward(module, x, residual, post_residual_addition=None):
    if _MODE == "off":
        return module.forward_native(x, residual, post_residual_addition)
    accepted = eligible(module, x, residual, post_residual_addition)
    if _MODE == "candidate" and not _AUDIT_ENABLED:
        if not accepted:
            return module.forward_native(x, residual, post_residual_addition)
        try:
            if _GRAPH_ENABLED:
                return _candidate(x, residual, module.weight, module.variance_epsilon,
                                  allow_cuda_graph=True)
            return _candidate(x, residual, module.weight, module.variance_epsilon)
        except Exception as exc:
            raise RuntimeError(f"Residual RMSNorm candidate failed: layer={module._slime_qwen3_rms_layer} kind={module._slime_qwen3_rms_kind}") from exc
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Residual RMSNorm audit requires CUDA graphs disabled")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    layer, kind = module._slime_qwen3_rms_layer, module._slime_qwen3_rms_kind
    key = (rank, layer, kind)
    _CALLS[key] = _CALLS.get(key, 0) + 1
    row = dict(mode=_MODE, rank=rank, layer=layer, kind=kind, call=_CALLS[key],
               a_shape=list(x.shape), eligible=accepted, compared_elements=0,
               mismatches=0, nonfinite_elements=0)
    if not accepted:
        row["reason"] = "outside_explicit_Qwen3_BF16_residual_small_M_scope"
        result = module.forward_native(x, residual, post_residual_addition)
    else:
        try:
            candidate = _candidate(x, residual, module.weight, module.variance_epsilon)
            if _MODE == "shadow":
                reference = module.forward_native(x, residual, post_residual_addition)
                for actual, expected in zip(candidate, reference):
                    row["compared_elements"] += actual.numel()
                    row["mismatches"] += int((actual.view(torch.int16) != expected.view(torch.int16)).sum().item())
                    row["nonfinite_elements"] += int((~torch.isfinite(actual)).sum().item())
                result = reference
            else:
                result = candidate
        except Exception as exc:
            raise RuntimeError(f"Residual RMSNorm candidate failed: rank={rank} layer={layer} kind={kind}") from exc
    path = Path(_AUDIT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"rank-{rank}-pid-{os.getpid()}.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    if row["mismatches"] or row["nonfinite_elements"]:
        value = x.float() + residual.float()
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import mean_dim
        native_variance = value.pow(2).mean(dim=-1, keepdim=True)
        explicit_variance = mean_dim(value.pow(2), dim=-1, keepdim=True)
        dump = dict(context=row, x=x.detach().cpu(), residual=residual.detach().cpu(),
                    weight=module.weight.detach().cpu(), eps=module.variance_epsilon,
                    native_variance=native_variance.detach().cpu(),
                    explicit_variance=explicit_variance.detach().cpu(),
                    reference=[t.detach().cpu() for t in reference],
                    candidate=[t.detach().cpu() for t in candidate])
        torch.save(dump, path / f"failure-rank-{rank}-layer-{layer}-{kind}-call-{row['call']}.pt")
        raise RuntimeError(f"Residual RMSNorm exact gate FAILED: {json.dumps(row)}")
    return result

