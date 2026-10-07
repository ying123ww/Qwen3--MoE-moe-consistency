"""Opt-in Qwen3 FP32 router tile experiment; the production default is unchanged."""
import json
import os
from pathlib import Path

import torch

_MODE = os.environ.get("SLIME_QWEN3_ROUTER_TILE_MODE", "off")
if _MODE not in ("off", "shadow", "candidate"):
    raise ValueError(f"Invalid SLIME_QWEN3_ROUTER_TILE_MODE: {_MODE!r}")
_AUDIT_DIR = os.environ.get("SLIME_QWEN3_ROUTER_TILE_AUDIT_DIR")
if _MODE != "off" and not _AUDIT_DIR:
    raise ValueError("Router tile experiments require SLIME_QWEN3_ROUTER_TILE_AUDIT_DIR")
_CALLS = {}


def eligible_shape(a_shape, b_shape):
    # Deliberately limited to the Qwen3-30B-A3B router measured in the pilot.
    return (len(a_shape) == len(b_shape) == 2 and
            0 < a_shape[0] <= 16 and a_shape[1] == b_shape[0] == 2048 and
            b_shape[1] == 128)


def _candidate(a, b):
    import triton
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_kernel_persistent

    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (min(sms, triton.cdiv(m, 16) * triton.cdiv(n, 64)),)
    matmul_kernel_persistent[grid](
        a, b, out, None, m, n, k, *a.stride(), *b.stride(), *out.stride(),
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8,
        NUM_SMS=sms, A_LARGE=a.numel() > 2**31, B_LARGE=b.numel() > 2**31,
        C_LARGE=out.numel() > 2**31, HAS_BIAS=False, num_stages=3, num_warps=4,
    )
    return out


def router_matmul(a, b, *, layer_id):
    if _MODE == "off":
        return torch.mm(a, b)
    if a.dtype != torch.float32 or b.dtype != torch.float32 or not a.is_cuda or a.device != b.device:
        raise RuntimeError("Router tile experiment requires FP32 CUDA operands on one device")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Router tile audit requires disable_cuda_graph=True")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    key = (rank, layer_id)
    _CALLS[key] = _CALLS.get(key, 0) + 1
    eligible = eligible_shape(a.shape, b.shape)
    row = dict(mode=_MODE, rank=rank, layer=int(layer_id), call=_CALLS[key],
               a_shape=list(a.shape), b_shape=list(b.shape), eligible=eligible,
               compared_elements=0, mismatches=0)
    if not eligible:
        row["reason"] = "outside_explicit_small_M_Qwen3_router_scope"
        result = torch.mm(a, b)
    else:
        try:
            result = _candidate(a, b)
            if _MODE == "shadow":
                reference = torch.mm(a, b)
                row["compared_elements"] = result.numel()
                row["mismatches"] = int((result.view(torch.int32) != reference.view(torch.int32)).sum().item())
                row["nonfinite_elements"] = int((~torch.isfinite(result)).sum().item())
                # Shadow keeps the reference data flow after checking the candidate.
                result = reference
        except Exception as exc:
            raise RuntimeError(f"Router tile candidate failed: rank={rank} layer={layer_id} shape={list(a.shape)}") from exc
    path = Path(_AUDIT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"rank-{rank}-pid-{os.getpid()}.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    if row["mismatches"] or row.get("nonfinite_elements", 0):
        raise RuntimeError(f"Router tile exact gate FAILED: {json.dumps(row)}")
    return result
