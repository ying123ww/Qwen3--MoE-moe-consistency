"""Opt-in experiment hook; upstream model/norm files stay untouched."""
from collections import Counter
from functools import wraps
import json
from pathlib import Path

import torch

_INSTALLED = None
_COUNTS = Counter()
_FIRST = False


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def eligible(q, k, q_norm, k_norm, head_dim, allow_cuda_graph=False):
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.runtime_context import get_exec
    return (is_batch_invariant_mode_enabled()
            and get_exec().deterministic.rl_on_policy_target != 'fsdp'
            and not q_norm.cast_x_before_out_mul and not k_norm.cast_x_before_out_mul
            and q_norm.variance_size_override is None and k_norm.variance_size_override is None
            and q_norm.override_orig_dtype is None and k_norm.override_orig_dtype is None
            and q.is_cuda and q.device == k.device == q_norm.weight.device == k_norm.weight.device
            and q.ndim == k.ndim == 2 and q.shape[0] == k.shape[0] and 1 <= q.shape[0] <= 4
            and q.dtype == k.dtype == torch.bfloat16 and head_dim == 128
            and q.stride(1) == k.stride(1) == 1
            and q_norm.weight.dtype == k_norm.weight.dtype
            and q_norm.weight.dtype in (torch.bfloat16,torch.float32)
            and q_norm.weight.is_contiguous() and k_norm.weight.is_contiguous()
            and (allow_cuda_graph or not torch.cuda.is_current_stream_capturing()))


def install(config):
    global _INSTALLED
    mode = config['mode']
    if mode not in ('off','shadow','joint','strided'):
        raise ValueError('Unknown Q/K runtime experiment mode')
    audit_dir = Path(config['audit_dir']) if config.get('audit_dir') else None
    allow_cuda_graph = config.get('allow_cuda_graph', False)
    if allow_cuda_graph and (mode not in ('joint', 'strided') or audit_dir is not None):
        raise ValueError('Q/K CUDA Graph requires unaudited joint/strided mode')
    if mode == 'shadow' and audit_dir is None:
        raise ValueError('Shadow gate requires a distinct audit directory')
    if _INSTALLED is not None:
        if _INSTALLED != mode:
            raise RuntimeError('Do not switch experiment modes inside an Engine')
        return lambda module,inputs,output:None
    import sglang.srt.models.qwen3_moe as model
    import qk_norm_prototype as prototype
    original = model.apply_qk_norm
    attention = model.Qwen3MoeAttention.apply_qk_norm_rope

    @wraps(attention)
    def identify(module,qkv,positions,forward_batch):
        module.q_norm._p1_qk_runtime_layer = module.attn.layer_id
        module.k_norm._p1_qk_runtime_layer = module.attn.layer_id
        return attention(module,qkv,positions,forward_batch)

    @wraps(original)
    def wrapped(q,k,q_norm,k_norm,head_dim,*args,**kwargs):
        global _FIRST
        if mode == 'off':
            return original(q,k,q_norm,k_norm,head_dim,*args,**kwargs)
        supported = eligible(q,k,q_norm,k_norm,head_dim,allow_cuda_graph)
        if not supported:
            if audit_dir:
                _COUNTS['fallback'] += 1
            return original(q,k,q_norm,k_norm,head_dim,*args,**kwargs)
        selected = 'joint' if mode == 'shadow' else mode
        if allow_cuda_graph and torch.cuda.is_current_stream_capturing():
            _COUNTS['captured'] += 1
        result = prototype.forward(q,k,q_norm.weight,k_norm.weight,head_dim,
                                   q_norm.variance_epsilon,k_norm.variance_epsilon,selected)
        if not _FIRST:
            print('QK_NORM_CANDIDATE_FIRST_USE '+json.dumps(dict(mode=mode,q_shape=list(q.shape),k_shape=list(k.shape),
                  rank=torch.distributed.get_rank(),layer=q_norm._p1_qk_runtime_layer)),flush=True)
            _FIRST = True
        if mode != 'shadow':
            return result
        reference = original(q,k,q_norm,k_norm,head_dim,*args,**kwargs)
        match = all(bits_equal(a,b) for a,b in zip(result,reference))
        _COUNTS['eligible'] += 1
        _COUNTS['compared_elements'] += q.numel()+k.numel()
        _COUNTS[f'layer={q_norm._p1_qk_runtime_layer}/m={q.shape[0]}'] += 1
        if not match:
            audit_dir.mkdir(parents=True,exist_ok=True)
            rank=torch.distributed.get_rank()
            torch.save(dict(q=q.cpu(),k=k.cpu(),q_weight=q_norm.weight.cpu(),k_weight=k_norm.weight.cpu(),
                       reference=[x.cpu() for x in reference],candidate=[x.cpu() for x in result]),
                       audit_dir/f'failure-rank-{rank}-layer-{q_norm._p1_qk_runtime_layer}.pt')
            raise RuntimeError('Q/K shadow bit gate failed')
        return reference

    model.Qwen3MoeAttention.apply_qk_norm_rope = identify
    model.apply_qk_norm = wrapped
    _INSTALLED = mode
    def flush(module,inputs,output):
        if audit_dir:
            audit_dir.mkdir(parents=True,exist_ok=True)
            rank=torch.distributed.get_rank()
            (audit_dir/f'rank-{rank}.json').write_text(json.dumps(dict(mode=mode,counts=dict(_COUNTS)),indent=2)+'\n')
        return None
    return flush
