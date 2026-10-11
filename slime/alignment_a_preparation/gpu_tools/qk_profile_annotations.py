"""Profile-only Q/K scopes and first-warmup layout/tensor evidence."""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.profiler import record_function
import profiling_annotations as base

_BASE_INSTALL = base.install
_INSTALLED = False
_CONTEXT = None
_SEEN = set()
_SAMPLES = set()


def describe(x):
    return dict(shape=list(x.shape), stride=list(x.stride()), dtype=str(x.dtype),
                contiguous=x.is_contiguous(), storage_offset=x.storage_offset())


def evidence(kind, key, value):
    if torch.autograd._profiler_enabled() or (kind, key) in _SEEN:
        return
    _SEEN.add((kind, key))
    rank = torch.distributed.get_rank()
    directory = Path(os.environ['SLIME_QK_PROFILE_DETAIL_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'rank-{rank}-metadata.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(kind=kind, rank=rank, **value)) + '\n')


def install(config):
    global _INSTALLED
    hook = _BASE_INSTALL(config)
    if _INSTALLED:
        return hook
    import sglang.srt.models.qwen3_moe as model
    import sglang.srt.models.utils as utils
    import sglang.srt.layers.layernorm as norms
    original_attention = model.Qwen3MoeAttention.apply_qk_norm_rope
    original_norm = norms.RMSNorm.forward
    norm_unwrapped = getattr(original_norm, '__wrapped__', original_norm)
    original_native = norms.RMSNorm.forward_native
    original_batch = norms.rms_norm_batch_invariant
    original_reshape = utils._reshape_for_qk_norm
    original_qk = model.apply_qk_norm

    @wraps(original_attention)
    def attention(module, qkv, positions, forward_batch):
        layer = module.attn.layer_id
        for norm, name in ((module.q_norm, 'q_norm'), (module.k_norm, 'k_norm')):
            norm._p1_qk_kind = name
            norm._p1_qk_layer = layer
        with record_function(f'P1/qk_norm_rope/layer={layer}/m={qkv.shape[0]}'):
            return original_attention(module, qkv, positions, forward_batch)

    @wraps(original_norm)
    def norm(module, *args, **kwargs):
        global _CONTEXT
        kind = getattr(module, '_p1_qk_kind', None)
        if kind is None:
            return original_norm(module, *args, **kwargs)
        previous = _CONTEXT
        _CONTEXT = (kind, module._p1_qk_layer)
        try:
            with record_function(f'P1/{kind}/layer={module._p1_qk_layer}/m={args[0].shape[0]}'):
                return norm_unwrapped(module, *args, **kwargs)
        finally:
            _CONTEXT = previous

    @wraps(original_native)
    def native(module, *args, **kwargs):
        if _CONTEXT is None:
            return original_native(module, *args, **kwargs)
        kind, layer = _CONTEXT
        evidence('norm_path', (kind, layer), dict(component=kind, layer=layer, implementation='forward_native',
                 cast_x_before_out_mul=module.cast_x_before_out_mul, weight_dtype=str(module.weight.dtype), eps=module.variance_epsilon))
        with record_function(f'P1/{kind}_native/layer={layer}'):
            return original_native(module, *args, **kwargs)

    @wraps(original_batch)
    def batch(*args, **kwargs):
        if _CONTEXT is None:
            return original_batch(*args, **kwargs)
        kind, layer = _CONTEXT
        evidence('norm_path', (kind, layer), dict(component=kind, layer=layer, implementation='rms_norm_batch_invariant'))
        with record_function(f'P1/{kind}_batch_invariant/layer={layer}'):
            return original_batch(*args, **kwargs)

    @wraps(original_reshape)
    def reshape(x, head_dim):
        with record_function(f'P1/qk_layout/m={x.shape[0]}/head_dim={head_dim}'):
            result = original_reshape(x, head_dim)
        meta = dict(input=describe(x), output=describe(result), head_dim=head_dim,
                    copied=x.untyped_storage().data_ptr() != result.untyped_storage().data_ptr())
        key = (tuple(x.shape), tuple(x.stride()), str(x.dtype), head_dim)
        evidence('layout', key, meta)
        return result

    @wraps(original_qk)
    def qk(q, k, q_norm, k_norm, head_dim, *args, **kwargs):
        layer = q_norm._p1_qk_layer
        key = (layer, tuple(q.shape), tuple(k.shape), tuple(q.stride()), tuple(k.stride()))
        meta = dict(layer=layer, q=describe(q), k=describe(k), head_dim=head_dim,
                    q_weight_dtype=str(q_norm.weight.dtype), k_weight_dtype=str(k_norm.weight.dtype),
                    q_cast_before_weight=q_norm.cast_x_before_out_mul, k_cast_before_weight=k_norm.cast_x_before_out_mul)
        evidence('qk_input', key, meta)
        capture = not torch.autograd._profiler_enabled() and q.shape[0] <= 4 and key not in _SAMPLES
        if capture:
            before = dict(q=q.detach().cpu(), k=k.detach().cpu(), q_weight=q_norm.weight.detach().cpu(),
                          k_weight=k_norm.weight.detach().cpu(), q_eps=q_norm.variance_epsilon,
                          k_eps=k_norm.variance_epsilon, metadata=meta)
        with record_function(f'P1/qk_norm/layer={layer}/m={q.shape[0]}'):
            result = original_qk(q, k, q_norm, k_norm, head_dim, *args, **kwargs)
        if capture:
            _SAMPLES.add(key)
            rank = torch.distributed.get_rank()
            directory = Path(os.environ['SLIME_QK_PROFILE_DETAIL_DIR'])
            before.update(q_output=result[0].detach().cpu(), k_output=result[1].detach().cpu())
            torch.save(before, directory / f'rank-{rank}-layer-{layer}-m-{q.shape[0]}.pt')
        return result

    model.Qwen3MoeAttention.apply_qk_norm_rope = attention
    norms.RMSNorm.forward = norm
    norms.RMSNorm.forward_native = native
    norms.rms_norm_batch_invariant = batch
    utils._reshape_for_qk_norm = reshape
    model.apply_qk_norm = qk
    _INSTALLED = True
    evidence('annotation_source', 'source', dict(path=str(Path(__file__)),
             sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('QK_DETAIL_PROFILE_INSTALLED', flush=True)
    return hook
