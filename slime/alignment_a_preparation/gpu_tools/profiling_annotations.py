"""Experiment-only CPU scopes, installed through SGLang's forward hook factory."""
from functools import wraps

from torch.profiler import record_function

_INSTALLED = False


def annotate(original, component):
    @wraps(original)
    def wrapped(module, *args, **kwargs):
        name = component
        if component == 'rms':
            residual = kwargs.get('residual', args[1] if len(args) > 1 else None)
            name = 'residual_rms' if residual is not None else 'other_rms'
        shape = getattr(args[0], 'shape', ()) if args else ()
        m = shape[0] if shape else -1
        layer = getattr(module, 'layer_id', getattr(module, '_slime_qwen3_rms_layer', -1))
        with record_function(f'P1/{name}/layer={layer}/m={m}'):
            return original(module, *args, **kwargs)
    wrapped._p1_annotation = True
    return wrapped


def install(config):
    """Runs once in each worker; the root post-hook itself leaves outputs intact."""
    global _INSTALLED
    if not _INSTALLED:
        from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock, Qwen3MoeAttention
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
        from sglang.srt.layers.logits_processor import LogitsProcessor
        targets = (
            (Qwen3MoeSparseMoeBlock, '_router_logits', 'router_logits'),
            (Qwen3MoeSparseMoeBlock, '_select_experts', 'router_select'),
            (Qwen3MoeSparseMoeBlock, 'forward_deepep', 'moe'),
            (Qwen3MoeAttention, 'forward', 'attention'),
            (QKVParallelLinear, 'forward', 'attention_qkv'),
            (RowParallelLinear, 'forward', 'attention_output'),
            (RMSNorm, 'forward', 'rms'),
            (LogitsProcessor, 'forward', 'logits'),
        )
        for owner, method, component in targets:
            original = getattr(owner, method)
            if not getattr(original, '_p1_annotation', False):
                setattr(owner, method, annotate(original, component))
        _INSTALLED = True
        print('P1_ATTRIBUTION_HOOKS_INSTALLED', flush=True)
    return lambda module, inputs, output: None
