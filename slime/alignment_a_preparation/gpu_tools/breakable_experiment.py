"""Explicit Plan 2 configuration and one-time successful replay evidence."""
import inspect
import json
from pathlib import Path


def configure(config, mode, evidence):
    result = dict(config)
    if mode not in ('eager', 'breakable', 'plan1', 'baseline'):
        raise ValueError(mode)
    result.update(disable_cuda_graph=mode != 'breakable', disable_prefill_cuda_graph=True)
    result['cuda_graph_config'] = {
        'decode': dict(backend='breakable' if mode == 'breakable' else 'disabled',
                       bs=[2, 4], max_bs=4),
        'prefill': dict(backend='disabled'),
    }
    result['forward_hooks'] = [dict(name='breakable-proof', target_modules=['model'],
        hook_factory='breakable_experiment:install', config=dict(output_dir=str(evidence)))]
    if mode == 'plan1':
        result['forward_hooks'].append(dict(name='qk-experiment', target_modules=['model'],
            hook_factory='qk_runtime_hooks:install', config=dict(mode='joint', audit_dir=None)))
    return result


def install(config):
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import BreakableCudaGraphBackend
    import torch.distributed as dist
    rank = dist.get_rank()
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    original = BreakableCudaGraphBackend.replay
    seen = set()
    def replay(self, shape_key, static_forward_batch, **kwargs):
        result = original(self, shape_key, static_forward_batch, **kwargs)
        if shape_key not in seen:
            seen.add(shape_key)
            graph = self._graphs[shape_key]
            labels = []
            for fn in graph._break_fns:
                inner = inspect.getclosurevars(fn).nonlocals.get('captured_inner')
                labels.append(getattr(inner, '__qualname__', repr(inner)))
            record = dict(rank=rank, backend=type(self).__name__, shape_key=str(shape_key),
                          size=shape_key.size, segments=len(graph._segments), breaks=len(graph._break_fns),
                          break_functions=labels, successful_replay=True,
                          limitation='One record per successfully replayed shape; no per-token file I/O')
            (output / f'rank-{rank}-size-{shape_key.size}.json').write_text(json.dumps(record, indent=2)+'\n')
            print('BREAKABLE_SUCCESSFUL_REPLAY '+json.dumps({k:v for k,v in record.items() if k!='break_functions'}), flush=True)
        return result
    BreakableCudaGraphBackend.replay = replay
    return lambda module, inputs, output: None
