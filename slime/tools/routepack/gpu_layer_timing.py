"""Opt-in MoE CUDA-event attribution; never enable in throughput measurement rounds.

Events bracket module calls on the current stream (normally the default stream).
Intervals include stream dependency waits and host enqueue gaps; expert intervals
are observable wrapper time, not isolated GEMM time. No per-layer synchronization
or tensor reads occur. results() synchronizes once, after the attributed round.
"""
from collections import Counter, defaultdict
import os


class LayerTiming:
    def __init__(self, model, *, event_factory=None, synchronize=None, rank=None):
        self.model = model if isinstance(model, (list, tuple)) else [model]
        self.event_factory = event_factory
        self.synchronize = synchronize
        self.rank = rank
        self.handles = []
        self.rows = []
        self.counts = Counter()
        self.stacks = defaultdict(list)
        self.active_moe = defaultdict(list)
        self.targets = []
        self.entered = False
        self.closed = False
        self.failed = False
        self.cached = None

    def __enter__(self):
        if self.entered:
            raise RuntimeError('LayerTiming contexts are single-use')
        self.entered = True
        if self.event_factory is None or self.synchronize is None or self.rank is None:
            import torch
            if self.event_factory is None:
                self.event_factory = lambda: torch.cuda.Event(enable_timing=True)
            if self.synchronize is None:
                self.synchronize = torch.cuda.synchronize
            if self.rank is None:
                self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        try:
            seen = set()
            for chunk, model in enumerate(self.model):
                modules = dict(model.named_modules())
                for name, mlp in modules.items():
                    if name.split('.')[-1] != 'mlp' or not hasattr(mlp, 'experts') or not hasattr(mlp, 'token_dispatcher'):
                        continue
                    if mlp.experts is None or id(mlp) in seen:
                        continue
                    seen.add(id(mlp))
                    parent = modules.get(name.rsplit('.', 1)[0]) if '.' in name else model
                    layer_number = getattr(mlp, 'layer_number', None) or getattr(parent, 'layer_number', None)
                    layer = int(layer_number) - 1 if layer_number is not None else getattr(mlp, '_slime_sglang_deepep_global_layer', None)
                    if layer is None:
                        raise ValueError(f'Missing global layer identity for {name}')
                    target = (chunk, int(layer), name)
                    self.targets.append({'chunk': chunk, 'layer': int(layer), 'module': name})
                    self._install(mlp, target, 'moe')
                    self._install(mlp.experts, target, 'experts')
            if not self.targets:
                raise ValueError('No identifiable MoE mlp modules found')
        except BaseException:
            self._remove()
            self.closed = True
            self.failed = True
            raise
        return self

    def _install(self, module, target, kind):
        def before(_module, _inputs):
            stage = os.environ.get('ROUTING_REPLAY_STAGE', 'unset')
            key = (target, stage, kind)
            slot = self.counts[key]
            self.counts[key] += 1
            parent = self.active_moe[target][-1] if self.active_moe[target] else None
            parent_id = parent['event_id'] if kind == 'experts' and parent is not None else None
            if parent_id is not None:
                slot = parent['slot']
            start, end = self.event_factory(), self.event_factory()
            row = dict(event_id=len(self.rows), chunk=target[0], layer=target[1], module=target[2],
                stage=stage, slot=slot, kind=kind, parent_moe_event_id=parent_id,
                start=start, end=end, completed=False)
            self.rows.append(row)
            self.stacks[(target, kind)].append(row)
            if kind == 'moe':
                self.active_moe[target].append(row)
            start.record()

        def after(_module, _inputs, _output):
            stack = self.stacks[(target, kind)]
            if not stack:
                return
            row = stack.pop()
            row['end'].record()
            row['completed'] = True
            if kind == 'moe':
                self.active_moe[target].pop()

        self.handles.append(module.register_forward_pre_hook(before))
        self.handles.append(module.register_forward_hook(after, always_call=True))

    def _remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type, exc, traceback):
        self.failed = exc_type is not None
        self.closed = True
        self._remove()
        return False

    def results(self):
        if not self.closed:
            raise RuntimeError('Read results only after leaving the attribution context')
        if self.cached is not None:
            return self.cached
        self.synchronize()
        rows = []
        for row in self.rows:
            item = {k: v for k, v in row.items() if k not in ('start', 'end')}
            item['elapsed_ms'] = row['start'].elapsed_time(row['end']) if row['completed'] and not self.failed else None
            rows.append(item)
        parents_with_experts = {r['parent_moe_event_id'] for r in rows if r['kind'] == 'experts'}
        coverage = {
            'moe_calls': sum(r['kind'] == 'moe' for r in rows),
            'expert_calls': sum(r['kind'] == 'experts' for r in rows),
            'expert_calls_without_outer_moe': sum(r['kind'] == 'experts' and r['parent_moe_event_id'] is None for r in rows),
            'moe_calls_without_expert_hook': sum(r['kind'] == 'moe' and r['event_id'] not in parents_with_experts for r in rows),
        }
        self.cached = dict(rank=self.rank, failed=self.failed, targets=self.targets, coverage=coverage, events=rows,
            slot_policy='Zero-based occurrence per chunk/layer/replay-stage. Nested expert calls inherit the outer MoE slot. This is not a globally identified sample/step slot; use one train step per context or supply external schedule mapping.',
            timing_scope='Current-stream CUDA events around module calls; includes queue/dependency/communication waits and host enqueue gaps. Expert wrapper time is not isolated compute. Side-stream work is included only if joined to the recorded stream.',
            recompute_scope='replay_backward forward calls indicate recomputation, not autograd backward kernels. Internal MoE checkpoint recompute can invoke experts without outer MoE; those calls are explicitly unpaired. Zero expert hooks never implies zero expert compute.')
        return self.cached


def layer_timing(model, **kwargs):
    """with layer_timing(model) as trace: train(...); then trace.results()."""
    return LayerTiming(model, **kwargs)
