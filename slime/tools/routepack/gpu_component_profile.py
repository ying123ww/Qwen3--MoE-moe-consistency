"""Opt-in first-microbatch F/B plus optimizer profiler, never formal timing.

Current runner requires one model chunk and PP=1. Kineto records CUDA activity
across streams; CPU range durations are never substituted for kernel durations.
"""
from collections import Counter
import functools
import json
import os
from pathlib import Path
import sys


class ComponentProfile:
    def __init__(self, model, optimizer, output_dir, rank, *, slot=0, torch_module=None, schedules_module=None, aligned_module=None):
        chunks = model if isinstance(model, (list, tuple)) else [model]
        if len(chunks) != 1 or slot < 0:
            raise ValueError('Requires one model chunk, PP=1 and nonnegative microbatch slot')
        self.model, self.optimizer = chunks[0], optimizer
        self.output_dir, self.rank, self.slot = Path(output_dir), rank, slot
        self.torch, self.schedules, self.aligned = torch_module, schedules_module, aligned_module
        self.undo, self.installed = [], []
        self.calls = Counter()
        self.active = False
        self.used = False
        self.phases = []
        self.summary = None

    def _patch(self, owner, name, replacement):
        existed = name in vars(owner)
        previous = vars(owner).get(name)
        setattr(owner, name, replacement)
        self.undo.append((owner, name, existed, previous))

    def _restore(self):
        for owner, name, existed, previous in reversed(self.undo):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self.undo.clear()

    def _range(self, owner, method, label):
        original = getattr(owner, method, None)
        if not callable(original):
            return
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            stage = os.getenv('ROUTING_REPLAY_STAGE', 'unset')
            with self.torch.profiler.record_function('RoutePack/' + label + '/' + stage):
                return original(*args, **kwargs)
        self._patch(owner, method, wrapped)
        self.installed.append(label)

    def _window(self, owner, method, phase):
        original = getattr(owner, method)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            index = self.calls[phase]
            self.calls[phase] += 1
            selected = index == (0 if phase == 'optimizer' else self.slot)
            if not selected:
                return original(*args, **kwargs)
            if self.active:
                raise RuntimeError('Unexpected nested collection window')
            # Exclude preceding loss/zero-grad asynchronous tails.
            self.torch.cuda.synchronize()
            profiler = self.torch.profiler.profile(activities=self.activities, record_shapes=False, profile_memory=False, with_stack=False)
            profiler.__enter__()
            self.active = True
            try:
                with self.torch.profiler.record_function('RoutePack/window/' + phase):
                    return original(*args, **kwargs)
            finally:
                error = sys.exc_info()
                try:
                    self.torch.cuda.synchronize()
                finally:
                    self.active = False
                    profiler.__exit__(*error)
                path = self.output_dir / f'components-rank{self.rank}-{phase}.json'
                profiler.export_chrome_trace(str(path))
                ranges = []
                for event in profiler.key_averages():
                    if event.key.startswith('RoutePack/'):
                        ranges.append(dict(name=event.key, calls=event.count, cpu_inclusive_us=event.cpu_time_total,
                            device_inclusive_us=getattr(event, 'device_time_total', None)))
                self.phases.append(dict(phase=phase, trace_path=str(path), failed=error[0] is not None, ranges=ranges))
        self._patch(owner, method, wrapped)

    def __enter__(self):
        if self.used:
            raise RuntimeError('Profiler context is single-use')
        self.used = True
        if self.torch is None:
            import torch
            self.torch = torch
        if self.schedules is None:
            from megatron.core.pipeline_parallel import schedules
            self.schedules = schedules
        if self.aligned is None:
            self.aligned = sys.modules.get('slime.backends.megatron_utils.alignment.qwen3')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / f'components-rank{self.rank}-summary.json'
        if self.summary_path.exists() or any((self.output_dir / f'components-rank{self.rank}-{phase}.json').exists() for phase in ('forward','backward','optimizer')):
            raise FileExistsError('Refuse to overwrite component profile evidence')
        self.activities = [self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA]
        try:
            seen = set()
            for name, mlp in self.model.named_modules():
                if name.split('.')[-1] != 'mlp' or not hasattr(mlp, 'experts') or id(mlp) in seen:
                    continue
                seen.add(id(mlp))
                prefix = name + '/'
                for method, component in [('forward','moe'),('dispatch','dispatch'),('combine','combine')]:
                    self._range(mlp, method, prefix + component)
                dispatcher = getattr(mlp, 'token_dispatcher', None)
                if dispatcher is not None:
                    for method in ('dispatch_preprocess','dispatch_postprocess','combine_preprocess','combine_postprocess','token_dispatch','token_combine'):
                        self._range(dispatcher, method, prefix + method)
                    manager = getattr(dispatcher, '_comm_manager', None)
                    if manager is not None:
                        for method in ('get_permuted_hidden_states_by_experts','get_restored_hidden_states_by_experts'):
                            self._range(manager, method, prefix + method)
                experts = mlp.experts
                self._range(experts, 'forward', prefix + 'expert_wrapper')
                for linear in ('linear_fc1','linear_fc2'):
                    if getattr(experts, linear, None) is not None:
                        self._range(getattr(experts, linear), 'forward', prefix + 'native_TE_' + linear)
            if self.aligned is not None:
                self._range(self.aligned, '_run_sglang_triton_moe', 'aligned_triton_moe')
            self._window(self.model, 'forward', 'forward')
            self._window(self.schedules, 'backward_step', 'backward')
            self._window(self.optimizer, 'step', 'optimizer')
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._restore()
        ranges = [row for phase in self.phases for row in phase['ranges']]
        self.summary = dict(rank=self.rank, slot=self.slot, failed=exc_type is not None, calls=dict(self.calls),
            phases=self.phases, trace_paths=[p['trace_path'] for p in self.phases], installed_ranges=self.installed, ranges=ranges,
            scope='Only selected microbatch model.forward and schedules.backward_step plus first optimizer.step; not full-step throughput. Synchronization at selected-window starts and ends excludes prior work and drains CUDA tails; model.forward excludes subsequent loss-forward work. Each window has an independent profiler context; trace export overhead is attribution-only.',
            limitations=[
                'CPU and device inclusive times are distinct and nested; never add component totals as exclusive costs.',
                'Use Chrome CUDA kernel/memcpy events and correlation IDs for multi-stream overlap or union; key averages are not critical-path wall time.',
                'native_TE_linear_fc1/2 cover native linear calls only, not the full original expert closure, activations or all native backward work.',
                'backward window includes recompute and autograd kernels; nested recompute component ranges overlap it.',
                'Aligned Triton function excludes argument construction such as stacked weights; expert_wrapper includes that work.',
                'First backward occurrence maps to selected forward only for the current PP=1 single-step schedule; no pipeline-general slot claim.'])
        self.summary_path.write_text(json.dumps(self.summary, indent=2)+'\n')
        return False

    def results(self):
        if self.summary is None:
            raise RuntimeError('Read results after leaving component_profile')
        return self.summary


def component_profile(model, optimizer, output_dir, rank, **kwargs):
    return ComponentProfile(model, optimizer, output_dir, rank, **kwargs)
