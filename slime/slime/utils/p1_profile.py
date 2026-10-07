"""Bounded opt-in P1 attribution; profiled timings are not benchmark results."""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import time

_ACTIVE = False


@contextmanager
def span(name):
    if not _ACTIVE:
        yield
        return
    import torch
    with torch.profiler.record_function('P1/' + name):
        yield


def rpc_profile(stage):
    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            folder = os.environ.get('SLIME_P1_PROFILE_DIR')
            if not folder or stage not in os.environ.get('SLIME_P1_PROFILE_STAGES', 'train,weight_sync').split(','):
                return fn(self, *args, **kwargs)
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            if stage == 'train':
                self._p1_rollout_id = kwargs.get('rollout_id', args[0] if args else None)
            rollout_id = getattr(self, '_p1_rollout_id', None)
            rank = self.args.rank
            if rollout_id != int(os.environ.get('SLIME_P1_PROFILE_ROLLOUT', '3')) or rank not in (0, 4):
                return fn(self, *args, **kwargs)
            import torch
            global _ACTIVE
            out = Path(folder)/'train'
            out.mkdir(parents=True, exist_ok=True)
            stem = f'{stage}-rollout{rollout_id}-rank{rank}'
            started = time.perf_counter()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False, with_stack=False,
                                        profile_memory=False) as prof:
                _ACTIVE = True
                try:
                    with span(stage + '_rpc'):
                        result = fn(self, *args, **kwargs)
                        torch.cuda.synchronize()
                finally:
                    _ACTIVE = False
            elapsed = time.perf_counter()-started
            prof.export_chrome_trace(str(out/(stem+'.json.gz')))
            rows = []
            for e in prof.key_averages():
                rows.append(dict(name=e.key, count=e.count, cpu_total_us=e.cpu_time_total,
                                 self_cpu_us=e.self_cpu_time_total,
                                 device_total_us=getattr(e,'device_time_total',0),
                                 self_device_us=getattr(e,'self_device_time_total',0)))
            (out/(stem+'-operators.json')).write_text(json.dumps(rows,indent=2)+'\n')
            (out/(stem+'-meta.json')).write_text(json.dumps(dict(rank=rank,stage=stage,
                rollout_id=rollout_id,profiled_wall_s=elapsed,torch=torch.__version__,
                scope='CPU+CUDA representative rank; capture/export perturb execution; not a throughput benchmark'))+'\n')
            return result
        return wrapped
    return decorate


def named_span(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with span(name):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


def start_rollout_profile(rollout_manager, rollout_id):
    folder = os.environ.get('SLIME_P1_PROFILE_DIR')
    if os.environ.get('SLIME_P1_PROFILE_STAGES') == 'train':
        return
    if not folder or rollout_id != int(os.environ.get('SLIME_P1_PROFILE_ROLLOUT','3')):
        return
    import ray
    engines = ray.get(rollout_manager.get_updatable_engines_and_lock.remote())[0]
    if not engines:
        raise RuntimeError('P1 requires a live rollout engine')
    # Separate prefill from the first eight decode steps; auto-export before training.
    responses = ray.get([e.start_profile.remote(output_dir=str(Path(folder)/'rollout'),
                          num_steps=8, activities=['CPU','GPU'], profile_by_stage=True,
                          with_stack=False, record_shapes=False) for e in engines])
    for response in responses:
        # Current SGLang returns plain text 'Start profiling.', not JSON.
        if response is not None:
            response.raise_for_status()
