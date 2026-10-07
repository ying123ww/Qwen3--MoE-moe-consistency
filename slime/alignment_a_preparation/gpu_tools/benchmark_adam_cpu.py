"""CPU-only StatelessAdam attribution; does not modify the training optimizer."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--pairs', type=int, default=4)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
source = Path(__file__).resolve().parents[2] / 'slime/backends/megatron_utils/stateless_adam.py'
code = source.read_text()
spec = importlib.util.spec_from_file_location('actual_adam', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Preserve operation order and scalar values; only reuse the newly allocated abs output.
needle = 'grad_for_update.abs().mul(denominator_scale).add_(eps)'
assert code.count(needle) == 1
namespace = {}
exec(compile(code.replace(needle, 'grad_for_update.abs().mul_(denominator_scale).add_(eps)'), str(source), 'exec'), namespace)
classes = {'original': module.StatelessAdam, 'inplace_mul': namespace['StatelessAdam']}
configs = [('original', True, True), ('original', True, False), ('original', False, True), ('inplace_mul', True, True)]
torch.set_num_interop_threads(1)
sizes = [6 * 1024**2 // 4, 12 * 1024**2 // 4] * args.pairs
rows = []
reference_hash = None
for rep in range(3):
    threads_order = [1, 2, 4, 8] if rep % 2 == 0 else [8, 4, 2, 1]
    for threads in threads_order:
        torch.set_num_threads(threads)
        for variant, deterministic, fill in (configs if rep % 2 == 0 else configs[::-1]):
            torch.use_deterministic_algorithms(deterministic)
            torch.utils.deterministic.fill_uninitialized_memory = fill
            torch.manual_seed(123)
            params = [torch.nn.Parameter(torch.randn(n)) for n in sizes]
            for param in params:
                param.grad = torch.randn_like(param)
            # Match HybridDeviceOptimizer's one sub-optimizer per parameter.
            opts = [classes[variant]([param], lr=1e-3) for param in params]
            timings = []
            for step in range(8):
                start = time.perf_counter()
                for opt in opts:
                    opt.step()
                if step >= 3:
                    timings.append(time.perf_counter() - start)
            digest = hashlib.sha256()
            for param in params:
                digest.update(param.detach().numpy().tobytes())
            result_hash = digest.hexdigest()
            if reference_hash is None:
                reference_hash = result_hash
            assert result_hash == reference_hash, (rep, threads, variant, deterministic, fill)
            row = dict(rep=rep, threads=threads, variant=variant, deterministic=deterministic,
                       fill=fill, seconds=timings, median_s=statistics.median(timings), sha256=result_hash)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del opts, params, param, opt

# Separate attribution from unprofiled timing. with_stack exports available CPU call stacks.
torch.set_num_threads(4)
profiles = []
for variant, deterministic, fill in configs:
    torch.use_deterministic_algorithms(deterministic)
    torch.utils.deterministic.fill_uninitialized_memory = fill
    param = torch.nn.Parameter(torch.ones(12 * 1024**2 // 4))
    param.grad = torch.full_like(param, 0.25)
    opt = classes[variant]([param], lr=1e-3)
    name = f'{variant}-det{int(deterministic)}-fill{int(fill)}'
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
        opt.step()
    prof.export_chrome_trace(str(args.output / (name + '.trace.json')))
    events = [dict(name=e.name, parent=e.cpu_parent.name if e.cpu_parent else None,
                   self_cpu_us=e.self_cpu_time_total, stack=e.stack) for e in prof.events() if e.name == 'aten::fill_']
    operators = [dict(name=e.key, count=e.count, self_cpu_us=e.self_cpu_time_total) for e in prof.key_averages()]
    profiles.append(dict(name=name, fills=events, operators=operators))

# Exercise numerical branches and special FP32 bit patterns, beyond the timing fixture.
checks = []
torch.use_deterministic_algorithms(True)
torch.utils.deterministic.fill_uninitialized_memory = True
for correction in [True, False]:
    for maximize in [True, False]:
        for adamw in [True, False]:
            for decay in [0., 0.1]:
                torch.manual_seed(567)
                initial = torch.randn(10003)
                grad = torch.randn_like(initial)
                grad[:8] = torch.tensor([0., -0., 1e-38, -1e-38, float('inf'), -float('inf'), float('nan'), 1e30])
                outputs = []
                for cls in classes.values():
                    p = torch.nn.Parameter(initial.clone())
                    p.grad = grad.clone()
                    opt = cls([p], lr=1e-3, bias_correction=correction, maximize=maximize, adam_w_mode=adamw, weight_decay=decay)
                    for _ in range(5):
                        opt.step()
                    outputs.append(p.detach().view(torch.int32))
                assert torch.equal(*outputs)
                checks.append(dict(bias_correction=correction, maximize=maximize, adam_w_mode=adamw, decay=decay, bits_equal=True))
assert not torch.cuda.is_initialized()
result = dict(host=platform.node(), torch=torch.__version__, affinity=sorted(os.sched_getaffinity(0)),
              cpu_only=True, cuda_initialized=torch.cuda.is_initialized(), source=str(source),
              source_sha256=hashlib.sha256(code.encode()).hexdigest(), parameter_bytes=sum(sizes)*4,
              scope=f'{len(sizes)} CPU FP32 parameters, synthetic fixed gradients, 3 warmup + 5 measured steps, 3 alternating rounds in one process; no H2D or full training',
              rows=rows, profiles=profiles, branch_checks=checks)
(args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print('COMPLETE CPU-only; all hashes and branch checks passed', flush=True)
