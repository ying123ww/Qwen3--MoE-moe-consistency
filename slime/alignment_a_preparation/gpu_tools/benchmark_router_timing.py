"""Interleaved single-GPU checks of the audit-free Qwen3 router candidate."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite existing evidence')
    os.environ['SLIME_QWEN3_ROUTER_TILE_MODE'] = 'candidate'
    os.environ['SLIME_QWEN3_ROUTER_TILE_AUDIT'] = '0'
    os.environ.pop('SLIME_QWEN3_ROUTER_TILE_AUDIT_DIR', None)
    import torch
    import triton
    from sglang.srt.batch_invariant_ops import batch_invariant_ops as original
    project = Path(__file__).resolve().parents[3]
    helper = project / 'experiments/sglang-main-sm80/python/sglang/srt/models/qwen3_router_tile.py'
    spec = importlib.util.spec_from_file_location('router_timing_candidate', helper)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    torch.cuda.set_device(0)
    rows = []
    probes = []
    for m in (1, 2):
        for seed in range(3):
            torch.manual_seed(20261007 + seed)
            # Router operands originate from BF16 activations and weights, then cast to FP32.
            a = torch.randn((m + 1, 2048), device='cuda', dtype=torch.bfloat16).float()
            b = torch.randn((128, 2048), device='cuda', dtype=torch.bfloat16).float().t()
            reference = original.matmul_persistent(a[:m], b)
            result = candidate.router_matmul(a[:m], b, layer_id=0)
            expanded = original.matmul_persistent(a, b)[:m]
            mismatch = int((reference.view(torch.int32) != result.view(torch.int32)).sum())
            batch_mismatch = int((expanded.view(torch.int32) != result.view(torch.int32)).sum())
            probes.append(dict(m=m, seed=seed, compared_elements=m * 128,
                               mismatches=mismatch, batch_mismatches=batch_mismatch))
            if mismatch or batch_mismatch:
                raise RuntimeError('Single-GPU bit check failed: ' + json.dumps(probes[-1]))
        functions = {
            'original': lambda: original.matmul_persistent(a[:m], b),
            'candidate_no_audit': lambda: candidate.router_matmul(a[:m], b, layer_id=0),
        }
        for function in functions.values():
            for _ in range(100):
                function()
        torch.cuda.synchronize()
        measurements = {name: [] for name in functions}
        for repeat in range(12):
            order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
            for name in order:
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(250):
                    functions[name]()
                torch.cuda.synchronize()
                measurements[name].append((time.perf_counter() - started) * 1e6 / 250)
        medians = {name: statistics.median(values) for name, values in measurements.items()}
        rows.append(dict(m=m, k=2048, n=128, dtype='float32',
                         wall_us=medians, samples_wall_us=measurements,
                         latency_reduction_percent=(1 - medians['candidate_no_audit'] / medians['original']) * 100))
    if candidate._CALLS:
        raise RuntimeError('Timing mode unexpectedly updated audit counters')
    sources = (Path(__file__), helper, Path(original.__file__))
    result = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, triton=triton.__version__,
                  device_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'), rows=rows, bit_probes=probes,
                  compared_elements=sum(p['compared_elements'] for p in probes),
                  audit_counters_empty=True,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  gpu_state=subprocess.check_output(['nvidia-smi', '--query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu', '--format=csv'], text=True),
                  scope='Single-GPU synthetic BF16-origin FP32 operands, three seeds, M=1/2; wall time includes allocation, Python/Triton launch and GPU completion. Not a model, cross-framework, pure-kernel or end-to-end performance gate.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

