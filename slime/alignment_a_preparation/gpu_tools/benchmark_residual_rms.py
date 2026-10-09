"""Finite numerical and interleaved timing pilot; never modifies a live model."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite evidence')
    import torch
    import residual_rms_candidate as norm
    torch.cuda.set_device(0)
    probes = []
    for m in (1, 2, 4):
        for weight_dtype in (torch.bfloat16, torch.float32):
            for scale in (0., .01, 1., 10.):
                for seed in range(3):
                    torch.manual_seed(20261007 + seed)
                    x = (torch.randn(m, 2048, device='cuda') * scale).to(torch.bfloat16)
                    residual = (torch.randn_like(x.float()) * scale).to(torch.bfloat16)
                    x[:, 0], residual[:, 0] = -0., -0.
                    residual[:, 1] = -x[:, 1]
                    weight = torch.randn(2048, device='cuda', dtype=weight_dtype)
                    expected = norm.reference(x, residual, weight)
                    actual = norm.candidate(x, residual, weight)
                    mismatches = [int((a.view(torch.int16) != b.view(torch.int16)).sum())
                                  for a, b in zip(actual, expected)]
                    probes.append(dict(m=m, weight_dtype=str(weight_dtype), scale=scale, seed=seed,
                                       elements=2*x.numel(), mismatches=mismatches))
    failures = [p for p in probes if any(p['mismatches'])]
    result = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, probes=probes,
                  compared_elements=sum(p['elements'] for p in probes), failures=failures,
                  exact=not failures, rows=[],
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path(norm.__file__))},
                  scope='Single-GPU finite synthetic input/output bit and signed-zero checks. BF16 residual, cast-after-weight, no override/extra residual. Uses original batch-invariant mean chunks. No real-model, batch or cross-framework gate.')
    if not failures:
        for m in (1, 2, 4):
            x = torch.randn(m, 2048, device='cuda', dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            weight = torch.randn(2048, device='cuda', dtype=torch.bfloat16)
            functions = dict(original=lambda: norm.reference(x, residual, weight),
                             fused=lambda: norm.candidate(x, residual, weight))
            for f in functions.values():
                for _ in range(50): f()
            times = {name:[] for name in functions}
            for repeat in range(10):
                order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
                for name in order:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    for _ in range(100): functions[name]()
                    torch.cuda.synchronize()
                    times[name].append((time.perf_counter()-started)*1e6/100)
            medians = {name:statistics.median(values) for name,values in times.items()}
            result['rows'].append(dict(m=m, wall_us=medians, samples_us=times,
                                       latency_reduction_percent=(1-medians['fused']/medians['original'])*100))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('probes','source_sha256')},indent=2),flush=True)
    if failures:
        raise RuntimeError('Fused residual RMSNorm prototype has bit mismatches; see saved probes')


if __name__ == '__main__':
    main()

