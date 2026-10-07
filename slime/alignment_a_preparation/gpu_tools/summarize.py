"""Aggregate synchronized complete-step measurements across all eight ranks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def summarize(run, warmup):
    audits = [json.loads((run / f'audit-rank{rank}.json').read_text()) for rank in range(8)]
    rows = {}
    for rank in range(8):
        seen = set()
        for line in (run / f'measurements-rank{rank}.jsonl').read_text().splitlines():
            row = json.loads(line)
            key = (row['rollout_id'], row['step_id'])
            assert key not in seen and row['rank'] == rank
            assert math.isfinite(row['elapsed_s']) and row['elapsed_s'] > 0
            assert math.isfinite(row['grad_norm']) and row['grad_norm'] > 0
            assert row['global_batch_size'] == 8
            seen.add(key)
            rows.setdefault(key, []).append(row)
    assert all(len(group) == 8 for group in rows.values())
    keys = sorted(rows)[warmup:]
    if not keys:
        raise ValueError('No post-warmup measurements')
    times = [max(r['elapsed_s'] for r in rows[key]) for key in keys]
    measured = [r for key in keys for r in rows[key]]
    source = Path('/public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt')
    result = {'run': run.name, 'variant': audits[0]['variant'],
              'scope': audits[0]['timing_scope'], 'first_rollout_unmeasured': True,
              'additional_warmup_steps': warmup, 'measured_steps': len(keys),
              'input_tokens_per_step': 2242, 'target_tokens_per_step': 1024,
              'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'step_time_median_s': statistics.median(times), 'step_time_p95_s': percentile(times, .95),
              'step_time_min_s': min(times), 'step_time_max_s': max(times),
              'input_tokens_per_s': 2242 / statistics.median(times),
              'peak_allocated_gib_max_rank': max(r['peak_allocated_bytes'] for r in measured) / 2**30,
              'peak_reserved_gib_max_rank': max(r['peak_reserved_bytes'] for r in measured) / 2**30,
              'grad_norm_range': [min(r['grad_norm'] for r in measured), max(r['grad_norm'] for r in measured)],
              'step_times_max_rank_s': times,
              'memory_note': 'PyTorch allocator peaks, not total GPU/NVSHMEM memory',
              'quality_note': 'fixed trace, lr=0, StatelessAdam; small regression batch, not saturation or online RL throughput'}
    (run / 'measurement-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--warmup', type=int, default=5)
    args = p.parse_args()
    print(json.dumps(summarize(args.run, args.warmup), indent=2))
