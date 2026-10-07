"""CPU summary of captured lengths and verified collective loads; no GPU claims."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantiles(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    return {'count': int(a.size), 'min': float(a.min()), 'p10': float(np.percentile(a, 10)),
            'p50': float(np.percentile(a, 50)), 'p90': float(np.percentile(a, 90)),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max()), 'mean': float(a.mean())}


def h_summary(h):
    means = h.mean(axis=-1)
    ratios = np.divide(h.max(axis=-1), means, out=np.full(means.shape, np.nan), where=means != 0)
    return {'peak_mean_ratio_per_collective_layer': quantiles(ratios[np.isfinite(ratios)]),
            'zero_load_collective_layers': int((means == 0).sum()),
            'E': float((h.max(axis=-1) - means).sum())}


def summarize(capture, b2, benchmarks):
    manifest = json.loads((capture / 'manifest.json').read_text())
    captured = rows(capture / 'captured/samples_captured.jsonl')
    reasons = Counter()
    for sample in captured:
        reason = sample.get('finish_reason')
        kind = reason.get('type') if isinstance(reason, dict) else reason
        reasons[str(kind) if kind is not None else 'unknown'] += 1
    result = {'scope': 'Offline measured loads/CPU timings; not GPU scheduler speedup',
              'capture': str(capture.resolve()), 'sampling': manifest.get('sampling'),
              'lengths': {'prompt': quantiles([s['token_count']-s['response_length'] for s in captured]),
                          'response': quantiles([s['response_length'] for s in captured]),
                          'total': quantiles([s['token_count'] for s in captured])},
              'finish': {'reason_counts': dict(reasons), 'confirmed_length_truncated_count': reasons['length'],
                         'stop_count': reasons['stop'], 'unknown_count': reasons['unknown'],
                         'note': 'stop is engine stop, not necessarily EOS; missing reasons remain unknown. Length equal to cap does not establish finish reason.'}}
    # A stop token is EOS only if it matches a configured EOS token id.
    config_path = Path(manifest.get('requested_server_args', {}).get('model_path', '')) / 'config.json'
    eos_ids = set()
    if config_path.is_file():
        eos = json.loads(config_path.read_text()).get('eos_token_id', [])
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    eos_count = 0
    for s in captured:
        reason = s.get('finish_reason')
        if isinstance(reason, dict) and reason.get('type') == 'stop' and isinstance(reason.get('matched'), int):
            eos_count += reason['matched'] in eos_ids
    result['finish'].update(eos_token_ids=sorted(eos_ids), confirmed_eos_count=eos_count,
                            eos_identity_available=bool(eos_ids))
    if b2:
        samples = rows(b2/'samples.jsonl')
        collectives = rows(b2/'collectives.jsonl')
        schedule = json.loads((b2/'schedule.json').read_text())
        by_id = {s['sample_id']: i for i, s in enumerate(samples)}
        with np.load(b2/'route_loads.npz', allow_pickle=False) as arrays:
            h = arrays['collective_H']
            no_layout = h-arrays['layout_padding_H']
            raw = no_layout.copy()
            keys = {(c['step'], c['group_id'], c['slot']): i for i, c in enumerate(collectives)}
            for cell in schedule['cells']:
                k = keys[(cell['step'], cell['group_id'], cell['slot'])]
                for sid in cell['sample_ids']:
                    raw[k] -= arrays['terminal_H'][by_id[sid]]
            assert np.all(raw >= 0) and len(h) == len(collectives)
            assert np.isclose(h_summary(h)['E'], sum(c['EP_excess'] for c in collectives))
            train = sum(c['training_tokens'] for c in collectives)
            pad = sum(c['layout_padding_tokens'] for c in collectives)
            assert train == sum(s['token_count'] for s in samples)
            result['b2'] = {'path': str(b2.resolve()), 'samples': len(samples), 'bins': len(schedule['cells']),
                            'collectives': len(collectives), 'training_tokens': train, 'layout_padding_tokens': pad,
                            'layout_padding_fraction': pad/(train+pad),
                            'H_including_terminal_and_layout_padding': h_summary(h),
                            'H_excluding_layout_padding_including_terminal': h_summary(no_layout),
                            'H_raw_excluding_layout_padding_and_synthetic_terminal': h_summary(raw)}
    result['benchmarks'] = []
    for path in benchmarks:
        b = json.loads(path.read_text())
        result['benchmarks'].append({'path': str(path.resolve()), **{k: b.get(k) for k in (
            'seed', 'budgets', 'repeats', 'excess', 'same_swap_count', 'routepack_swaps', 'random_swaps',
            'cpu_wall_ms', 'determinism_passed')}})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', required=True, type=Path, help='Directory with manifest.json and captured/')
    p.add_argument('--b2', type=Path, help='Verified B2 directory; optional during capture')
    p.add_argument('--benchmark', action='append', default=[], type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.write_text(json.dumps(summarize(args.capture, args.b2, args.benchmark), indent=2)+'\n')
    print(args.output)
