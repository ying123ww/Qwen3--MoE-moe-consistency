"""Recheck real B2 capture files and count logical experts without EP assumptions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(directory, output, run_id):
    records = [json.loads(line) for line in (directory / 'samples_captured.jsonl').read_text().splitlines() if line.strip()]
    if not records or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('Missing/duplicate capture-local sample IDs')
    histograms, shapes, qualified, arrays_hashes = [], [], [], []
    total_rows = total_tokens = total_bytes = 0
    id_min, id_max = 128, -1
    for i, row in enumerate(records):
        routes = np.load(directory / row['route_path'], allow_pickle=False)
        tokens = np.load(directory / row['token_path'], allow_pickle=False)
        if routes.dtype != np.int32 or routes.shape != (row['token_count'] - 1, 48, 8):
            raise ValueError('Wrong route dtype/shape')
        if tokens.dtype != np.int64 or tokens.shape != (row['token_count'],):
            raise ValueError('Wrong token dtype/shape')
        if row['traced_token_count'] != routes.shape[0] or row['layer_ids'] != list(range(48)):
            raise ValueError('Trace metadata coverage mismatch')
        if routes.min() < 0 or routes.max() >= 128 or np.any(np.diff(np.sort(routes, axis=-1), axis=-1) == 0):
            raise ValueError('Invalid logical expert IDs or duplicate topk')
        for value, key in ((routes, 'trace_hash'), (tokens, 'token_hash')):
            if hashlib.sha256(value.astype('<i8').tobytes()).hexdigest() != row[key]:
                raise ValueError(f'{key} mismatch')
        counts = np.stack([np.bincount(routes[:, layer].reshape(-1), minlength=128) for layer in range(48)])
        if not np.all(counts.sum(axis=1) == routes.shape[0] * 8):
            raise ValueError('Per-expert count does not conserve assignments')
        histograms.append(counts)
        shapes.append(list(routes.shape))
        total_rows += routes.shape[0]
        total_tokens += len(tokens)
        total_bytes += routes.nbytes
        id_min, id_max = min(id_min, int(routes.min())), max(id_max, int(routes.max()))
        qualified.append({**row, 'capture_local_sample_id': row['sample_id'], 'sample_id': f'{run_id}:{i}', 'run_id': run_id})
        arrays_hashes.append(row['trace_hash'])
    sample_counts = np.stack(histograms).astype(np.int64)
    summary = {'capture_directory': str(directory.resolve()), 'run_id': run_id,
               'sample_count': len(records), 'unique_route_hashes': len(set(arrays_hashes)),
               'shapes': shapes, 'total_input_tokens': total_tokens,
               'response_tokens': sum(r['response_length'] for r in records), 'traced_token_rows': total_rows,
               'raw_trace_bytes': total_bytes, 'logical_expert_id_range': [id_min, id_max],
               'total_assignments': int(sample_counts.sum()), 'hash_shape_range_topk_gate': 'passed',
               'per_expert_axes': ['sample', 'layer', 'logical_expert'],
               'scope': 'raw prompt+response-1 routes only; no synthetic terminal/padding and no inferred EP owner',
               'identity_note': 'Capture-local IDs are qualified with explicit run ID in samples_verified.jsonl; null source rollout IDs preserved, no training schedule inferred',
               'training_topology_gate': 'not_run', 'training_dispatch_gate': 'not_run', 'alignment_replay_gate': 'not_run'}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'samples_verified.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in qualified))
    np.savez_compressed(output / 'per_expert_counts.npz', sample_counts=sample_counts, all_samples_counts=sample_counts.sum(axis=0))
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    verify(args.directory, args.output, args.run_id)


if __name__ == '__main__':
    main()
