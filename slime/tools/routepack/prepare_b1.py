"""Convert verified Engine capture into fixed debug rollout for the B1 gate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def prepare(capture, output):
    capture, output = Path(capture), Path(output)
    rows = [json.loads(line) for line in (capture / 'samples_captured.jsonl').read_text().splitlines()]
    samples = []
    for i, row in enumerate(rows):
        if row.get('effective_logprob_temperature') != 1.0:
            raise ValueError('B1 requires declared effective logprob temperature 1.0')
        tokens = np.load(capture / row['token_path'], allow_pickle=False)
        routes = np.load(capture / row['route_path'], allow_pickle=False)
        logprobs = np.load(capture / row['response_logprob_path'], allow_pickle=False)
        if tokens.ndim != 1 or routes.shape != (len(tokens) - 1, 48, 8):
            raise ValueError('Invalid Qwen3 B1 tokens/routes shape')
        if logprobs.shape != (row['response_length'],) or not np.isfinite(logprobs).all():
            raise ValueError('Missing/nonfinite response logprobs')
        if hashlib.sha256(tokens.astype('<i8').tobytes()).hexdigest() != row['token_hash']:
            raise ValueError('Token hash mismatch')
        if hashlib.sha256(routes.astype('<i8').tobytes()).hexdigest() != row['trace_hash']:
            raise ValueError('Trace hash mismatch')
        if routes.min() < 0 or routes.max() >= 128 or np.any(np.diff(np.sort(routes, axis=-1), axis=-1) == 0):
            raise ValueError('Invalid logical top-k expert IDs')
        samples.append({'index': i, 'group_index': row['reward_group_id'] if row.get('reward_group_id') is not None else i // 2, 'rollout_id': i,
                        'tokens': tokens.tolist(), 'response_length': row['response_length'],
                        'rollout_log_probs': logprobs.astype(np.float32).tolist(),
                        'rollout_routed_experts': torch.from_numpy(routes.copy()),
                        'loss_mask': [1] * row['response_length'], 'reward': 0.0, 'status': 'completed',
                        'metadata': {'routepack_sample_id': row['sample_id'], 'capture': str(capture),
                                     'reward_note': 'placeholder unused by forward-only gate'}})
    if not samples:
        raise ValueError('Empty capture')
    # Exclusive creation prevents accidentally replacing evidence from an earlier run.
    with output.open('xb') as handle:
        torch.save({'samples': samples}, handle)
    # Prove this newly authored payload is readable without arbitrary pickle globals.
    checked = torch.load(output, map_location='cpu', weights_only=True)
    assert len(checked['samples']) == len(samples)
    return {'samples': len(samples), 'path': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('capture')
    parser.add_argument('output')
    args = parser.parse_args()
    print(json.dumps(prepare(args.capture, args.output), indent=2))
