"""Inventory a trusted project rollout using PyTorch's restricted loader."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    import torch
    data = torch.load(args.rollout, map_location='cpu', weights_only=True)
    samples = []
    for position, sample in enumerate(data['samples']):
        route = sample.get('rollout_routed_experts')
        samples.append({'sample_position': position, 'sample_index': sample['index'],
                        'rollout_id': sample.get('rollout_id'), 'reward_group_id': sample.get('group_index'),
                        'token_count': len(sample['tokens']), 'response_length': sample['response_length'],
                        'has_route_trace': route is not None,
                        'trace_shape': list(route.shape) if hasattr(route, 'shape') else None})
    report = {'input': str(args.rollout.resolve()), 'sha256': hashlib.sha256(args.rollout.read_bytes()).hexdigest(),
              'loader': 'torch.load(map_location=cpu, weights_only=True)',
              'cuda_initialized': torch.cuda.is_initialized(), 'sample_count': len(samples), 'samples': samples,
              'trace_coverage_count': sum(s['has_route_trace'] for s in samples),
              'actual_dispatch_route_gate': 'not_run', 'topology_gate': 'not_run'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('sample_count', 'trace_coverage_count', 'cuda_initialized')}))


if __name__ == '__main__':
    main()
