"""Recompute saved target logprobs on CPU; native differences are reported."""
import argparse
import json
import os
from pathlib import Path
import runpy

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('/public/home/yingxy2024/moe-consistency/slime')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--mode', choices=['restore', 'aligned', 'native'], required=True)
    args = p.parse_args()
    compare = runpy.run_path(str(ROOT / 'slime/utils/compare_qwen3_logprobs.py'))['compare_log_probs']
    stats_list = []
    steps = (1, 2) if args.mode == 'restore' else (1,)
    for step in steps:
        rollout = args.run / f'rollout/{step}.pt' if args.mode == 'restore' else Path('/public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt')
        train = args.run / f'train/{step}.pt'
        stats = compare(rollout, train)
        result = {'rollout_data': str(rollout), 'train_data': str(train), 'log_probs': stats}
        (args.run / f'logprob-comparison-step{step}.json').write_text(json.dumps(result, indent=2) + '\n')
        assert stats['samples'] == 8 and stats['tokens'] == 1024, 'Unexpected sample/token count'
        assert stats['rollout_nonfinite'] == 0 and stats['train_nonfinite'] == 0
        if args.mode != 'native':
            assert stats['exact'], f'Alignment failed: {stats["max_abs"]}'
        stats_list.append({'step': step, **{k: stats[k] for k in ['samples', 'tokens', 'max_abs', 'mean_abs', 'mismatched_tokens', 'exact']}})
    import torch
    assert not torch.cuda.is_initialized()
    (args.run / 'correctness-summary.json').write_text(json.dumps(stats_list, indent=2) + '\n')
    print(json.dumps(stats_list), flush=True)


if __name__ == '__main__':
    main()
