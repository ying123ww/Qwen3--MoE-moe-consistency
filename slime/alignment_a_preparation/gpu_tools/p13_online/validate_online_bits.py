"""Strict target-token bit gate; CPU tests never substitute for the real run."""
import argparse
import json
from pathlib import Path
import torch


def compare(rollout,train):
    samples=rollout['samples'];training={int(x['rollout_position']):x for x in train['samples']}
    if len(samples)!=8 or set(training)!=set(range(8)):raise RuntimeError('Expected all8 rollout samples with unique train positions')
    total=0;mismatches=0
    for index,sample in enumerate(samples):
        teacher=training[index]
        if int(sample['response_length'])!=128:raise RuntimeError('Expected128 real target tokens per sample')
        a=torch.as_tensor(sample['tokens'],dtype=torch.int64);b=torch.as_tensor(teacher['tokens'],dtype=torch.int64)
        if not torch.equal(a,b):raise RuntimeError('Training did not use the actual rollout token sequence')
        observed=torch.as_tensor(sample['rollout_log_probs'],dtype=torch.float32).reshape(-1)
        expected=torch.as_tensor(teacher['log_probs'],dtype=torch.float32).reshape(-1)
        if observed.numel()!=128 or expected.shape!=observed.shape or not torch.isfinite(observed).all() or not torch.isfinite(expected).all():raise RuntimeError('Incomplete/nonfinite target logprobs')
        mismatches+=int((observed.view(torch.int32)!=expected.view(torch.int32)).sum());total+=128
    return dict(target_tokens=total,bit_mismatches=mismatches,bitwise_exact=mismatches==0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    report={}
    for version in (0,1,2):
        rollout=torch.load(args.run/'rollout'/f'{version}.pt',map_location='cpu',weights_only=False)
        train=torch.load(args.run/'train'/f'{version}.pt',map_location='cpu',weights_only=False)
        report[version]=compare(rollout,train)
    (args.run/'target-logprob-bit-gate.json').write_text(json.dumps(report,indent=2)+'\n')
    if not all(x['bitwise_exact'] and x['target_tokens']==1024 for x in report.values()):raise RuntimeError('Real online target-token bit gate failed')
    print('P13_REAL_TARGET_LOGPROBS_BITWISE_EXACT_3X1024')
