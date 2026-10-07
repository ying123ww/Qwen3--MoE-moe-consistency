"""Inspect launch correlation and enclosing shaped CPU operators."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace', type=Path)
    args = ap.parse_args()
    with gzip.open(args.trace, 'rt') as f:
        events = json.load(f)['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel' and 'matmul_kernel_persistent' in e.get('name', '')]
    ids = {e.get('args', {}).get('External id') for e in kernels[:8]}
    corr = {e.get('args', {}).get('correlation') for e in kernels[:8]}
    cpu = [e for e in events if e.get('cat') in ('cpu_op','cuda_runtime','cuda_driver') and (e.get('args',{}).get('External id') in ids or e.get('args',{}).get('correlation') in corr)]
    for k in kernels[:8]:
        launches = [e for e in cpu if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('args',{}).get('correlation') == k['args']['correlation']]
        print(json.dumps({'kernel':k, 'launch':launches, 'parents': [e for e in events if e.get('cat') == 'cpu_op' and launches and e.get('tid') == launches[0].get('tid') and e['ts'] <= launches[0]['ts'] <= e['ts']+e.get('dur',0)]}))


if __name__ == '__main__':
    main()
