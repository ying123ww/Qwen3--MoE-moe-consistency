"""Attribute persistent matmul kernels using recorded mm shapes and launch IDs."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def classify(k, n):
    if (k, n) == (2048, 2560):
        return 'attention_qkv'
    if (k, n) == (2048, 2048):
        return 'attention_output'
    if (k, n) == (2048, 128):
        return 'router'
    if k == 2048 and n in (151936, 75968, 37984, 18992):
        return 'lm_head'
    return 'unclassified'


def summarize(events):
    kernels = [e for e in events if e.get('cat') == 'kernel' and 'matmul_kernel_persistent' in e.get('name','')]
    wanted = {e['args'].get('External id') for e in kernels}
    mm = {e.get('args',{}).get('External id'):e for e in events if e.get('cat') == 'cpu_op' and e.get('name') == 'aten::mm' and e.get('args',{}).get('External id') in wanted}
    correlations = {e['args'].get('correlation') for e in kernels}
    launches = {e.get('args',{}).get('correlation'):e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('args',{}).get('correlation') in correlations}
    steps = sorted([e for e in events if e.get('cat') == 'user_annotation' and e.get('name','').startswith('step[')], key=lambda e:e['ts'])
    starts = [e['ts'] for e in steps]
    groups = defaultdict(lambda: {'calls':0,'gpu_ms':0.,'cpu_mm_ms':0.,'cpu_launch_ms':0.})
    unlinked = 0
    for kernel in kernels:
        external = kernel['args'].get('External id')
        op = mm.get(external)
        launch = launches.get(kernel['args'].get('correlation'))
        dims = op.get('args',{}).get('Input Dims',[]) if op else []
        if not op or not launch or launch.get('args',{}).get('External id') != external or len(dims) < 2 or len(dims[0]) != 2 or len(dims[1]) != 2:
            unlinked += 1
            continue
        m,k = dims[0]
        bk,n = dims[1]
        if k != bk:
            raise ValueError('Incompatible recorded mm dimensions')
        idx = bisect_right(starts, op['ts'])-1
        phase = 'unknown'
        if idx >= 0 and op['ts']+op['dur'] <= steps[idx]['ts']+steps[idx]['dur']:
            phase = 'decode' if steps[idx]['name'].startswith('step[DECODE') else 'prefill'
        types = op.get('args',{}).get('Input type', [])
        key = (phase, classify(k,n),m,k,n,str(types[:2]))
        row = groups[key]
        row['calls'] += 1
        row['gpu_ms'] += kernel['dur']/1000
        row['cpu_mm_ms'] += op['dur']/1000
        row['cpu_launch_ms'] += launch['dur']/1000
    rows = [dict(phase=key[0],family=key[1],m=key[2],k=key[3],n=key[4],input_types=key[5],**row) for key,row in sorted(groups.items())]
    return {'kernel_count':len(kernels),'unlinked_kernels':unlinked,'rows':rows,
            'scope':'GPU event duration sums, CPU mm wall intervals and CPU launch intervals overlap; do not add them or treat them as end-to-end cost. Shape family mapping is specific to Qwen3-30B-A3B with attention TP=2.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    with gzip.open(args.trace,'rt') as f:
        result = summarize(json.load(f)['traceEvents'])
    result.update(trace=str(args.trace),trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
