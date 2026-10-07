"""Offline screening of existing training traces; never reconstructs routing IDs."""
import argparse
import bisect
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total(intervals):
    return sum(b-a for a,b in merge(intervals))


def clipped(events, lo, hi):
    return [(max(lo,e['ts']), min(hi,e['ts']+e['dur'])) for e in events
            if e['ts'] < hi and e['ts']+e['dur'] > lo]


def analyze(path):
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    events = [e for e in data['traceEvents'] if e.get('ph')=='X' and e.get('dur',0)>0]
    spans = {e['name']: e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('P1/')}
    gpu = sorted([e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')],key=lambda e:e['ts'])
    apis = {e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
    grouped = defaultdict(list)
    for e in events:
        if e.get('cat')=='cpu_op' and e['name'] in ('_GroupedLinear','_GroupedLinearBackward'):
            grouped[e['pid'],e['tid']].append(e)
    starts = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda e:e['ts'])
        assert all(a['ts']+a['dur'] <= b['ts']+.01 for a,b in zip(rows,rows[1:])), 'Overlapping grouped scopes'
        starts[key] = [e['ts'] for e in rows]
    def grouped_parent(api):
        key = api['pid'],api['tid']
        i = bisect.bisect_right(starts.get(key,[]),api['ts'])-1
        if i>=0:
            e=grouped[key][i]
            if api['ts']+api['dur'] <= e['ts']+e['dur']+.01:
                return e['name']
        return None
    def category(e):
        if e['cat']!='kernel': return e['cat']
        n=e['name'].lower()
        if 'deep_ep' in n:
            if 'get_dispatch_layout' in n:return 'deepep_layout'
            if 'notify' in n:return 'deepep_notify'
            if 'dispatch' in n:return 'deepep_dispatch'
            if 'combine' in n:return 'deepep_combine'
            return 'deepep_other'
        if 'nccl' in n:return 'nccl'
        if 'gemm' in n or 'matmul' in n or 'cutlass' in n or n=='fused_moe_kernel':
            api=apis.get(e.get('args',{}).get('correlation'))
            parent=grouped_parent(api) if api else None
            return 'expert_gemm_'+parent if parent else 'other_or_unattributed_gemm'
        return 'other_kernel'
    stages={}
    for name,span in spans.items():
        lo,hi=span['ts'],span['ts']+span['dur']
        active=[e for e in gpu if e['ts']<hi and e['ts']+e['dur']>lo]
        union=merge(clipped(active,lo,hi))
        cats=defaultdict(list)
        for e in active: cats[category(e)].append(e)
        gaps=[];previous=lo
        for a,b in union:
            if a>previous:gaps.append((previous,a))
            previous=max(previous,b)
        if previous<hi:gaps.append((previous,hi))
        sync=[e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'synchronize' in e['name'].lower() and e['ts']<hi and e['ts']+e['dur']>lo]
        sync_names=defaultdict(list)
        for e in sync:sync_names[e['name']].append(e)
        stages[name]={'cpu_span_s':span['dur']/1e6,'gpu_activity_union_s':total(union)/1e6,
                      'uncovered_window_s':sum(b-a for a,b in gaps)/1e6,
                      'categories':{k:{'events':len(v),'clipped_sum_s':sum(b-a for a,b in clipped(v,lo,hi))/1e6,'clipped_union_s':total(clipped(v,lo,hi))/1e6} for k,v in cats.items()},
                      'sync_api':{k:{'events':len(v),'clipped_sum_s':sum(b-a for a,b in clipped(v,lo,hi))/1e6} for k,v in sync_names.items()}}
        if name=='P1/forward_backward':
            stages[name]['largest_uncovered_intervals']=[{'start_relative_s':(a-lo)/1e6,'duration_ms':(b-a)/1000,
                'overlapping_sync_api_names':sorted({e['name'] for e in sync if e['ts']<b and e['ts']+e['dur']>a})} for a,b in sorted(gaps,key=lambda x:x[1]-x[0],reverse=True)[:10]]
            stages[name]['gaps_over_100us']={'count':sum(b-a>100 for a,b in gaps),'total_s':sum(b-a for a,b in gaps if b-a>100)/1e6}
    fb=spans['P1/forward_backward'];lo,hi=fb['ts'],fb['ts']+fb['dur']
    kernel_stats=defaultdict(list)
    for e in gpu:
        if e['cat']=='kernel' and lo<=e['ts']<hi and 'deep_ep' in e['name'].lower():kernel_stats[e['name']].append(e['dur'])
    distributions={}
    for k,v in kernel_stats.items():
        v.sort();distributions[k]={'count':len(v),'sum_ms':sum(v)/1000,'median_us':v[len(v)//2],'p95_us':v[min(len(v)-1,int(.95*len(v)))],'max_us':v[-1]}
    tail=[]
    for e in gpu:
        api=apis.get(e.get('args',{}).get('correlation'))
        if api and lo<=api['ts']<hi and e['ts']+e['dur']>hi:
            tail.append(e)
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'stages':stages,
            'deepep_kernel_distributions':distributions,'forward_launched_gpu_tail_events':len(tail),
            'forward_launched_gpu_tail_max_s':max([0]+[(e['ts']+e['dur']-hi)/1e6 for e in tail]),
            'grouped_cpu_scopes':dict(Counter(e['name'] for rows in grouped.values() for e in rows)),
            'note':'Stage metrics are clipped to CPU annotations. GPU activity union is not hardware utilization. Sums may overlap; uncovered intervals include profiler/host effects, not proven communication waits. Expert GEMM attribution uses correlation to launch API contained in same-thread GroupedLinear scopes.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('traces',nargs='+',type=Path);args=ap.parse_args()
    results=[]
    for p in args.traces:
        r=analyze(p);results.append(r)
        print(p.name,r['stages']['P1/forward_backward']['cpu_span_s'],flush=True)
    args.output.write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'traces':results},indent=2)+'\n')


if __name__=='__main__':main()
