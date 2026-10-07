"""CPU-only exclusive main-thread accounting and observed transfer overlap."""
import argparse
from bisect import bisect_right
from collections import defaultdict, Counter
import gzip
import hashlib
import heapq
import json
from pathlib import Path
from analyze_native_routepack_trace import merge, total


def intersection(a,b):
    a,b=merge(a),merge(b);i=j=0;out=[]
    while i<len(a) and j<len(b):
        lo,hi=max(a[i][0],b[j][0]),min(a[i][1],b[j][1])
        if hi>lo:out.append((lo,hi))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out


def exclusive(events,lo,hi,classify):
    points=defaultdict(list);points[lo];points[hi]
    for i,e in enumerate(events):
        a,b=max(lo,e['ts']),min(hi,e['ts']+e['dur'])
        if b>a:points[a].append((1,i));points[b].append((-1,i))
    live=set();heap=[];out=defaultdict(list);times=sorted(points)
    for t,nxt in zip(times,times[1:]):
        for action,i in points[t]:
            if action<0:live.discard(i)
            else:
                live.add(i);e=events[i]
                heapq.heappush(heap,(0 if e['cat'] in ('cuda_runtime','cuda_driver') else 1,-e['ts'],e['dur'],i))
        while heap and heap[0][3] not in live:heapq.heappop(heap)
        label=classify(events[heap[0][3]]) if heap else 'unattributed_host'
        out[label].append((t,nxt))
    return {k:merge(v) for k,v in out.items()}


def analyze(path):
    with gzip.open(path,'rt') as f:es=json.load(f)['traceEvents']
    span=next(e for e in es if e.get('cat')=='user_annotation' and e['name']=='P1/optimizer_step')
    lo,hi=span['ts'],span['ts']+span['dur'];thread=(span['pid'],span['tid'])
    cpu=[e for e in es if e.get('ph')=='X' and e.get('dur',0)>0 and (e.get('pid'),e.get('tid'))==thread and e.get('cat') in ('cpu_op','cuda_runtime','cuda_driver') and e['ts']<hi and e['ts']+e['dur']>lo]
    api={e['args']['correlation']:e for e in cpu if e['cat'] in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
    directions={};gpu=defaultdict(list);bytes_by=Counter();copies=[]
    for e in es:
        if e.get('cat')!='gpu_memcpy':continue
        r=api.get(e.get('args',{}).get('correlation'))
        if not r or not lo<=r['ts']<hi:continue
        direction='H2D' if 'HtoD' in e['name'] else 'D2H' if 'DtoH' in e['name'] else 'other_copy'
        directions[e['args']['correlation']]=direction;gpu[direction].append((e['ts'],e['ts']+e['dur']));bytes_by[direction]+=e['args'].get('bytes',0);copies.append((e,r,direction))
    math={'aten::abs','aten::mul','aten::mul_','aten::add_','aten::addcdiv_','aten::add','aten::neg'}
    def classify(e):
        if e['cat'] in ('cuda_runtime','cuda_driver'):
            if 'synchronize' in e['name'].lower():return 'wait/'+e['name']
            direction=directions.get(e.get('args',{}).get('correlation'))
            if direction:return 'api/'+direction
            return 'api/other'
        if e['name'] in math:return 'math/'+e['name']
        if e['name'] in ('aten::empty','aten::empty_strided','aten::empty_like'):return 'allocation_ops'
        if e['name'] in ('aten::copy_','aten::_to_copy','aten::to'):return 'copy_ops_outside_api'
        return 'other_cpu_ops/'+e['name']
    parts=exclusive(cpu,lo,hi,classify);seconds={k:total(v)/1e6 for k,v in parts.items()}
    assert abs(sum(seconds.values())-(hi-lo)/1e6)<1e-7
    math_intervals=merge([p for k,v in parts.items() if k.startswith('math/') for p in v])
    scopes=sorted([e for e in es if e.get('cat')=='user_annotation' and e.get('name')=='Optimizer.step#StatelessAdam.step' and (e.get('pid'),e.get('tid'))==thread and lo<=e['ts']<hi],key=lambda e:e['ts'])
    assert all(a['ts']+a['dur']<=b['ts']+.01 for a,b in zip(scopes,scopes[1:])), 'Overlapping suboptimizer scopes'
    starts=[e['ts'] for e in scopes];grouped=defaultdict(lambda:{'copies':0,'scope_ids':set(),'api_s':0,'gpu_s':0});inside=0
    for e,r,d in copies:
        if d!='H2D':continue
        idx=bisect_right(starts,r['ts'])-1
        if idx>=0 and r['ts']+r['dur']<=scopes[idx]['ts']+scopes[idx]['dur']+.01:inside+=1
        else:idx=-1
        row=grouped[e['args'].get('bytes',0)];row['copies']+=1;row['scope_ids'].add(idx);row['api_s']+=r['dur']/1e6;row['gpu_s']+=e['dur']/1e6
    groups=[]
    for size,row in sorted(grouped.items()):
        ids=row.pop('scope_ids');row.update(bytes_each=size,suboptimizer_scopes=len(ids),scope_total_s=sum(scopes[i]['dur']/1e6 for i in ids if i>=0));groups.append(row)
    overlap={d:{'union_s':total(v)/1e6,'overlap_cpu_math_s':total(intersection(v,math_intervals))/1e6,'tail_after_optimizer_s':total([(max(hi,a),b) for a,b in v if b>hi])/1e6} for d,v in gpu.items()}
    for direction in ('H2D','D2H'):
        overlap[direction]['overlap_same_direction_api_s']=total(intersection(gpu[direction],parts.get('api/'+direction,[])))/1e6
        first_math=math_intervals[0][0] if math_intervals else hi
        overlap[direction]['before_first_cpu_math_s']=total([(a,min(b,first_math)) for a,b in gpu[direction] if a<first_math])/1e6
    overlap['H2D_D2H_intersection_s']=total(intersection(gpu['H2D'],gpu['D2H']))/1e6
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'optimizer_wall_s':(hi-lo)/1e6,'exclusive_main_thread_s':seconds,'main_thread':thread,'other_observed_cpu_tids':sorted({e['tid'] for e in es if e.get('cat')=='cpu_op' and e.get('pid')==span['pid'] and lo<=e['ts']<hi and e['tid']!=span['tid']}),'suboptimizer_count':len(scopes),'h2d_calls_inside_suboptimizer':inside,'transfer_bytes':dict(bytes_by),'gpu_observed_overlap':overlap,'h2d_by_size':groups}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('traces',type=Path,nargs='+');args=ap.parse_args();rows=[]
    for p in args.traces:
        r=analyze(p);rows.append(r);print(p.name,r['exclusive_main_thread_s'],flush=True)
    args.output.write_text(json.dumps({'method':'Exclusive optimizer main-thread wall attribution: CUDA API takes priority; innermost CPU op otherwise. CPU math durations include internal thread waits; they are not CPU core time. GPU overlap is observed intersection, not causal speedup. Missing host spans are unattributed, not assumed Python overhead.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':rows},indent=2)+'\n')


if __name__=='__main__':main()
