"""Actual graph-outside prefill CPU/GPU source attribution, per process/thread."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
from summarize_full_graph_timeline import connect,labels,associate,union_ns

def report(db,record):
    rank=record['rank'];pid=record['pid']
    mapped=list(db.execute('select distinct globalPid from PROCESSES where pid=?',(pid,)))
    if len(mapped)!=1:raise RuntimeError('PID mapping ambiguous')
    gp=mapped[0][0];ranges=[x for x in labels(db) if x[2]>>24==gp>>24]
    prefill=[x for x in ranges if x[3].startswith(f'P11/eager_prefill/rank={rank}/')]
    if len(prefill)!=record['counts']['prefill']:raise RuntimeError('Actual prefill NVTX count mismatch')
    runtime=list(db.execute('select e.*,s.value name from CUPTI_ACTIVITY_KIND_RUNTIME e join StringIds s on s.id=e.nameId where e.globalTid>>24=?',(gp>>24,)))
    tables={x[0] for x in db.execute("select name from sqlite_master where type='table'")}
    if 'CUPTI_ACTIVITY_KIND_DRIVER' in tables:
        runtime+=list(db.execute('select e.*,s.value name from CUPTI_ACTIVITY_KIND_DRIVER e join StringIds s on s.id=e.nameId where e.globalTid>>24=?',(gp>>24,)))
    sources=associate([(x['start'],x['globalTid'],x['correlationId']) for x in runtime],ranges)
    modes=associate([(x['start'],x['globalTid'],x['correlationId']) for x in runtime],
                    [x for x in ranges if x[3].split('/')[1] in ('eager_prefill','eager_decode','eager_idle')])
    kernels=list(db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=?',(gp,)))
    if not kernels:raise RuntimeError('No actual prefill GPU kernels')
    parts=defaultdict(list);mode_parts=defaultdict(lambda:defaultdict(list));source_counts=Counter();graph_correlations={x['correlationId'] for x in runtime if 'GraphLaunch' in x['name']};graph_kernels=[]
    for x in kernels:
        if x['correlationId'] in graph_correlations:
            graph_kernels.append(x);continue
        label=sources.get(x['correlationId']);part=label.split('/')[1] if label else 'unmapped'
        if label and f'/rank={rank}/' not in label:raise RuntimeError('Source crossed ranks')
        parts[part].append((x['start'],x['end']));source_counts[label or 'unmapped']+=1
        kind=modes.get(x['correlationId'],'P11/outside_eager_forward').split('/')[1]
        mode_parts[kind][part].append((x['start'],x['end']))
    for required in ('qkv','output_projection','attention_core','router','experts','deepep_dispatch','deepep_combine','sampling'):
        if required not in parts:raise RuntimeError('Actual prefill/sampling component missing: '+required)
    for required in ('qkv','output_projection','attention_core','router','experts'):
        if required not in mode_parts['eager_prefill']:raise RuntimeError('Prefill component not isolated: '+required)
    native_calls=sum(x['name'].startswith('cudaGraphLaunch') for x in runtime)
    if native_calls!=record['counts'].get('full_replays',0):raise RuntimeError('Incidental native graph count mismatch')
    intervals=[(x['start'],x['end']) for x in kernels];copy_counts=Counter();copy_ms=0
    for table in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        if table in tables:
            for x in db.execute(f'select * from {table} where globalPid=?',(gp,)):
                intervals.append((x['start'],x['end']));copy_counts[str(x['copyKind']) if table.endswith('MEMCPY') else 'memset']+=1;copy_ms+=(x['end']-x['start'])/1e6
    start=min(x[0] for x in intervals);end=max(x[1] for x in intervals)
    api=defaultdict(list)
    for x in runtime:api[x['name']].append((x['start'],x['end']))
    return dict(rank=rank,pid=pid,prefill_calls=record['counts']['prefill'],actual_shapes=record['shapes'],incidental_full_replays=native_calls,incidental_graph_kernel_activities=len(graph_kernels),gpu_activity_union_ms=union_ns(intervals)/1e6,gpu_activity_duration_sum_ms=sum(b-a for a,b in intervals)/1e6,gpu_activity_envelope_ms=(end-start)/1e6,gpu_inactive_in_envelope_ms=(end-start-union_ns(intervals))/1e6,kernels=len(kernels),components={k:dict(kernel_activities=len(v),duration_sum_ms=sum(b-a for a,b in v)/1e6,activity_union_ms=union_ns(v)/1e6) for k,v in parts.items()},components_by_eager_mode={mode:{k:dict(kernel_activities=len(v),duration_sum_ms=sum(b-a for a,b in v)/1e6,activity_union_ms=union_ns(v)/1e6) for k,v in rows.items()} for mode,rows in mode_parts.items()},memcpy_kind_counts=dict(copy_counts),copy_memset_duration_sum_ms=copy_ms,cuda_api={k:dict(calls=len(v),duration_sum_ms=sum(b-a for a,b in v)/1e6) for k,v in api.items()},source_counts=dict(source_counts),scope='Actual uncached8prompt1218input,8firstresponse tokens prefill+sampling and explicitly reported incidental decode/idle/graph. Source uses runtime/driver correlation and innermost NVTX on same process/thread. Incidental graph kernels excluded from eager component sums, included in whole-window union. Eager prefill/decode/idle and outside-forward activities separated by containing NVTX. Kernel/API sums are not wall partitions or uninstrumented latency.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if json.loads((a.run/'state.json').read_text())['status']!='complete':raise RuntimeError('GPU run incomplete')
    out={}
    for mode in ('A','C'):
        r=a.run/mode;db=connect(r/'nsys/prefill.sqlite');out[mode]=[report(db,json.loads((r/'profile-evidence'/f'outer-rank-{rank}.json').read_text())) for rank in (0,4)];db.close()
    (a.run/'outer-timeline-summary.json').write_text(json.dumps(out,indent=2)+'\n');print('P11_ACTUAL_PREFILL_SOURCE_SUMMARY_PASSED')

if __name__=='__main__':main()
