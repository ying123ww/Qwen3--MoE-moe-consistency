"""Nsight P11 attribution using capture node provenance and activity interval unions."""
import argparse
from collections import Counter,defaultdict
import heapq
import json
from pathlib import Path
import sqlite3


def union_ns(intervals):
    total=0;right=None
    for start,end in sorted(intervals):
        if end<start:raise ValueError('Negative activity interval')
        if right is None or start>right:total+=end-start;right=end
        elif end>right:total+=end-right;right=end
    return total


def associate(points,ranges):
    """Map points to the innermost containing source range on the same thread."""
    by_thread=defaultdict(list);events=defaultdict(list);result={}
    for start,end,tid,label in ranges:
        if end is not None and end>=start:by_thread[tid].append((start,end,label))
    for stamp,tid,key in points:events[tid].append((stamp,key))
    for tid,items in events.items():
        bounds=sorted(by_thread[tid]);cursor=0;ends=[];starts=[];active={}
        for stamp,key in sorted(items):
            while cursor<len(bounds) and bounds[cursor][0]<=stamp:
                start,end,label=bounds[cursor];active[cursor]=label
                heapq.heappush(ends,(end,cursor));heapq.heappush(starts,(-start,-cursor,cursor));cursor+=1
            while ends and ends[0][0]<stamp:
                _,index=heapq.heappop(ends);active.pop(index,None)
            while starts and starts[0][2] not in active:heapq.heappop(starts)
            if starts:result[key]=active[starts[0][2]]
    return result


def resolve_node(key,direct,parents):
    visited=set()
    while key not in visited:
        visited.add(key)
        if key in direct:return direct[key]
        if key not in parents:return None
        key=parents[key]
    raise ValueError('Cyclic graph node provenance')


def connect(path):
    db=sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    return db


def labels(db):
    return [(r['start'],r['end'],r['globalTid'],r['label']) for r in db.execute(
        "select e.start,e.end,e.globalTid,coalesce(e.text,s.value) label from NVTX_EVENTS e "
        "left join StringIds s on s.id=e.textId where coalesce(e.text,s.value) like 'P11/%'")]


def node_provenance(db):
    nodes=list(db.execute('select * from CUDA_GRAPH_NODE_EVENTS'))
    # Nsight graphNodeId is process-local: the 8 workers reuse the same IDs.
    # Scope both original and cloned IDs by the serialized process identifier.
    direct=associate([(r['start'],r['globalTid'],(r['globalTid']>>24,r['graphNodeId'])) for r in nodes],labels(db))
    parents={(r['globalTid']>>24,r['graphNodeId']):(r['globalTid']>>24,r['originalGraphNodeId'])
        for r in nodes if r['originalGraphNodeId'] is not None}
    return direct,parents


def rank_report(db,record,source):
    rank=record['rank'];pid=record['pid']
    pids=list(db.execute('select distinct globalPid from PROCESSES where pid=?',(pid,)))
    if len(pids)!=1:raise ValueError(f'Unambiguous Nsight PID mapping missing: {pid}')
    global_pid=pids[0][0]
    ranges=[r for r in labels(db) if r[2]>>24==global_pid>>24]
    steps=[r for r in ranges if r[3].startswith(f'P11/steady_decode/rank={rank}/')]
    expected={f'step={i}/' for i in range(32)}
    if len(steps)!=32 or {f'step={r[3].split("/step=")[1].split("/")[0]}/' for r in steps}!=expected:
        raise ValueError(f'Incomplete rank {rank} steady window')
    runtime=list(db.execute('select e.*,s.value name from CUPTI_ACTIVITY_KIND_RUNTIME e '
        'join StringIds s on s.id=e.nameId where (e.globalTid>>24)=?',(global_pid>>24,)))
    full_launches=[r for r in runtime if (r['name']=='cudaGraphLaunch' or r['name'].startswith('cudaGraphLaunch_v')) and any(
        start<=r['start'] and r['end']<=end and tid==r['globalTid'] for start,end,tid,_ in steps)]
    if len(full_launches)!=32:raise ValueError(f'Expected 32 native graph launches, rank {rank}: {len(full_launches)}')
    correlations={r['correlationId'] for r in full_launches}
    kernels=list(db.execute('select e.*,s.value name from CUPTI_ACTIVITY_KIND_KERNEL e '
        'join StringIds s on s.id=e.demangledName where e.globalPid=?',(global_pid,)))
    graph_kernels=[r for r in kernels if r['graphNodeId'] is not None and r['correlationId'] in correlations]
    if not graph_kernels:raise ValueError('No graph node GPU activities correlated to full replay')
    cpu_sources=associate([(r['start'],r['globalTid'],r['correlationId']) for r in runtime],ranges)
    direct,parents=source;grouped=defaultdict(list);kernel_names=Counter();unmapped=Counter()
    provenance=Counter();unmapped_graph=Counter();named_intervals=defaultdict(list)
    for r in kernels:
        label=resolve_node((global_pid>>24,r['graphNodeId']),direct,parents) if r['graphNodeId'] is not None else None
        if r['correlationId'] in correlations and label is None:unmapped_graph[r['name']]+=1
        if label is None:label=cpu_sources.get(r['correlationId'])
        if label is not None:
            if f'/rank={rank}/' not in label:raise ValueError('Source provenance crossed ranks')
            provenance[(label,r['name'])]+=1
        component=label.split('/')[1] if label else 'unmapped'
        grouped[component].append((r['start'],r['end']))
        kernel_names[(component,r['name'])]+=1
        named_intervals[(component,r['name'])].append((r['start'],r['end']))
        if label is None:unmapped[r['name']]+=1
    activities=[(r['start'],r['end']) for r in kernels]
    copies=[]
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    for name in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        if name in tables:
            rows=list(db.execute(f'select * from {name} where globalPid=?',(global_pid,)))
            activities.extend((r['start'],r['end']) for r in rows)
            if name.endswith('MEMCPY'):
                copies=[dict(r) for r in rows]
    left=min([r[0] for r in steps]+[r[0] for r in activities]);right=max([r[1] for r in steps]+[r[1] for r in activities])
    busy=union_ns(activities);summed=sum(end-start for start,end in activities)
    api=defaultdict(list)
    for r in runtime:api[r['name']].append((r['start'],r['end']))
    per_replay=[]
    for launch in sorted(full_launches,key=lambda r:r['start']):
        intervals=[(r['start'],r['end']) for r in graph_kernels if r['correlationId']==launch['correlationId']]
        if not intervals:raise ValueError('A native replay has no GPU node activity')
        per_replay.append(dict(correlation_id=launch['correlationId'],kernel_activities=len(intervals),
            kernel_envelope_ms=(max(b for _,b in intervals)-min(a for a,_ in intervals))/1e6,
            kernel_activity_union_ms=union_ns(intervals)/1e6,
            host_graph_launch_ms=(launch['end']-launch['start'])/1e6))
    return dict(rank=rank,pid=pid,global_pid=global_pid,steps=32,full_graph_launches=32,
        full_graph_kernel_activities=len(graph_kernels),host_and_gpu_envelope_ms=(right-left)/1e6,
        gpu_activity_union_ms=busy/1e6,gpu_activity_duration_sum_ms=summed/1e6,
        gpu_overlap_excess_ms=(summed-busy)/1e6,gpu_inactive_in_envelope_ms=(right-left-busy)/1e6,
        components={k:dict(kernel_activities=len(v),duration_sum_ms=sum(b-a for a,b in v)/1e6,
            activity_union_ms=union_ns(v)/1e6) for k,v in grouped.items()},
        kernel_provenance=[dict(component=k,name=n,count=v,
            duration_sum_ms=sum(b-a for a,b in named_intervals[(k,n)])/1e6,
            activity_union_ms=union_ns(named_intervals[(k,n)])/1e6) for (k,n),v in kernel_names.items()],
        per_replay=per_replay,
        source_provenance=[dict(source=k,name=n,count=v) for (k,n),v in provenance.items()],
        full_graph_unmapped_kernel_activities=dict(unmapped_graph),
        unmapped_kernel_activities=dict(unmapped),memcpy= copies,
        cuda_api={k:dict(calls=len(v),duration_sum_ms=sum(b-a for a,b in v)/1e6,
            interval_union_ms=union_ns(v)/1e6) for k,v in api.items()},
        scope='32 selected instrumented replays. GPU union is per rank/device across streams. Inactive time includes host/dependency waits; not all is CPU launch cost. Component sums and inclusive source event intervals are not additive wall partitions.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();report={}
    for mode in ('A','C'):
        folder=args.run/mode;windows=[]
        for path in (folder/'nsys').glob('window-*.sqlite'):
            db=connect(path);windows.append((db,labels(db)))
        capture=[db for db,ranges in windows if not any('/steady_decode/' in r[3] for r in ranges)]
        steady=[db for db,ranges in windows if any('/steady_decode/' in r[3] for r in ranges)]
        if len(capture)!=1 or len(steady)!=1:raise ValueError('Expected one capture and one selected-replay timeline')
        source=node_provenance(capture[0])
        report[mode]=[rank_report(steady[0],json.loads((folder/'profile-evidence'/f'components-rank-{r}.json').read_text()),source) for r in (0,4)]
        for db,_ in windows:db.close()
    (args.run/'nsys-timeline-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NSYS_TIMELINE_SUMMARY_COMPLETE',flush=True)


if __name__=='__main__':main()
