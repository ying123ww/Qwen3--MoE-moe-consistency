"""Verify scalar D2H and synchronization provenance in existing decode traces."""
import argparse
import bisect
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from summarize_remaining_inference import scope_at, scope_index


def audit(path):
    with gzip.open(path,'rt') as handle:events=json.load(handle)['traceEvents']
    scopes,_,_=scope_index(events)
    ops={e.get('args',{}).get('External id'):e for e in events if e.get('ph')=='X' and
         e.get('cat')=='cpu_op' and e.get('args',{}).get('External id') is not None}
    apis=defaultdict(list);items=defaultdict(list)
    for event in events:
        if event.get('ph')!='X':continue
        key=(event.get('pid'),event.get('tid'))
        if event.get('cat') in ('cuda_runtime','cuda_driver'):apis[key].append(event)
        if event.get('cat')=='cpu_op' and event.get('name')=='aten::item':items[key].append(event)
    for rows in [*apis.values(),*items.values()]:rows.sort(key=lambda event:event['ts'])
    api_ts={key:[event['ts'] for event in rows] for key,rows in apis.items()}
    item_ts={key:[event['ts'] for event in rows] for key,rows in items.items()}
    counts=Counter();nested=Counter();byte_counts=Counter();by_layer=Counter();example=None
    for event in events:
        if event.get('cat')!='gpu_memcpy' or 'DtoH' not in event.get('name',''):continue
        op=ops.get(event.get('args',{}).get('External id'))
        if not op or op['name']!='aten::_local_scalar_dense':continue
        if scope_at(scopes,op)!=('attention','decode'):continue
        key=(op.get('pid'),op.get('tid'));start=op['ts'];end=start+op['dur']
        counts['copies']+=1
        position=bisect.bisect_right(item_ts.get(key,[]),start)-1
        enclosing=items[key][position] if position>=0 else None
        if enclosing and enclosing['ts']<=start and enclosing['ts']+enclosing['dur']>=end:
            counts['enclosed_by_aten_item']+=1
        times=api_ts.get(key,[]);rows=apis.get(key,[])
        lo=bisect.bisect_left(times,start);hi=bisect.bisect_right(times,end)
        within=[row for row in rows[lo:hi] if row['ts']+row.get('dur',0)<=end]
        nested.update(row['name'] for row in within)
        if any('Synchronize' in row['name'] for row in within):counts['scalar_ops_with_sync_api']+=1
        if event.get('args',{}).get('bytes') is not None:byte_counts[str(event['args']['bytes'])]+=1
        if example is None:
            example=dict(cpu_item=enclosing,cpu_scalar=op,cpu_apis=within,gpu_copy=event)
    return dict(trace=str(path),counts=dict(counts),nested_cpu_apis=dict(nested),copy_byte_histogram=dict(byte_counts),
                example=example,limitation='Profiler provenance only; this does not measure unprofiled latency saved')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    rows=[]
    for mode in ('profile-baseline','profile-candidate'):
        for summary in sorted((args.profile_run/mode/'detail-summary').glob('rank-*.json')):
            detail=json.loads(summary.read_text());rank=int(summary.stem.split('-')[-1])
            result=audit(Path(detail['trace']));result.update(mode=mode,rank=rank)
            (args.output/f'{mode}-rank-{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
            row={key:value for key,value in result.items() if key!='example'};rows.append(row)
            print(json.dumps(row),flush=True)
    if len(rows)!=16:raise RuntimeError('Expected eight ranks in both modes')
    for row in rows:
        expected=6144 if row['mode']=='profile-candidate' else 0
        if any(row['counts'].get(key,0)!=expected for key in ('copies','enclosed_by_aten_item','scalar_ops_with_sync_api')):
            raise RuntimeError('Unexpected scalar copy callchain count')
        if expected and row['copy_byte_histogram']!={'1':6144}:raise RuntimeError('Unexpected copy size')
    (args.output/'complete.json').write_text(json.dumps(dict(complete=True,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
