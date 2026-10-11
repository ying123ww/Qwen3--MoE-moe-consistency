"""Summarize source-marked GPU intervals without adding inclusive parents/children."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def summarize(record):
    if record['steps']!=32:raise ValueError('Expected the complete 32-replay window')
    grouped=defaultdict(list)
    for row in record['components']:
        value=row['last_replay_gpu_ms']
        if not math.isfinite(value) or value<0:raise ValueError('Invalid CUDA timing interval')
        grouped[row['component']].append(row)
    if len(grouped['model_forward'])!=1:raise ValueError('Missing/duplicated complete forward interval')
    result={}
    for name,rows in grouped.items():
        values=[r['last_replay_gpu_ms'] for r in rows]
        result[name]=dict(intervals=len(rows),inclusive_sum_ms=sum(values),median_interval_ms=statistics.median(values),
            min_interval_ms=min(values),max_interval_ms=max(values),layers=sorted({r['layer'] for r in rows}),
            depths=sorted({r['depth'] for r in rows}),paths=sorted({r['path'] for r in rows}))
    return dict(rank=record['rank'],mode=record['mode'],last_size=record['last_size'],steps=32,
        components=result,scope='One selected replay per rank; layer interval distributions are not independent experiments; inclusive rows cannot be added')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    report={}
    for mode in ('A','C'):
        directory=args.run/mode/'profile-evidence'
        if not directory.exists():continue
        data=[summarize(json.loads(p.read_text())) for p in sorted(directory.glob('components-rank-*.json'))]
        if {r['rank'] for r in data}!=set(range(8)):raise ValueError('Incomplete rank provenance')
        report[mode]=data
    if not report:raise ValueError('No actual component records')
    (args.run/'component-source-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.run/'component-source-summary.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['mode','rank','component','intervals','inclusive_sum_ms','median_interval_ms'])
        for mode,records in report.items():
            for record in records:
                if record['rank'] not in (0,4):continue
                for name,row in record['components'].items():
                    writer.writerow([mode,record['rank'],name,row['intervals'],row['inclusive_sum_ms'],row['median_interval_ms']])
    if set(report)=={'A','C'}:
        differences=[]
        for rank in (0,4):
            a=next(r for r in report['A'] if r['rank']==rank)['components']
            c=next(r for r in report['C'] if r['rank']==rank)['components']
            for name in sorted(set(a)|set(c)):
                left=a.get(name,{}).get('inclusive_sum_ms',0);right=c.get(name,{}).get('inclusive_sum_ms',0)
                differences.append(dict(rank=rank,component=name,A_inclusive_ms=left,C_inclusive_ms=right,delta_inclusive_ms=right-left))
        (args.run/'component-source-differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    print('SOURCE_COMPONENT_SUMMARY_COMPLETE',flush=True)


if __name__=='__main__':main()
