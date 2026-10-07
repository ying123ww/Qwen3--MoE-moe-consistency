"""Copy-event overlap with CPU ranges, descriptive temporal attribution only."""
import gzip,json
from pathlib import Path
import sys

root=Path(sys.argv[1])
rows=[]
for path in sorted((root/'profiles/train').glob('train-*.json.gz')):
    with gzip.open(path,'rt') as f: events=json.load(f)['traceEvents']
    spans=[e for e in events if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e.get('name','').startswith('P1/')]
    copies=[e for e in events if e.get('ph')=='X' and e.get('cat')=='gpu_memcpy']
    result={}
    for s in spans:
        d=result.setdefault(s['name'],{})
        for e in copies:
            overlap=max(0,min(s['ts']+s['dur'],e['ts']+e['dur'])-max(s['ts'],e['ts']))
            if overlap:
                d[e['name']]=d.get(e['name'],0)+overlap/1e6
    rows.append(dict(file=path.name,copy_overlap_s=result,
        command_buffer_events=[{k:e[k] for k in ('name','cat','dur') if k in e} for e in events if e.get('name')=='Command Buffer Full']))
(root/'p1-copy-windows.json').write_text(json.dumps(rows,indent=2)+'\n')
print('wrote',root/'p1-copy-windows.json')
