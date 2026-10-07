"""Per-file GPU event attribution; summed durations are NOT elapsed time."""
import gzip
import json
from collections import Counter
from pathlib import Path
import sys


def summarize(path):
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    totals = Counter()
    counts = Counter()
    spans = Counter()
    intervals = []
    for e in data.get('traceEvents', []):
        if e.get('ph') != 'X':
            continue
        duration = e.get('dur', 0)
        category = e.get('cat', '')
        name = e.get('name', '')
        if category in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            key = category + ': ' + name
            totals[key] += duration
            counts[key] += 1
            intervals.append((e['ts'], e['ts'] + duration))
        elif category == 'user_annotation' and name.startswith('P1/'):
            spans[name] += duration
    intervals.sort()
    union = 0
    end = float('-inf')
    for a, b in intervals:
        union += max(0, b-max(a, end))
        end = max(end, b)
    return dict(file=str(path), gpu_events=sum(counts.values()), gpu_union_s=union/1e6,
                gpu_sum_s=sum(totals.values())/1e6,
                gpu_observed_window_s=(max(b for a,b in intervals)-intervals[0][0])/1e6 if intervals else 0,
                cpu_spans_s={k:v/1e6 for k,v in spans.items()},
                top_gpu_events=[dict(name=k,seconds=v/1e6,count=counts[k]) for k,v in totals.most_common(30)])


if __name__ == '__main__':
    root=Path(sys.argv[1])
    rows=[summarize(p) for p in sorted(root.glob('profiles/**/*.json.gz')) if 'merged' not in p.name.lower()]
    if not rows or not all(r['gpu_events'] for r in rows):
        raise RuntimeError('Missing GPU trace events; attribution incomplete')
    (root/'p1-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([dict(file=Path(r['file']).name,gpu_events=r['gpu_events']) for r in rows]))
