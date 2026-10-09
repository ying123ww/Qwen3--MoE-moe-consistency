"""CPU scopes and correlated GPU families; durations are not an elapsed partition."""
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from summarize_matmul import classify


def union(intervals):
    total, end = 0., float('-inf')
    for start, stop in sorted(intervals):
        total += max(0., stop - max(start, end))
        end = max(end, stop)
    return total


def scope_index(events):
    boundaries = defaultdict(list)
    counts = Counter()
    for index, event in enumerate(events):
        name = event.get('name', '')
        if event.get('cat') != 'user_annotation' or event.get('ph') != 'X' or event.get('dur', 0) <= 0:
            continue
        component = name.split('/')[1] if name.startswith('P1/') else None
        phase = ('decode' if name.startswith('step[DECODE') else 'prefill') if name.startswith('step[') else None
        if component is None and phase is None:
            continue
        if component: counts[component] += 1
        thread = (event.get('pid'), event.get('tid'))
        row = (component, phase, event['ts'])
        boundaries[thread].append((event['ts'], 1, index, row))
        boundaries[thread].append((event['ts'] + event['dur'], 0, index, row))
    result, exclusive = {}, Counter()
    for thread, edges in boundaries.items():
        edges.sort()
        active, starts, segments = {}, [], []
        previous = None
        for timestamp, opening, index, row in edges:
            if previous is not None and timestamp > previous:
                components = [r for r in active.values() if r[0] is not None]
                phases = [r for r in active.values() if r[1] is not None]
                component = max(components, key=lambda r:r[2])[0] if components else 'unscoped'
                phase = max(phases, key=lambda r:r[2])[1] if phases else 'unknown'
                starts.append(previous)
                segments.append((timestamp, component, phase))
                if component != 'unscoped': exclusive[component] += timestamp - previous
            if opening: active[index] = row
            else: active.pop(index, None)
            previous = timestamp
        result[thread] = (starts, segments)
    return result, counts, exclusive


def scope_at(index, event):
    starts, segments = index.get((event.get('pid'), event.get('tid')), ([], []))
    position = bisect_right(starts, event['ts']) - 1
    if position >= 0 and event['ts'] < segments[position][0]:
        return segments[position][1:]
    return 'unscoped', 'unknown'


def matrix_family(op):
    if op is None: return None
    dims = op.get('args', {}).get('Input Dims', [])
    if op.get('name') == 'aten::addmm': dims = dims[1:]
    if len(dims) < 2 or len(dims[0]) != 2 or len(dims[1]) != 2: return None
    m, k = dims[0]
    bk, n = dims[1]
    if k != bk: return None
    family = classify(k, n)
    return (family, m, k, n) if family != 'unclassified' else None


def summarize(events):
    scopes, scope_counts, exclusive = scope_index(events)
    launches = {e.get('args', {}).get('correlation'): e for e in events
                if e.get('ph') == 'X' and e.get('cat') in ('cuda_runtime', 'cuda_driver')
                and e.get('args', {}).get('correlation') is not None}
    ops = {e.get('args', {}).get('External id'): e for e in events
           if e.get('cat') == 'cpu_op' and e.get('name') in ('aten::mm', 'aten::addmm')
           and e.get('args', {}).get('External id') is not None}
    totals, counts, phase_totals = Counter(), Counter(), Counter()
    names, mm_shapes = Counter(), Counter()
    intervals, gpu_events, devices = [], [], set()
    unlinked = 0
    for event in events:
        category = event.get('cat')
        if event.get('ph') != 'X' or category not in ('kernel', 'gpu_memcpy', 'gpu_memset'): continue
        name, duration = event.get('name', ''), event.get('dur', 0)
        launch = launches.get(event.get('args', {}).get('correlation'))
        component, phase = scope_at(scopes, launch) if launch else ('unscoped', 'unknown')
        matrix = matrix_family(ops.get(event.get('args', {}).get('External id')))
        low = name.lower()
        if category != 'kernel': family = category
        elif 'nccl' in low: family = 'communication_nccl'
        elif 'notify_dispatch' in low or 'notify_combine' in low: family = 'communication_deepep_notify'
        elif 'fused_moe_kernel' in name: family = 'fused_moe'
        elif matrix:
            family = matrix[0] + '_matmul'
            mm_shapes[str(matrix)] += 1
        elif 'matmul_kernel_persistent' in name:
            family = 'unlinked_persistent_matmul'
            unlinked += 1
        elif component == 'router_logits': family = 'router_logits_other'
        elif component == 'router_select': family = 'router_select'
        elif component in ('residual_rms', 'other_rms'): family = component
        elif component in ('attention_qkv', 'attention_output'): family = component + '_other'
        elif component == 'attention': family = 'attention_other'
        elif component == 'moe': family = 'moe_other'
        elif component == 'logits': family = 'logits_other'
        else: family = 'unscoped_gpu'
        totals[family] += duration
        counts[family] += 1
        phase_totals[phase + ':' + family] += duration
        names[category + ':' + name] += duration
        intervals.append((event['ts'], event['ts'] + duration))
        gpu_events.append((event, launch))
        if category == 'kernel' and event.get('args', {}).get('device') is not None:
            devices.add(event['args']['device'])
    if not intervals: raise ValueError('No GPU events')
    late, queued, unknown, end = 0., 0., 0., None
    for event, launch in sorted(gpu_events, key=lambda item:item[0]['ts']):
        start, stop = event['ts'], event['ts'] + event['dur']
        if end is not None and start > end:
            gap = start - end
            if launch is None: unknown += gap
            else:
                before = min(gap, max(0., launch['ts'] - end))
                late += before
                queued += gap - before
        end = max(end, stop) if end is not None else stop
    api_totals, api_counts = Counter(), Counter()
    for event in events:
        if event.get('ph') == 'X' and event.get('cat') in ('cuda_runtime', 'cuda_driver'):
            api_totals[event.get('name', '')] += event.get('dur', 0)
            api_counts[event.get('name', '')] += 1
    return dict(gpu_devices=sorted(devices),gpu_events=sum(counts.values()),gpu_union_s=union(intervals)/1e6,
                gpu_sum_s=sum(totals.values())/1e6,
                gpu_window_s=(max(b for a,b in intervals)-min(a for a,b in intervals))/1e6,
                gpu_families={k:dict(count=counts[k],seconds=v/1e6) for k,v in totals.items()},
                gpu_phase_families_s={k:v/1e6 for k,v in phase_totals.items()},
                cpu_scope_counts=dict(scope_counts),cpu_scope_exclusive_s={k:v/1e6 for k,v in exclusive.items()},
                gpu_gap_s=dict(before_next_launch=late/1e6,after_launch_started=queued/1e6,unlinked=unknown/1e6),
                matrix_shapes=dict(mm_shapes),unlinked_persistent_matmul=unlinked,
                top_gpu_events=[dict(name=k,seconds=v/1e6) for k,v in names.most_common(25)],
                top_cpu_apis=[dict(name=k,count=api_counts[k],seconds=v/1e6) for k,v in api_totals.most_common(20)],
                limitation='Profiled durations. CPU scopes/API sums and GPU sums overlap and are not an elapsed partition. Before-launch gaps can include host scheduling and cross-rank waits, not solely CPU computation.')


def summarize_file(path):
    with gzip.open(path, 'rt') as handle: data = json.load(handle)
    result = summarize(data['traceEvents'])
    result['trace'] = str(path)
    info = data.get('distributedInfo', {})
    result['distributed_rank'] = info.get('rank') if isinstance(info, dict) else None
    return result


def collect(run):
    rows = []
    for path in sorted(run.glob('profiles/**/*.json.gz')):
        if 'merged' in path.name.lower(): continue
        row = summarize_file(path)
        if not row['cpu_scope_counts'].get('router_logits') or not row['cpu_scope_counts'].get('router_select'):
            raise ValueError('Required router scopes missing: ' + str(path))
        if row['unlinked_persistent_matmul']:
            raise ValueError('Unlinked persistent matmul: ' + str(path))
        rows.append(row)
    ranks = {r['distributed_rank'] for r in rows}
    devices = {tuple(r['gpu_devices']) for r in rows}
    if len(rows) != 8 or (ranks != set(range(8)) and devices != {(i,) for i in range(8)}):
        raise ValueError('Expected eight separate, GPU-labelled worker traces')
    (run/'component-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows
