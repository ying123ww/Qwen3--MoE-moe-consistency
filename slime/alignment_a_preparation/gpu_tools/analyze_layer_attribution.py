"""CPU-only strict EP collective join for opt-in layer attribution rounds."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values)-1)] - values[lo]) * (index-lo)


def stats(values):
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values), p95=percentile(values,.95), maximum=max(values))


def topology_and_layout(topology_path, schedule_path):
    topology, schedule = read(topology_path), read(schedule_path)
    ranks = {r['global_rank']: r for r in topology['ranks']}
    if len(ranks) != topology['world_size']:
        raise ValueError('Incomplete topology')
    groups = sorted({tuple(r['groups']['ep']) for r in ranks.values()})
    mapping = {}
    for group_id, members in enumerate(groups):
        for rank in members:
            if rank in mapping or tuple(ranks[rank]['groups']['ep']) != members:
                raise ValueError('Overlapping/inconsistent EP groups')
            if any(len(ranks[rank]['groups'][key]) != 1 for key in ('pp','tp','cp')):
                raise ValueError('This ordinal slot join requires PP=TP=CP=1')
            mapping[rank] = group_id
    steps = {c['step'] for c in schedule['cells']}
    if len(steps) != 1:
        raise ValueError('One training step per context required')
    slots = defaultdict(set)
    cells = set()
    for cell in schedule['cells']:
        rank = cell['source_rank']
        key = (rank,cell['slot'])
        if key in cells or cell['group_id'] != mapping[rank]:
            raise ValueError('Duplicate slot or schedule/topology group mismatch')
        cells.add(key)
        slots[rank].add(cell['slot'])
    if set(slots) != set(mapping):
        raise ValueError('Incomplete schedule rank set')
    for rank, indices in slots.items():
        if indices != set(range(len(indices))):
            raise ValueError('Schedule slots must be contiguous zero-based ordinals')
    return groups, mapping, slots


def analyze(path, groups, mapping, slots):
    data = read(path)
    if not data.get('passed'):
        raise ValueError(f'Timing gate did not pass: {path}')
    rounds = [it for it in data['iterations'] if it and it[0]['phase']=='attribution']
    if not rounds:
        raise ValueError('No attribution rounds')
    joined = []
    expected_keys = None
    coverage = []
    for round_index, iteration in enumerate(rounds):
        rank_rows = {r['rank']:r for r in iteration}
        if len(rank_rows) != len(iteration) or set(rank_rows) != set(mapping):
            raise ValueError('Duplicate or missing attribution ranks')
        tables = {}
        for rank, result in rank_rows.items():
            trace = result['layer_attribution']
            if result['phase'] != 'attribution' or trace['rank'] != rank or trace['failed']:
                raise ValueError('Invalid attribution phase/rank/status')
            coverage.append(dict(round=round_index, rank=rank, **trace['coverage']))
            events = trace['events']
            ids = {e['event_id']:e for e in events}
            if len(ids) != len(events):
                raise ValueError('Duplicate event IDs')
            table = {}
            for e in events:
                key = (e['chunk'],e['stage'],e['slot'],e['layer'],e['kind'])
                if key in table or not e['completed'] or not isinstance(e['elapsed_ms'],(int,float)) or not math.isfinite(e['elapsed_ms']) or e['elapsed_ms'] < 0:
                    raise ValueError('Duplicate/incomplete/nonfinite event')
                if e['stage'] not in ('replay_forward','replay_backward') or e['kind'] not in ('moe','experts') or e['chunk'] != 0:
                    raise ValueError('Unexpected stage/kind/chunk; explicit mapping needed')
                if e['slot'] not in slots[rank]:
                    raise ValueError('Attribution slot absent in schedule')
                if e['kind']=='experts':
                    parent = ids.get(e['parent_moe_event_id'])
                    if parent is None or parent['kind'] != 'moe' or any(parent[k] != e[k] for k in ('chunk','stage','slot','layer')):
                        raise ValueError('Unpaired/misaligned expert call; do not join as full MoE')
                table[key] = e['elapsed_ms']
            layers = {t['layer'] for t in trace['targets']}
            required = {(0,stage,slot,layer,kind) for stage in ('replay_forward','replay_backward') for slot in slots[rank] for layer in layers for kind in ('moe','experts')}
            if set(table) != required:
                raise ValueError('Incomplete layer-slot-stage-kind coverage')
            if expected_keys is None:
                expected_keys = set(table)
            elif set(table) != expected_keys:
                raise ValueError('Layer-slot-stage-kind keys differ across ranks/rounds')
            tables[rank] = table
        for group_id,members in enumerate(groups):
            for key in sorted(expected_keys):
                times = [tables[rank][key] for rank in members]
                average = statistics.mean(times)
                joined.append(dict(round=round_index,group_id=group_id,ranks=list(members),chunk=key[0],stage=key[1],slot=key[2],layer=key[3],kind=key[4],rank_ms=times,
                    max_ms=max(times),mean_ms=average,min_ms=min(times),max_minus_mean_ms=max(times)-average,
                    max_over_mean=max(times)/average if average else None,
                    max_duration_ranks=[rank for rank,value in zip(members,times) if value==max(times)]))
    grouped = defaultdict(list)
    for row in joined:
        grouped[(row['group_id'],row['stage'],row['kind'])].append(row)
    summary = []
    for (group,stage,kind),rows in sorted(grouped.items()):
        summary.append(dict(group_id=group,ranks=list(groups[group]),stage=stage,kind=kind,
            collective_count=len(rows),rank_duration_ms=stats([t for row in rows for t in row['rank_ms']]),
            max_duration_ms=stats([row['max_ms'] for row in rows]),
            duration_spread_ms=stats([row['max_minus_mean_ms'] for row in rows]),
            max_over_mean=stats([row['max_over_mean'] for row in rows]),
            max_duration_rank_counts=dict(Counter(rank for row in rows for rank in row['max_duration_ranks']))))
    measured = [max(r['full_step_s'] for r in it) for it in data['iterations'] if it and it[0]['phase']=='measured']
    return dict(path=str(Path(path).resolve()),sha256=digest(path),coverage=coverage,summary=summary,
        formal_measured_max_rank_step_s=stats(measured) if measured else None,
        attribution_rounds=len(rounds)), joined, expected_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology',type=Path,required=True)
    parser.add_argument('--schedule',type=Path,required=True,help='Matching slot layout; different sample assignments are allowed, changed slot count is rejected')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('timings',type=Path,nargs='+')
    args = parser.parse_args()
    groups,mapping,slots = topology_and_layout(args.topology,args.schedule)
    reports=[]; all_joined=[]; expected=None
    for path in args.timings:
        report,joined,keys=analyze(path,groups,mapping,slots)
        if expected is not None and keys!=expected:
            raise ValueError('Variants have different layer-slot-count; cannot compare')
        expected=keys;reports.append(report);all_joined.append(dict(path=str(path.resolve()),collectives=joined))
    limitations = [
        'Durations are current-stream inclusive intervals, not isolated expert compute. Communication dependencies, host enqueue gaps and rank arrival skew can affect them.',
        'Maximum duration rank is not necessarily the latest-finishing rank: events contain durations without cross-rank aligned absolute timestamps.',
        'Forward and replay-backward recompute are separate; recompute timings do not include all autograd backward kernels.',
        'MoE and expert intervals are nested: do not add their durations. Sums of per-collective rank maxima are not step time.',
        'Ordinal slot join requires one train step/context, PP=TP=CP=1, matching replay order and validated equal layer-slot-stage coverage.',
        'One attributed round of a fixed batch gives descriptive evidence, not a causal routing-skew proof or independent batch replication.']
    output=dict(topology_path=str(args.topology.resolve()),topology_sha256=digest(args.topology),schedule_path=str(args.schedule.resolve()),schedule_sha256=digest(args.schedule),
        ep_groups=groups,rank_div_four_matches_topology=all(rank//4==group for rank,group in mapping.items()),limitations=limitations,results=reports)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    args.output.with_name(args.output.stem+'-collectives.json').write_text(json.dumps(all_joined)+'\n')
    for report in reports:
        print(Path(report['path']).parent.name, 'validated',len(report['coverage']),'rank-rounds; step median',report['formal_measured_max_rank_step_s']['median'])
        for row in report['summary']:
            print(row['group_id'],row['stage'],row['kind'],'max/mean median',round(row['max_over_mean']['median'],3),'p95',round(row['max_over_mean']['p95'],3),'max-minus-mean median ms',round(row['duration_spread_ms']['median'],3))


if __name__=='__main__':
    main()
