"""Separate GPU event kinds, launch API layers, and Q/K CPU operators."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from summarize_remaining_inference import scope_index, scope_at


def summarize(events):
    scopes, counts, exclusive = scope_index(events)
    launches = {}
    for category in ('cuda_driver', 'cuda_runtime'):
        for event in events:
            if event.get('ph') == 'X' and event.get('cat') == category:
                correlation = event.get('args', {}).get('correlation')
                if correlation is not None:
                    launches[correlation] = event
    operations = {e.get('args', {}).get('External id'): e for e in events
                  if e.get('ph') == 'X' and e.get('cat') == 'cpu_op'
                  and e.get('args', {}).get('External id') is not None}
    categories, api_counts = Counter(), Counter()
    gpu_groups, cpu_groups = defaultdict(Counter), defaultdict(Counter)
    gpu_seconds, cpu_seconds = defaultdict(Counter), defaultdict(Counter)
    cpu_shapes, annotation_shapes = Counter(), Counter()
    for event in events:
        if event.get('ph') != 'X':
            continue
        category, name = event.get('cat'), event.get('name', '')
        duration = event.get('dur', 0) / 1e6
        if category == 'user_annotation' and name.startswith('P1/'):
            component, phase = scope_at(scopes, event)
            fields = dict(part.split('=', 1) for part in name.split('/')[2:] if '=' in part)
            annotation_shapes[json.dumps([name.split('/')[1], phase, fields], sort_keys=True)] += 1
        if category in ('cuda_runtime', 'cuda_driver'):
            if 'launchkernel' in name.lower():
                api_counts[f'{category}:{name}'] += 1
        if category == 'cpu_op':
            component, phase = scope_at(scopes, event)
            if component.startswith(('q_norm', 'k_norm', 'qk_')):
                key = component + ':' + phase
                cpu_groups[key][name] += 1
                cpu_seconds[key][name] += duration
                cpu_shapes[json.dumps([component, phase, name, event.get('args', {}).get('Input Dims', []),
                                      event.get('args', {}).get('Input type', [])])] += 1
        if category not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        categories[category] += 1
        launch = launches.get(event.get('args', {}).get('correlation'))
        component, phase = scope_at(scopes, launch) if launch else ('unscoped', 'unknown')
        key = component + ':' + phase
        operator = operations.get(event.get('args', {}).get('External id'))
        opname = operator.get('name', 'unlinked') if operator else 'unlinked'
        label = category + ':' + opname + ':' + name
        gpu_groups[key][label] += 1
        gpu_seconds[key][label] += duration
    return dict(gpu_event_categories=dict(categories), kernel_launch_api_counts=dict(api_counts),
        scope_call_counts=dict(counts), cpu_scope_exclusive_s={k:v/1e6 for k,v in exclusive.items()},
        gpu_components={k:dict(count=sum(v.values()), seconds=sum(gpu_seconds[k].values()),
            operators=[dict(name=n, count=c, seconds=gpu_seconds[k][n]) for n,c in v.most_common()])
            for k,v in gpu_groups.items()},
        qk_cpu_ops={k:[dict(name=n,count=c,seconds=cpu_seconds[k][n]) for n,c in v.most_common()]
                    for k,v in cpu_groups.items()},
        qk_cpu_op_shapes=dict(cpu_shapes), annotation_shapes=dict(annotation_shapes),
        limitation='API layers are reported separately; nested CPU ops overlap. GPU counts include memcopies/memsets. Durations are not an elapsed partition.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.trace, 'rt') as handle:
        data = json.load(handle)
    result = summarize(data['traceEvents'])
    result.update(trace=str(args.trace), distributed_info=data.get('distributedInfo'))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), gpu_event_categories=result['gpu_event_categories'],
                          kernel_launch_api_counts=result['kernel_launch_api_counts'])), flush=True)


if __name__ == '__main__':
    main()
