"""CPU-only Kineto GPU activity attribution; no host-time-as-kernel-time fallback."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
        else:out.append((a,b))
    return out


def duration(intervals):
    return sum(b-a for a,b in merge(intervals))


def metrics(events):
    intervals=[(e['ts'],e['ts']+e['dur']) for e in events]
    return dict(events=len(events),duration_sum_ms=sum(e['dur'] for e in events)/1000,
        interval_union_ms=duration(intervals)/1000,
        streams=sorted({e.get('args',{}).get('stream') for e in events if e.get('args',{}).get('stream') is not None}))


def component(name):
    parts=name.split('/')
    return parts[-2] if len(parts)>=3 else name


def kernel_bucket(event):
    if event['cat'] == 'gpu_memcpy':return 'memcpy'
    if event['cat'] == 'gpu_memset':return 'memset'
    name=event['name'].lower()
    if 'nccl' in name:return 'NCCL'
    if any(x in name for x in ('deep_ep', 'deepep', 'deep_ep::')):return 'DeepEP'
    if any(x in name for x in ('gemm', 'matmul', 'cublas')):return 'GEMM'
    return 'unknown'


def analyze(path):
    es=json.loads(path.read_text())['traceEvents']
    spans=[e for e in es if e.get('ph')=='X' and e.get('dur',0)>0]
    windows=[e for e in spans if e.get('cat')=='user_annotation' and e['name'].startswith('RoutePack/window/')]
    if not windows or len({e['name'] for e in windows})!=len(windows):
        raise ValueError('Expected one unique forward/backward/optimizer window per rank')
    def window_at(ts):
        found=[e['name'].rsplit('/',1)[1] for e in windows if e['ts']<=ts<=e['ts']+e['dur']]
        if len(found)>1:raise ValueError('Overlapping capture windows')
        return found[0] if found else None
    ranges=[e for e in spans if e.get('cat')=='user_annotation' and e['name'].startswith('RoutePack/') and not e['name'].startswith('RoutePack/window/')]
    cpu_ops=[e for e in spans if e.get('cat')=='cpu_op']
    apis=[e for e in spans if e.get('cat') in ('cuda_runtime','cuda_driver')]
    # Sweep CPU range nesting on each launch thread; CUDA activity follows its
    # correlation ID, not its timestamp overlap with a CPU annotation.
    markers=defaultdict(list)
    for index,e in enumerate(ranges):
        thread=(e.get('pid'),e.get('tid'))
        markers[thread].append((e['ts'],0,index))
        markers[thread].append((e['ts']+e['dur'],2,index))
    for index,e in enumerate(apis):
        markers[(e.get('pid'),e.get('tid'))].append((e['ts'],1,index))
    api_ranges={}
    for points in markers.values():
        active=set()
        for _,kind,index in sorted(points):
            if kind==0:active.add(index)
            elif kind==2:active.discard(index)
            else:
                e=apis[index]
                api_ranges[index]=sorted((i for i in active if ranges[i]['ts']<=e['ts'] and e['ts']+e['dur']<=ranges[i]['ts']+ranges[i]['dur']+.1), key=lambda i:(ranges[i]['ts'],-ranges[i]['dur']))
    correlation={}
    for i,e in enumerate(apis):
        corr=e.get('args',{}).get('correlation')
        if corr is not None:
            if corr in correlation:raise ValueError('Duplicate CUDA correlation ID')
            correlation[corr]=i
    external=defaultdict(list)
    for e in cpu_ops:
        ext=e.get('args',{}).get('External id')
        if ext is not None:external[ext].append(e)
    by_window=defaultdict(list); unattributed=defaultdict(list); components=defaultdict(list)
    links=defaultdict(int); outside=[]
    for e in spans:
        if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):continue
        args=e.get('args',{});index=correlation.get(args.get('correlation'))
        tags=[];window=None
        if index is not None:
            launch=apis[index];window=window_at(launch['ts']);tags=api_ranges.get(index,[])
            links['gpu_events_linked_to_cuda_api']+=1
            if args.get('External id') in external:links['gpu_events_with_matching_cpu_external_id']+=1
        else:links['gpu_events_missing_cuda_api_correlation']+=1
        if window is None:
            # Time membership identifies only the synchronized collection window,
            # never a component. Preserve the missing-correlation counter.
            window=window_at(e['ts'])
        if window is None:
            outside.append(e);continue
        by_window[window].append(e)
        if not tags:unattributed[window].append(e)
        for tag in {component(ranges[i]['name']) for i in tags}:
            components[(window,tag)].append(e)
    output=[]
    for span in sorted(windows,key=lambda e:e['ts']):
        name=span['name'].rsplit('/',1)[1];events=by_window[name];total=metrics(events);unknown=metrics(unattributed[name])
        categories={category:metrics([e for e in events if e['cat']==category]) for category in ('kernel','gpu_memcpy','gpu_memset')}
        cpu_ranges=defaultdict(list)
        for e in ranges:
            if window_at(e['ts'])==name:cpu_ranges[component(e['name'])].append(e)
        rows=[]
        for tag in sorted(set(cpu_ranges)|{tag for w,tag in components if w==name}):
            rows.append(dict(component=tag,cpu_range_count=len(cpu_ranges[tag]),
                cpu_inclusive_sum_ms=sum(e['dur'] for e in cpu_ranges[tag])/1000,
                gpu_inclusive=metrics(components[(name,tag)])))
        buckets={bucket:metrics([e for e in events if kernel_bucket(e)==bucket]) for bucket in sorted({kernel_bucket(e) for e in events})}
        output.append(dict(kernel_name_buckets=buckets,window=name,cpu_window_wall_ms=span['dur']/1000,gpu=total,categories=categories,
            unattributed_to_component=unknown,
            unattributed_gpu_duration_fraction=unknown['duration_sum_ms']/total['duration_sum_ms'] if total['duration_sum_ms'] else None,
            components=rows))
    gpu_count=sum(e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') for e in spans)
    return dict(gpu_activity_available=gpu_count>0, gpu_activity_event_count=gpu_count,
        validation_error=None if gpu_count else 'CPU-only trace: GPU duration and overlap are unavailable, not zero-cost evidence',
        path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),windows=output,
        linkage=dict(links),gpu_events_outside_windows=metrics(outside),
        limitations=[
            'CPU annotation durations are host inclusive time, never device kernel duration.',
            'GPU attribution uses CUDA API correlation plus containing same-thread component ranges. External IDs are checked as supplementary linkage evidence.',
            'Missing range attribution (especially native backward kernels) remains unattributed rather than guessed.',
            'Components are nested inclusive and not additive. GPU duration sum can exceed interval union with multi-stream concurrency.',
            'GPU union is observed activity coverage, not full component latency or isolated compute, and is not additive across components.',
            'Synchronized selected-window boundaries perturb scheduling and exclude other microbatches and loss forward; no throughput or cross-rank arrival-time claim.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('traces',type=Path,nargs='+')
    a=p.parse_args();results=[]
    for path in a.traces:
        result=analyze(path);results.append(result)
        print(path.name,[(r['window'],round(r['gpu']['duration_sum_ms'],2),round(r['gpu']['interval_union_ms'],2),r['unattributed_gpu_duration_fraction']) for r in result['windows']])
    a.output.write_text(json.dumps({'results':results},indent=2)+'\n')


if __name__=='__main__':main()
