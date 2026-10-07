"""Aggregate completed fixed-weight benchmark timings; CPU-only, no invented runs."""
import argparse
import json
import math
from pathlib import Path
import statistics


def require(ok, message):
    if not ok:
        raise ValueError(message)


def summarize_run(data, expected_ranks=8):
    require(data.get('passed') is True, 'Benchmark validity gate did not pass')
    checks=data.get('checks', [])
    require(len(checks)==expected_ranks and {c['rank'] for c in checks}==set(range(expected_ranks)), 'Missing or duplicate rank checks')
    require(all(c.get('weights_unchanged') is True and c.get('gradient_check',{}).get('finite') is True
                and c.get('gradient_check',{}).get('nonzero') is True for c in checks), 'Weight/gradient validity check failed')
    samples=[]; seen=set(); excluded={}; attribution=[]
    for gathered in data['iterations']:
        require(len(gathered)==expected_ranks and {r['rank'] for r in gathered}==set(range(expected_ranks)), 'Missing or duplicate iteration rank')
        require(len({r['iteration'] for r in gathered})==1 and len({r['phase'] for r in gathered})==1, 'Mixed rank iteration/phase')
        iteration=gathered[0]['iteration'];phase=gathered[0]['phase']
        require(iteration not in seen,'Duplicate iteration');seen.add(iteration)
        if phase!='measured':
            excluded[phase]=excluded.get(phase,0)+1
            if phase=='attribution':
                attribution.append({'iteration':iteration,'ranks_with_events':[
                    r['rank'] for r in gathered if r.get('layer_attribution',{}).get('events')]})
            continue
        values={r['rank']:r.get('full_step_s') for r in gathered}
        require(all(isinstance(t,(float,int)) and not isinstance(t,bool) and math.isfinite(t) and t>0 for t in values.values()),'Invalid full_step_s')
        require(all(r.get('optimizer_invocations')==1 for r in gathered),'Expected one optimizer invocation')
        samples.append({'iteration':iteration,'max_full_step_s':max(values.values()),
                        'mean_rank_full_step_s':statistics.mean(values.values()),
                        'slowest_ranks':[rank for rank,t in sorted(values.items()) if t==max(values.values())],
                        'rank_full_step_s':values})
    require(len(samples)==data['measured_repeats'] and samples,'Measured repeat coverage mismatch')
    return {'scope':data.get('scope'),'schedule_manifest':data.get('schedule_manifest'),
            'checkpoint_cpu_offload':data.get('checkpoint_cpu_offload'),
            'measured_repeats':len(samples),'raw_measured_iterations':samples,
            'p50_max_rank_full_step_s':statistics.median(s['max_full_step_s'] for s in samples),
            'excluded_phase_counts':excluded,'attribution_presence_only':attribution,
            'attribution_note':'No H/time join performed: occurrence slots and expert recomputation must be validated against schedule first. Instrumented iterations are excluded from formal timing.'}


def compare_runs(runs):
    require('native' in runs,'Native required for comparison')
    reference=runs['native'];baseline=reference['p50_max_rank_full_step_s']
    for name,run in runs.items():
        require(run['scope']==reference['scope'] and run['checkpoint_cpu_offload']==reference['checkpoint_cpu_offload'], 'Mixed benchmark scope/offload configurations')
        run['p50_time_change_vs_native_percent']=100*(run['p50_max_rank_full_step_s']/baseline-1)
        run['p50_speedup_ratio']=baseline/run['p50_max_rank_full_step_s']
    return {'scope':'max rank wall time per measured iteration, followed by median across iterations',
            'runs':runs,'included_methods':list(runs),
            'methods_without_supplied_timing':[name for name in ('native','random','routepack') if name not in runs],
            'missing_method_note':'No measurements or zero-valued placeholders are inferred for omitted methods.',
            'limitations':['Warmup, attribution and gradient-validation phases excluded.',
            'Cross-launch hardware, source revision and workload equivalence require separate provenance checks.',
            'A run validity pass is not a proof of gradient equality between schedules.',
            'No end-to-end RL or net online scheduler speedup is inferred.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--random',type=Path)
    p.add_argument('--routepack',type=Path)
    p.add_argument('--expected-ranks',type=int,default=8)
    p.add_argument('--offline-cpu-benchmark',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    runs={name:{'timing_file':str(path.resolve()),**summarize_run(json.loads(path.read_text()),args.expected_ranks)}
          for name in ('native','random','routepack') if (path:=getattr(args,name)) is not None}
    result=compare_runs(runs)
    if args.offline_cpu_benchmark:
        b=json.loads(args.offline_cpu_benchmark.read_text())
        result['offline_scheduler_cpu_cost_separate']={'source':str(args.offline_cpu_benchmark.resolve()),
            'cpu_wall_ms':b['cpu_wall_ms'],'repeats':b.get('repeats'),'seed':b.get('seed'),
            'note':'Offline histogram preparation/search/revalidation, not included in GPU step timing. Not automatically added or subtracted: online execution/amortization is unmeasured.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
