"""CPU-only comparison on existing B2 input; repeated runs never accumulate swaps."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from exchange import refine


def compare(samples, traces, schedule, topology, pad_multiplier, repeats=20, seed=0, **budgets):
    if repeats < 1: raise ValueError('repeats must be positive')
    # Warm-up is excluded. All measured runs start from the native schedule.
    refine(samples,traces,schedule,topology,pad_multiplier,seed=seed,**budgets)
    runs=[]; digest=None
    for _ in range(repeats):
        candidate, report, _=refine(samples,traces,schedule,topology,pad_multiplier,seed=seed,**budgets)
        current=hashlib.sha256(json.dumps(candidate,sort_keys=True).encode()).hexdigest()
        if digest is not None and digest != current: raise AssertionError('Schedule determinism failed')
        digest=current;runs.append(report)
    target=len(runs[0]['swaps'])
    random_budgets=dict(budgets);random_budgets['max_swaps']=target
    _,random_report,_=refine(samples,traces,schedule,topology,pad_multiplier,mode='random',seed=seed,**random_budgets)
    # Same count is an eligibility gate, never silently compare unmatched budgets.
    matched=len(random_report['swaps'])==target
    return {'scope':'CPU offline proxy comparison, not GPU speedup', 'python':platform.python_version(),
            'samples':len(samples),'bins':len(schedule['cells']),'repeats':repeats,'seed':seed,'budgets':budgets,
            'same_swap_count':matched,'routepack_swaps':target,'random_swaps':len(random_report['swaps']),
            'excess':{'native':runs[0]['before']['E'],'routepack':runs[0]['after']['E'],
                      'random':random_report['after']['E'] if matched else None},
            'schedule_sha256':digest,'determinism_passed':True,
            'cpu_wall_ms':{phase:{'p50':float(np.percentile([r['cpu_wall_ms'][phase] for r in runs],50)),
                                  'p95':float(np.percentile([r['cpu_wall_ms'][phase] for r in runs],95))}
                           for phase in runs[0]['cpu_wall_ms']},
            'routepack_runs':runs,'random_report':random_report}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--pad-multiplier',type=int,required=True);p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--max-candidates',type=int,default=500)
    p.add_argument('--max-iterations',type=int,default=20);p.add_argument('--max-swaps',type=int,default=20)
    a=p.parse_args()
    samples=[json.loads(l) for l in (a.input/'samples.jsonl').read_text().splitlines() if l.strip()]
    traces={s['sample_id']:np.load(a.input/s['route_path'],allow_pickle=False) for s in samples}
    result=compare(samples,traces,json.loads((a.input/'schedule.json').read_text()),
                   json.loads((a.input/'topology.json').read_text()),a.pad_multiplier,a.repeats,a.seed,
                   max_candidates=a.max_candidates,max_iterations=a.max_iterations,max_swaps=a.max_swaps)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('excess','same_swap_count','cpu_wall_ms')}))


if __name__=='__main__':main()
