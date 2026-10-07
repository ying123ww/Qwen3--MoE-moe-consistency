"""Audit and summarize the short real-online pair without conflating stage timers."""
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    summaries={}
    workloads={}
    for variant in ('aligned','native'):
        run=ROOT/'gpu_runs'/f'a-online-{variant}-20260910-01'
        assert json.loads((run/'job-status.json').read_text())['status']=='SUCCEEDED'
        rows=[json.loads(line) for line in (run/'iterations.jsonl').read_text().splitlines()]
        work=[json.loads(line) for line in (run/'workload.jsonl').read_text().splitlines()]
        assert [x['rollout_id'] for x in rows]==list(range(1,7))
        assert [x['rollout_id'] for x in work]==list(range(1,7))
        assert [x['warmup'] for x in rows]==[True,False,False,False,False,False]
        for row,w in zip(rows,work):
            assert w['samples']==8 and 0<w['response_tokens']<=1024
            assert math.isfinite(row['total_s']) and row['total_s']>0
            assert all(math.isfinite(t) and t>=0 for t in row['phases_s'].values())
            assert abs(sum(row['phases_s'].values())-row['total_s'])<1e-6
        argv=json.loads((run/'argv.json').read_text())
        assert argv[argv.index('--lr')+1]=='1e-6'
        assert '--use-stateless-adam' in argv and '--reset-optimizer-states' in argv
        assert '--load-debug-rollout-data' not in argv and '--save-debug-train-data' not in argv
        assert ('--deterministic-mode' in argv)==(variant=='aligned')
        assert ('--sglang-enable-deterministic-inference' in argv)==(variant=='aligned')
        log=(run/'job.log').read_text()
        grads=re.findall(r"'train/grad_norm': ([0-9.eE+\-]+)",log)
        assert len(grads)==6 and all(math.isfinite(float(x)) and float(x)>=0 for x in grads) and any(float(x)>0 for x in grads),grads
        formal=rows[1:]
        summary=dict(variant=variant,independent_runs=1,warmup_iterations=1,formal_iterations=5,
                     median_total_s=statistics.median(x['total_s'] for x in formal),
                     min_total_s=min(x['total_s'] for x in formal),max_total_s=max(x['total_s'] for x in formal),
                     mean_phases_s={k:statistics.mean(x['phases_s'][k] for x in formal) for k in formal[0]['phases_s']},
                     formal_response_tokens=sum(w['response_tokens'] for w in work[1:]),
                     formal_training_input_tokens=sum(w['training_input_tokens'] for w in work[1:]),
                     response_tokens_per_e2e_s=sum(w['response_tokens'] for w in work[1:])/sum(x['total_s'] for x in formal),
                     grad_norms=list(map(float,grads)),nonzero_gradient_iterations=sum(float(x)>0 for x in grads),iterations=rows,workload=work)
        (run/'online-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        summaries[variant]=summary
        workloads[variant]=work
    a,n=summaries['aligned'],summaries['native']
    result=dict(runs=summaries,median_total_overhead_pct=(a['median_total_s']/n['median_total_s']-1)*100,
                prompt_hashes_match=all(x['prompt_sha256']==y['prompt_sha256'] for x,y in zip(workloads['aligned'],workloads['native'])),
                response_lengths_match=all(x['response_lengths']==y['response_lengths'] for x,y in zip(workloads['aligned'],workloads['native'])),
                caveat='One independent online run per variant, five measured iterations. Real updates may diverge; not a fixed-token replay, long-horizon Adam, convergence claim, or multi-run confidence estimate.')
    (ROOT/'gpu_runs/online-comparison-20260910.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
    for v,s in summaries.items():
        print(v,json.dumps({k:val for k,val in s.items() if k not in ('iterations','workload')},indent=2))


if __name__=='__main__':
    main()
