"""Validate and compare three independent runs, retaining run-level variation."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

BASE=Path('/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation')
RUNS=BASE/'gpu_runs'


def main():
    records=[]
    inputs=set()
    scopes=set()
    for pair in (2,3,4):
        for variant in ('native','aligned'):
            root=RUNS/('a-%s-timing-20260910-%02d'%(variant,pair))
            summary=json.loads((root/'measurement-summary.json').read_text())
            assert json.loads((root/'job-status.json').read_text())['status']=='SUCCEEDED'
            assert summary['measured_steps']==20 and summary['additional_warmup_steps']==5
            inputs.add(summary['data_sha256']);scopes.add(summary['scope'])
            for rank in range(8):
                audit=json.loads((root/('audit-rank%d.json'%rank)).read_text())
                assert audit['variant']==variant and audit['lr']==0 and audit['ep_size']==4
                if variant=='native':assert not audit['wrappers']
                else:assert audit['wrappers'].get('_slime_qwen3_triton_moe_wrapped')==48
                rows=[json.loads(x) for x in (root/('measurements-rank%d.jsonl'%rank)).read_text().splitlines()]
                assert len(rows)==25 and [x['rollout_id'] for x in rows]==list(range(2,27))
            records.append({'pair':pair,**summary})
    assert len(inputs)==1 and len(scopes)==1
    ratios=[]
    for pair in (2,3,4):
        group={r['variant']:r for r in records if r['pair']==pair}
        ratios.append(100*(group['aligned']['step_time_median_s']/group['native']['step_time_median_s']-1))
    groups={v:[r['step_time_median_s'] for r in records if r['variant']==v] for v in ('native','aligned')}
    result={'runs':records,'paired_overhead_percent':ratios,
            'mean_paired_overhead_percent':statistics.mean(ratios),
            'run_medians_s':groups,
            'median_of_run_medians_s':{v:statistics.median(x) for v,x in groups.items()},
            'inference_limit':'n=3 independent starts per variant; descriptive paired comparison, not 60 independent replicates'}
    (RUNS/'repeated-comparison-20260910.json').write_text(json.dumps(result,indent=2)+'\n')
    with (RUNS/'repeated-comparison-20260910.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['pair','variant','median_s','p95_s','input_tokens_per_s','peak_allocated_gib'])
        for r in records:w.writerow([r['pair'],r['variant'],r['step_time_median_s'],r['step_time_p95_s'],r['input_tokens_per_s'],r['peak_allocated_gib_max_rank']])
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))

if __name__=='__main__':main()
