"""Independent P12 numerical/protocol/workload/native-replay audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import numpy as np
from full_graph_ab_workload import compare
from run_fixed_capacity_pipeline import full_proof


def read(path):return json.loads(path.read_text())


def audit(run):
    result=read(run/'results.json');manifest=read(run/'manifest.json')
    if not result['complete'] or read(run/'state.json')['status']!='complete':raise RuntimeError('Timing incomplete')
    for path,digest in manifest['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen runtime source changed')
    if len(result['records'])!=11:raise RuntimeError('Expected7pilot+4confirmation independent engines')
    pids=[];commits=set();frozen=set();common=None;record_count=0;formal_count=0;differences={};records={};lp_comparisons={}
    for item in result['records']:
        folder=Path(item['output']);label=folder.name;variant=item['variant'];worker=read(folder/'manifest.json');resolved=read(folder/'resolved-server-args.json')
        if not resolved['enable_deterministic_inference'] or resolved['cuda_graph_config']['decode']['backend']!='full' or resolved['cuda_graph_config']['decode']['bs']!=[2,4] or resolved['cuda_graph_config']['prefill']['backend']!='disabled':raise RuntimeError('Graph/numerical mode drift')
        if not resolved['enable_return_routed_experts'] or worker['sampling']!=dict(temperature=.6,max_new_tokens=128,ignore_eos=True):raise RuntimeError('Workload protocol drift')
        env=worker['environment']
        if env['SLIME_QWEN3_P12_VARIANT']!=variant or env['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']!='1':raise RuntimeError('Variant provenance missing')
        if env['SLIME_QWEN3_RESIDUAL_RMS_MODE']!=('off' if 'rms-off' in variant else 'candidate'):raise RuntimeError('RMS ablation was not applied')
        commits.add(worker['source_git_commit']);frozen.add(worker['frozen_sha256'])
        config={k:resolved[k] for k in ('model_path','dtype','tp_size','dp_size','attn_cp_size','ep_size','moe_dp_size','moe_dense_tp_size','attention_backend','moe_runner_backend','deepep_mode','deepep_config','random_seed')}
        if common is None:common=config
        elif common!=config:raise RuntimeError('Shared runtime configuration drift')
        rows=[json.loads(line) for line in (folder/'measurements.jsonl').read_text().splitlines()]
        if len(rows)!=23 or [x['warmup'] for x in rows]!=[True]*3+[False]*20:raise RuntimeError('Incomplete warmup/formal protocol')
        times=[x['elapsed_s'] for x in rows[3:]];summary=read(folder/'summary.json');ordered=sorted(times);at=19*.95;p95=ordered[int(at)]+(ordered[math.ceil(at)]-ordered[int(at)])*(at-int(at))
        if statistics.median(times)!=summary['median_s'] or p95!=summary['p95_s'] or summary!=item['summary']:raise RuntimeError('Summary does not match raw timing records')
        for row in rows:
            if row['cached_tokens']!=[0]*8 or row['output_tokens']!=1024 or not math.isfinite(row['elapsed_s']) or row['elapsed_s']<=0:raise RuntimeError('Cache/workload/timing invalid')
            with np.load(folder/row['artifact'],allow_pickle=False) as packet:
                if len(row['samples'])!=8:raise RuntimeError('Sample coverage missing')
                for sample in row['samples']:
                    i=sample['sample'];tokens=packet[f'tokens_{i}'];routes=packet[f'routes_{i}'];bits=packet[f'logprob_bits_{i}']
                    if tokens[-128:].tolist()!=sample['response_ids'] or routes.shape!=(len(tokens)-1,48,8) or bits.shape!=(128,) or not np.isfinite(bits.view('<f4')).all():raise RuntimeError('Actual token/route/logprob coverage invalid')
                    if hashlib.sha256(tokens.tobytes()).hexdigest()!=sample['token_sha256'] or hashlib.sha256(routes.tobytes()).hexdigest()!=sample['route_sha256']:raise RuntimeError('Observed workload hash mismatch')
            record_count+=1
        formal_count+=20
        full_proof(folder)
        if variant.startswith('dense'):
            dense=[read(x) for x in (folder/'graph-evidence').glob('dense-capture-size-*-rank-*.json')]
            if {(x['rank'],x['size']) for x in dense}!={(r,s) for r in range(8) for s in (2,4)} or any(any(x['layers'][kind]!=list(range(48)) for kind in ('qkv','output_projection')) for x in dense):raise RuntimeError('Dense48layer proof missing')
        capture=[read(x) for x in (folder/'graph-evidence').glob('p12-capture-size-*-rank-*.json')]
        if {(x['rank'],x['size']) for x in capture}!={(r,s) for r in range(8) for s in (2,4)}:raise RuntimeError('Capture coverage missing')
        for row in capture:
            if bool(row['rms_captured_calls'])!=('rms-off' not in variant) or row['qk_captured_calls']!=(48 if '-qk' in variant else 0):raise RuntimeError('Fusion capture mode drift')
            if variant.endswith('router') and (set(map(int,row['router_captured_calls']))!=set(range(48)) or any(v!=1 for v in row['router_captured_calls'].values())):raise RuntimeError('Router48layer coverage missing')
        runtime=[read(x) for x in (folder/'graph-evidence').glob('runtime-formal-rank-*.json')]
        if {x['rank'] for x in runtime}!=set(range(8)) or any(x['counts'].get('full_replays',0)!=2560 or x['counts'].get('eager_decode',0) or x['counts'].get('eager_decode_idle',0) for x in runtime):raise RuntimeError('Native replay/fallback scope failed')
        if read(run/f'{label}-exit.json')['returncode']!=0:raise RuntimeError('Worker exit failed')
        pids.append(read(run/f'{label}-launch.json')['pid']);records[(item['round'],variant)]=folder
    if len(set(pids))!=11 or commits!={manifest['git_commit']} or len(frozen)!=1:raise RuntimeError('Independent starts/samecommit/checkpoint not proved')
    paired=[]
    for r in (1,2,3):
        a=read(records[(r,'reference')]/'summary.json')['median_s'];b=read(records[(r,result['winner'])]/'summary.json')['median_s'];paired.append(dict(round=r,reference_s=a,winner_s=b,change_percent=(b/a-1)*100))
    if paired!=result['paired'] or statistics.median(x['change_percent'] for x in paired)!=result['median_paired_change_percent']:raise RuntimeError('Paired result mismatch')
    for (r,v),folder in records.items():
        if v=='reference':continue
        ref=records[(r,'reference')];lengths=read(ref/'manifest.json')['prompt_lengths']
        rows=[compare(ref/f'workload-{i:02d}.npz',folder/f'workload-{i:02d}.npz',lengths) for i in range(3,23)]
        differences[f'round{r}-{v}']={k:sum(x[k] for x in rows) for k in rows[0]}
        mismatches=0;elements=0
        for batch in range(3,23):
            with np.load(ref/f'workload-{batch:02d}.npz',allow_pickle=False) as a,np.load(folder/f'workload-{batch:02d}.npz',allow_pickle=False) as b:
                for sample in range(8):
                    x,y=a[f'logprob_bits_{sample}'],b[f'logprob_bits_{sample}']
                    mismatches+=int(np.count_nonzero(x!=y));elements+=x.size
        lp_comparisons[f'round{r}-{v}']=dict(target_tokens=elements,bit_mismatches=mismatches)
    (run/'timed-logprob-bit-comparison.json').write_text(json.dumps(lp_comparisons,indent=2)+'\n')
    if set(lp_comparisons)!=set(differences) or any(x['bit_mismatches']!=0 or x['target_tokens']!=20480 for x in lp_comparisons.values()):raise RuntimeError('Timed target-logprob bit gate incomplete')
    return dict(timed_target_logprob_bit_comparisons=lp_comparisons,passed=True,independent_engines=11,unique_pids=pids,source_git_commit=next(iter(commits)),standard_workloads=record_count,formal_batches=formal_count,paired=paired,median_paired_change_percent=result['median_paired_change_percent'],workload_differences=differences,shared_config=common,scope='Actual workloads and three paired winner launches verified; other axes each onepilot. Not RL or universal input proof.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();report=audit(args.run)
    (args.run/'measurement-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('workload_differences','shared_config')},indent=2))
