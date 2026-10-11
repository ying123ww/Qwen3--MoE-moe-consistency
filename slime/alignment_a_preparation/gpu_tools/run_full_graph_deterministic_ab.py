"""P10: gate A/B/C first, then measure nine independently started full-graph Engines."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time

import full_graph_ab_experiment as experiment
import residual_rms_performance as guard
from full_graph_ab_workload import compare as compare_workloads
from resource_preflight import snapshot,validate
from run_fixed_capacity_pipeline import full_proof

ORDERS=(('A','B','C'),('B','C','A'),('C','A','B'))


def sources():
    root=guard.PROJECT/'experiments/sglang-main-sm80/python/sglang/srt'
    return [Path(__file__),*guard.PREP.glob('gpu_tools/full_graph_ab_*.py'),
        guard.PREP/'gpu_tools/fixed_capacity_dispatch_hooks.py',guard.PREP/'gpu_tools/fixed_capacity_experiment.py',
        guard.PREP/'gpu_tools/breakable_model_gate.py',guard.PREP/'gpu_tools/router_tile_exact.py',
        guard.PREP/'gpu_tools/inference_benchmark.py',root/'server_args.py',root/'models/qwen3_moe.py',
        root/'models/utils.py',root/'layers/attention/triton_backend.py',root/'layers/layernorm.py',
        root/'batch_invariant_ops/batch_invariant_ops.py',root/'batch_invariant_ops/fused_rmsnorm_candidate.py',
        root/'layers/moe/token_dispatcher/deepep.py',root/'layers/moe/moe_runner/triton.py',
        root/'layers/moe/moe_runner/triton_utils/fused_moe.py',root/'managers/scheduler.py',
        root/'model_executor/runner/eager_runner.py',root/'model_executor/runner_backend/full_cuda_graph_backend.py']


def verify(output,mode,backend='full',phase='gate'):
    cfg=json.loads((output/'resolved-server-args.json').read_text())
    assert cfg['enable_deterministic_inference']==(mode=='C')
    assert cfg['cuda_graph_config']['decode']['backend']==('disabled' if backend=='eager' else 'full')
    assert cfg['cuda_graph_config']['decode']['bs']==[2,4]
    assert cfg['cuda_graph_config']['prefill']['backend']=='disabled'
    assert cfg['deepep_mode']=='normal' and cfg['moe_runner_backend']=='triton'
    if mode=='A':assert cfg['deepep_config'] is None
    else:assert json.loads(cfg['deepep_config'])==json.loads(experiment.fixed_config())
    assert cfg['prefill_max_requests']==(1 if phase=='gate' else None)
    if backend=='full':full_proof(output)
    records=[json.loads(p.read_text()) for p in (output/'graph-evidence').glob(f'runtime-{phase}-rank-*.json')]
    assert {p['rank'] for p in records}==set(range(8))
    for record in records:
        assert set(record['communication'])=={'dispatch','combine'}
        if mode!='A':
            for name,actual in record['communication'].items():
                assert actual==json.loads(experiment.fixed_config())['normal_'+name]
        if backend=='full':
            assert record['counts'].get('full_replays',0)>0
            assert record['counts'].get('eager_decode',0)==0
            assert record['counts'].get('eager_decode_idle',0)==0
        else:assert record['counts'].get('eager_decode',0)>0
    return records


def command(run,label,script,options,env,memory):
    guard.command(run,label,[str(guard.PYTHON),str(guard.PREP/'gpu_tools'/script),*options],env,memory,timeout=2400)


def gates(run,env,memory):
    for mode,backend in (('A','full'),('B','eager'),('B','full'),('C','eager'),('C','full')):
        label=f'exact-{mode}-{backend}';output=run/label
        options=['--mode',mode,'--backend',backend,'--output',str(output)]
        if backend=='eager':options.append('--shadow')
        command(run,label,'full_graph_ab_model_gate.py',options,env,memory)
        verify(output,mode,backend)
        if backend=='eager':
            records=[json.loads(p.read_text()) for p in (output/'graph-evidence').glob('fixed-shadow-rank-*.json')]
            assert {p['rank'] for p in records}==set(range(8))
            assert all(p['counts'].get('shadow_calls',0)>0 for p in records)
        if backend=='full' and mode!='A':
            reference=run/f'exact-{mode}-eager'
            script='breakable_model_gate.py' if mode=='C' else 'baseline_graph_model_gate.py'
            command(run,'compare-'+mode,script,['compare','--reference',str(reference),'--candidate',str(output),
                '--output',str(run/f'compare-{mode}.json')],env,memory)
    from router_tile_exact import load_case,compare_samples
    differences=[]
    for name in ('batch8','batch3','batch1','variable3'):
        differences.extend(compare_samples(load_case(run/'exact-A-full',name),load_case(run/'exact-B-full',name),name))
    guard.save(run/'A-B-greedy-differences.json',dict(failures=differences,
        scope='Cross-configuration comparison is descriptive; B full/eager equivalence is the strict gate'))
    guard.save(run/'gate-complete.json',dict(passed=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources()},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


def timing(run,gate_run,env,memory):
    proof=json.loads((gate_run/'gate-complete.json').read_text())
    assert proof['passed']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in proof['source_sha256'].items())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    summaries={};observations={}
    for round_index,order in enumerate(ORDERS,1):
        for mode in order:
            label=f'round-{round_index}-{mode}';output=run/label;output.mkdir()
            command(run,label,'full_graph_ab_benchmark.py',['--mode',mode,'--run',str(output)],env,memory)
            verify(output,mode,phase='formal')
            manifest=json.loads((output/'manifest.json').read_text());assert manifest['source_git_commit']==commit
            rows=[json.loads(x) for x in (output/'measurements.jsonl').read_text().splitlines()]
            assert len(rows)==23 and sum(not x['warmup'] for x in rows)==20
            assert all(x['cached_tokens']==[0]*8 and x['output_tokens']==1024 for x in rows)
            summary=json.loads((output/'summary.json').read_text())
            assert summary['median_s']==statistics.median(x['elapsed_s'] for x in rows if not x['warmup'])
            summaries[label]=summary;guard.save(run/'performance-progress.json',summaries)
        for left,right in (('A','B'),('B','C')):
            leftdir=run/f'round-{round_index}-{left}';rightdir=run/f'round-{round_index}-{right}'
            manifest=json.loads((leftdir/'manifest.json').read_text())
            comparison=[compare_workloads(leftdir/f'workload-{batch:02d}.npz',rightdir/f'workload-{batch:02d}.npz',
                manifest['prompt_lengths']) for batch in range(3,23)]
            observations[f'round-{round_index}-{left}-{right}']={k:sum(x[k] for x in comparison) for k in comparison[0]}
            guard.save(run/'workload-differences.json',observations)
    paired=[]
    for index in range(1,4):
        a,b,c=(summaries[f'round-{index}-{mode}']['median_s'] for mode in experiment.MODES)
        paired.append(dict(round=index,A_s=a,B_s=b,C_s=c,communication_config_overhead_percent=100*(b/a-1),
            deterministic_scheme_overhead_percent=100*(c/b-1),total_C_over_A_percent=100*(c/a-1)))
    result=dict(summaries=summaries,paired=paired,workload_differences=observations,independent_engines=9,
        formal_batches=180,source_git_commit=commit,
        median_paired_percent={key:statistics.median(x[key] for x in paired) for key in (
            'communication_config_overhead_percent','deterministic_scheme_overhead_percent','total_C_over_A_percent')},
        scope='All modes include actual route-return overhead; baseline batch-sensitive; not pure kernel or online RL costs')
    guard.save(run/'results.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('gate','timing'),required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--gate-run',type=Path)
    args=parser.parse_args()
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Run on ai_hgx_03')
    if Path(args.run_id).name!=args.run_id:raise ValueError('Plain run ID required')
    if args.phase=='timing' and args.gate_run is None:raise ValueError('Timing requires a completed gate run')
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if guard.gpu_pids():raise RuntimeError('Another GPU task is active')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup is missing')
        env=guard.environment(run);env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env['FLASHINFER_WORKSPACE_BASE']=str(Path('/dev/shm')/run.name)
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources()}
        guard.save(run/'manifest.json',dict(phase=args.phase,host=socket.gethostname(),pid=os.getpid(),
            source_sha256=hashes,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            orders=ORDERS,route_observation=True,restore_keepalive=False,gate_run=str(args.gate_run)))
        for name in hashes:
            p=Path(name);dest=run/'source-snapshots'/p.relative_to(guard.PROJECT)
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        guard.save(run/'resources.json',resources);guard.p2p_preflight(run)
        experiment.fixed_config(env['SLIME_QWEN3_FIXED_DEEPEP_CONFIG'])
        os.environ['SLIME_QWEN3_FIXED_DEEPEP_CONFIG']=env['SLIME_QWEN3_FIXED_DEEPEP_CONFIG']
        if args.phase=='gate':gates(run,env,groups[0]/'memory.usage_in_bytes')
        else:timing(run,args.gate_run,env,groups[0]/'memory.usage_in_bytes')
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        guard.save(run/'state.json',dict(status='complete',phase=args.phase,pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:guard.stop_worker();print('P10_REMAINING_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':main()
