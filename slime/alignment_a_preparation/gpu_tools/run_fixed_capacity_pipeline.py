"""After reverse-order timing, gate fixed receive capacity and attempt full decode."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

import residual_rms_performance as guard
from resource_preflight import snapshot,validate


def full_proof(output):
    evidence=output/'graph-evidence'
    captures=[json.loads(p.read_text()) for p in evidence.glob('fixed-capture-size-*-rank-*.json')]
    replays=[json.loads(p.read_text()) for p in evidence.glob('fixed-full-replay-size-*-rank-*.json')]
    required={(r,s) for r in range(8) for s in (2,4)}
    assert {(p['rank'],p['size']) for p in captures}==required
    assert {(p['rank'],p['size']) for p in replays}==required
    assert all(p['dispatches_this_shape']==48 and p['captured_layers']==list(range(48)) for p in captures)
    assert all(p['successful_replay'] and p['backend_class']=='FullCudaGraphBackend' for p in replays)
    return captures,replays


def main():
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs/fixed-capacity-full-hgx03-20261011-01';run.mkdir(exist_ok=False)
    try:
        reverse=guard.PREP/'gpu_runs/expert-graph-reverse-hgx03-20261011-01'
        guard.save(run/'state.json',dict(status='queued',phase='waiting-for-reverse',pid=os.getpid(),time=time.time()))
        while True:
            if guard.STOP:raise RuntimeError('Stop requested')
            state=json.loads((reverse/'state.json').read_text())
            if state['status']=='complete':break
            if state['status']=='failed':raise RuntimeError('Reverse benchmark failed; see its log')
            time.sleep(2)
        lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a')
        started=time.monotonic()
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()-started>60:raise RuntimeError('Previous driver still owns resource lock')
                time.sleep(1)
        if guard.gpu_pids():raise RuntimeError('Other GPU task exists')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        env=guard.environment(run);env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env.update(SGLANG_MEMORY_SAVER_CUDA_GRAPH='0',SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH='1')
        srt=guard.PROJECT/'experiments/sglang-main-sm80/python/sglang/srt'
        paths=[Path(__file__),*guard.PREP.glob('gpu_tools/fixed_capacity_*.py'),guard.PREP/'gpu_tools/breakable_model_gate.py',
            guard.PREP/'gpu_tools/inference_benchmark.py',srt/'layers/moe/token_dispatcher/deepep.py',
            srt/'layers/moe/moe_runner/triton.py',srt/'layers/moe/moe_runner/triton_utils/fused_moe.py',
            srt/'server_args.py',srt/'layers/attention/triton_backend.py',srt/'batch_invariant_ops/fused_rmsnorm_candidate.py',
            srt/'model_executor/runner_backend/full_cuda_graph_backend.py']
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for name in hashes:
            p=Path(name);dest=run/'source-snapshots'/p.relative_to(guard.PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        guard.save(run/'manifest.json',dict(host=socket.gethostname(),pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
            source_sha256=hashes,capacity=16,restore_keepalive=False,protocol=dict(warmup=3,steps=20,response_tokens=128)))
        guard.save(run/'resources.json',resources);guard.p2p_preflight(run)
        reference=guard.PREP/'gpu_runs/breakable-fusions-hgx03-20261011-03/exact-breakable-rms'
        for backend in ('eager','full'):
            output=run/('exact-'+backend)
            command=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/fixed_capacity_model_gate.py'),
                '--backend',backend,'--output',str(output)]
            if backend=='eager':command.append('--shadow')
            guard.command(run,'exact-'+backend,command,env,groups[0]/'memory.usage_in_bytes',timeout=1800)
            if backend=='full':full_proof(output)
            else:
                stats=[json.loads(p.read_text()) for p in (output/'graph-evidence').glob('fixed-shadow-rank-*.json')]
                assert {p['rank'] for p in stats}==set(range(8))
                assert all(p['counts'].get('shadow_calls',0)>0 for p in stats)
                guard.save(run/'fixed-shadow-summary.json',dict(passed=True,stats=stats))
            guard.command(run,'compare-'+backend,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_model_gate.py'),
                'compare','--reference',str(reference),'--candidate',str(output),
                '--output',str(run/('compare-'+backend+'.json'))],env,groups[0]/'memory.usage_in_bytes')
        output=run/'full-rms';output.mkdir();worker=dict(env);worker['SLIME_FIXED_CAPACITY_BACKEND']='full'
        guard.command(run,'full-rms',[str(guard.PYTHON),str(guard.PREP/'gpu_tools/fixed_capacity_inference_benchmark.py'),
            '--run',str(output),'--variant','aligned','--residual-rms-mode','candidate','--router-tile-mode','off',
            '--warmup','3','--steps','20','--response-tokens','128'],worker,groups[0]/'memory.usage_in_bytes',timeout=1800)
        full_proof(output)
        summary=json.loads((output/'summary.json').read_text())
        resolved=json.loads((output/'resolved-server-args.json').read_text())
        assert resolved['cuda_graph_config']['decode']['backend']=='full' and resolved['enable_deterministic_inference']
        assert summary['formal_batches']==20 and not summary['profiled']
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        reverse_result=json.loads((reverse/'results.json').read_text())
        baseline=reverse_result['summaries']['breakable-rms-experts']['median_s']
        result=dict(fixed_capacity_exact_passed=True,full_capture_replay_passed=True,summary=summary,
            compared_expert_graph_seconds=baseline,delay_reduction_vs_expert_graph_percent=100*(1-summary['median_s']/baseline),
            limitation='One sequential full-graph launch, compared with preceding reverse-order expert-graph trial')
        guard.save(run/'results.json',result);guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:guard.stop_worker();print('FIXED_CAPACITY_REMAINING_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':main()
