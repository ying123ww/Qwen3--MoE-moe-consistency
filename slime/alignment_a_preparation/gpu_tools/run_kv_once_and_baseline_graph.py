"""Untimed KV correctness gates, then one requested baseline graph generate."""
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


def main():
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs/kv-once-baseline-graph-hgx03-20261011-01';run.mkdir(exist_ok=False)
    try:
        lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if guard.gpu_pids():raise RuntimeError('Other GPU task exists')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        guard.save(run/'resources.json',resources);env=guard.environment(run)
        env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env['SGLANG_MEMORY_SAVER_CUDA_GRAPH']='0'
        srt=guard.PROJECT/'experiments/sglang-main-sm80/python/sglang/srt'
        paths=[Path(__file__),guard.PREP/'gpu_tools/kv_once_gpu_check.py',guard.PREP/'gpu_tools/breakable_model_gate.py',
               guard.PREP/'gpu_tools/breakable_experiment.py',guard.PREP/'gpu_tools/breakable_inference_benchmark.py',
               guard.PREP/'gpu_tools/inference_benchmark.py',srt/'layers/attention/triton_backend.py',srt/'server_args.py',
               srt/'models/qwen3_moe.py',srt/'layers/moe/ep_moe/layer.py',srt/'model_executor/runner_backend/breakable_cuda_graph_backend.py']
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for name in hashes:
            p=Path(name);target=run/'source-snapshots'/p.relative_to(guard.PROJECT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        guard.save(run/'manifest.json',dict(host=socket.gethostname(),pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
            source_sha256=hashes,kv_optimization_timing=False,baseline_graph_protocol=dict(warmup=3,steps=1,response_tokens=128),restore_keepalive=False))
        guard.p2p_preflight(run)
        guard.command(run,'gpu-kv-check',[str(guard.PYTHON),str(guard.PREP/'gpu_tools/kv_once_gpu_check.py'),
            '--output',str(run/'gpu-kv-check.json')],env,groups[0]/'memory.usage_in_bytes')
        old=guard.PREP/'gpu_runs/breakable-model-perf-hgx03-20261011-01'
        for mode in ('eager','breakable'):
            output=run/('exact-'+mode)
            guard.command(run,'exact-'+mode,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_model_gate.py'),
                'capture','--mode',mode,'--output',str(output)],env,groups[0]/'memory.usage_in_bytes')
            guard.command(run,'compare-'+mode,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_model_gate.py'),
                'compare','--reference',str(old/('exact-'+mode)),'--candidate',str(output),
                '--output',str(run/('compare-'+mode+'.json'))],env,groups[0]/'memory.usage_in_bytes')
        guard.command(run,'compare-eager-graph',[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_model_gate.py'),
            'compare','--reference',str(run/'exact-eager'),'--candidate',str(run/'exact-breakable'),
            '--output',str(run/'compare-eager-graph.json')],env,groups[0]/'memory.usage_in_bytes')
        guard.save(run/'optimization-complete.json',dict(correctness_passed=True,timing_performed=False))
        print('KV_ONCE_OPTIMIZATION_CORRECTNESS_COMPLETE',flush=True)
        output=run/'baseline-breakable';output.mkdir()
        worker=dict(env);worker['SLIME_BREAKABLE_EXPERIMENT_MODE']='breakable'
        worker['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']='0'
        guard.command(run,'baseline-breakable',[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_inference_benchmark.py'),
            '--run',str(output),'--variant','baseline','--residual-rms-mode','off','--router-tile-mode','off',
            '--warmup','3','--steps','1','--response-tokens','128'],worker,groups[0]/'memory.usage_in_bytes')
        resolved=json.loads((output/'resolved-server-args.json').read_text())
        assert not resolved['enable_deterministic_inference'] and resolved['cuda_graph_config']['decode']['backend']=='breakable'
        assert resolved['cuda_graph_config']['prefill']['backend']=='disabled'
        proof=[json.loads(p.read_text()) for p in (output/'graph-evidence').glob('*.json')]
        if {p['rank'] for p in proof}!=set(range(8)):raise RuntimeError('Baseline graph replay not observed on all ranks')
        if any(p['segments']!=49 or p['breaks']!=48 or not p['successful_replay'] for p in proof):raise RuntimeError('Unexpected baseline graph boundaries')
        summary=json.loads((output/'summary.json').read_text())
        assert summary['formal_batches']==1 and summary['output_tokens']==1024 and not summary['profiled']
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise RuntimeError('Source changed during run')
        result=dict(kv_correctness_passed=True,kv_timing_performed=False,baseline_graph_seconds=summary['median_s'],
                    baseline_graph_formal_batches=1,baseline_deterministic_inference=False,baseline_unified_decode=False,
                    baseline_graph_replay_ranks=8,limitation='One baseline graph batch, not a statistical comparison')
        guard.save(run/'results.json',result);guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:guard.stop_worker()


if __name__=='__main__':main()
