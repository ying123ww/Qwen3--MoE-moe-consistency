"""Reverse the two previously gated expert-graph benchmark configurations."""
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
from run_breakable_model_and_performance import graph_proof
from run_expert_graph_investigation import expert_proof


def main():
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    reference=guard.PREP/'gpu_runs/expert-graph-hgx03-20261011-01'
    manifest=json.loads((reference/'manifest.json').read_text())
    assert json.loads((reference/'exact-comparison.json').read_text())['model_exact']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest['source_sha256'].items())
    run=guard.PREP/'gpu_runs/expert-graph-reverse-hgx03-20261011-01';run.mkdir(exist_ok=False)
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if guard.gpu_pids():raise RuntimeError('Other GPU task exists')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        env=guard.environment(run);env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env.update(SGLANG_MEMORY_SAVER_CUDA_GRAPH='0',SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH='1',
                   SLIME_GRAPH_FUSION_PREWARM='1',SLIME_EXPERT_GRAPH_SHADOW='0')
        guard.save(run/'manifest.json',dict(reference=str(reference),source_sha256=manifest['source_sha256'],
            correctness_reused_byte_identical=True,order=['breakable-rms-experts','breakable-rms'],
            protocol=dict(warmup=3,steps=20,response_tokens=128),restore_keepalive=False))
        guard.save(run/'resources.json',resources);guard.p2p_preflight(run)
        summaries={}
        for mode in ('breakable-rms-experts','breakable-rms'):
            output=run/mode;output.mkdir();worker=dict(env);worker['SLIME_BREAKABLE_EXPERIMENT_MODE']=mode
            guard.command(run,mode,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_inference_benchmark.py'),
                '--run',str(output),'--variant','aligned','--residual-rms-mode','candidate','--router-tile-mode','off',
                '--warmup','3','--steps','20','--response-tokens','128'],worker,groups[0]/'memory.usage_in_bytes',timeout=1800)
            graph_proof(output)
            if mode.endswith('experts'):expert_proof(output)
            summaries[mode]=json.loads((output/'summary.json').read_text())
            guard.save(run/'performance-progress.json',summaries)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest['source_sha256'].items())
        result=dict(summaries=summaries,delay_reduction_percent=100*(1-summaries['breakable-rms-experts']['median_s']/summaries['breakable-rms']['median_s']))
        guard.save(run/'results.json',result);guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:guard.stop_worker();print('REVERSE_REMAINING_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':main()
