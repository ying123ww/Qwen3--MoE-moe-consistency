"""Finite expert graph shadow/model gates, then matched generate comparison."""
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


def expert_proof(output,shadow=False):
    records=[json.loads(p.read_text()) for p in (output/'graph-evidence').glob('expert-cache-rank-*.json')]
    assert {p['rank'] for p in records}==set(range(8))
    assert all(p['ready'] and p['cache_graphs']==768 for p in records)
    replays=[json.loads(p.read_text()) for p in (output/'graph-evidence').glob('expert-first-replay-rank-*.json')]
    assert {p['rank'] for p in replays}==set(range(8))
    assert all(p['replays']>0 and p['unsupported_fallbacks']==0 for p in replays)
    if shadow:
        rows=[json.loads(p.read_text()) for p in (output/'graph-evidence').glob('expert-shadow-rank-*.json')]
        assert {p['rank'] for p in rows}==set(range(8))
        assert all(p['shadow_calls']>0 and p['synthetic_calls']==96 and p['unsupported_fallbacks']==0
                   and set(map(int,p['layers']))==set(range(48)) for p in rows)
        return rows
    return replays


def main():
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs/expert-graph-hgx03-20261011-01';run.mkdir(exist_ok=False)
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if guard.gpu_pids():raise RuntimeError('Other GPU task exists')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        guard.save(run/'resources.json',resources);env=guard.environment(run)
        env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env.update(SGLANG_MEMORY_SAVER_CUDA_GRAPH='0',SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH='1')
        srt=guard.PROJECT/'experiments/sglang-main-sm80/python/sglang/srt'
        paths=[Path(__file__),guard.PREP/'gpu_tools/expert_graph_hooks.py',*guard.PREP.glob('gpu_tools/breakable_*.py'),
               guard.PREP/'gpu_tools/inference_benchmark.py',guard.PREP/'gpu_tools/graph_fusion_proof.py',
               srt/'layers/moe/ep_moe/layer.py',srt/'layers/moe/moe_runner/triton.py',
               srt/'layers/moe/moe_runner/triton_utils/fused_moe.py',srt/'layers/moe/token_dispatcher/deepep.py',
               srt/'batch_invariant_ops/fused_rmsnorm_candidate.py',srt/'model_executor/model_runner_components/cuda_graph_setup.py']
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for name in hashes:
            p=Path(name);dest=run/'source-snapshots'/p.relative_to(guard.PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        guard.save(run/'manifest.json',dict(host=socket.gethostname(),pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
            source_sha256=hashes,restore_keepalive=False,expert_cache_rows=list(range(1,17)),
            scope='RMS Breakable baseline vs same plus native expert compute graphs; DeepEP dispatch/combine eager',
            protocol=dict(warmup=3,steps=20,response_tokens=128,rounds=1)))
        guard.p2p_preflight(run)
        gate=run/'exact-experts';shadow=dict(env);shadow['SLIME_EXPERT_GRAPH_SHADOW']='1'
        guard.command(run,'exact-experts',[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_model_gate.py'),
            'capture','--mode','breakable-rms-experts','--output',str(gate)],shadow,groups[0]/'memory.usage_in_bytes',timeout=1800)
        graph_proof(gate);records=expert_proof(gate,shadow=True)
        reference=guard.PREP/'gpu_runs/breakable-fusions-hgx03-20261011-03/exact-breakable-rms'
        guard.command(run,'compare-experts',[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_model_gate.py'),
            'compare','--reference',str(reference),'--candidate',str(gate),'--output',str(run/'exact-comparison.json')],
            env,groups[0]/'memory.usage_in_bytes')
        guard.save(run/'shadow-summary.json',dict(passed=True,synthetic_calls=sum(p['synthetic_calls'] for p in records),
            synthetic_elements=sum(p['synthetic_elements'] for p in records),real_calls=sum(p['shadow_calls'] for p in records),
            real_elements=sum(p['shadow_elements'] for p in records)))
        summaries={}
        for mode in ('breakable-rms','breakable-rms-experts'):
            output=run/mode;output.mkdir()
            worker=dict(env);worker.update(SLIME_EXPERT_GRAPH_SHADOW='0',SLIME_BREAKABLE_EXPERIMENT_MODE=mode,
                SLIME_GRAPH_FUSION_PREWARM='1')
            guard.command(run,mode,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_inference_benchmark.py'),
                '--run',str(output),'--variant','aligned','--residual-rms-mode','candidate','--router-tile-mode','off',
                '--warmup','3','--steps','20','--response-tokens','128'],worker,groups[0]/'memory.usage_in_bytes',timeout=1800)
            graph_proof(output)
            if mode.endswith('experts'):expert_proof(output)
            summaries[mode]=json.loads((output/'summary.json').read_text())
            guard.save(run/'performance-progress.json',summaries)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        result=dict(exact_passed=True,summaries=summaries,
            delay_reduction_percent=100*(1-summaries['breakable-rms-experts']['median_s']/summaries['breakable-rms']['median_s']),
            limitation='One sequential Engine launch per mode; finite greedy/shadow gate, no cross-framework training gate')
        guard.save(run/'results.json',result);guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:
        guard.stop_worker();print('EXPERT_GRAPH_REMAINING_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':main()
