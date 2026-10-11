"""Gate and time transferable optimizations on non-deterministic Breakable."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

import baseline_graph_experiment as experiment
import residual_rms_performance as guard
from resource_preflight import snapshot,validate
from run_breakable_model_and_performance import graph_proof


def proof(output,mode,shadow=False):
    graph_proof(output)
    evidence=output/'graph-evidence'
    if mode in ('experts','fixed-experts'):
        records=[json.loads(p.read_text()) for p in evidence.glob('expert-cache-rank-*.json')]
        assert {p['rank'] for p in records}==set(range(8))
        expected=48 if mode=='fixed-experts' else 768
        assert all(p['ready'] and p['cache_graphs']==expected for p in records)
        replays=[json.loads(p.read_text()) for p in evidence.glob('expert-first-replay-rank-*.json')]
        assert {p['rank'] for p in replays}==set(range(8))
        assert all(p['replays']>0 and p['unsupported_fallbacks']==0 for p in replays)
        if shadow:
            records=[json.loads(p.read_text()) for p in evidence.glob('expert-shadow-rank-*.json')]
            assert {p['rank'] for p in records}==set(range(8))
            assert all(p['shadow_calls']>0 and p['synthetic_calls']==(6 if mode=='fixed-experts' else 96)
                       and p['unsupported_fallbacks']==0 for p in records)
    if mode in ('fixed','fixed-experts'):
        records=[json.loads(p.read_text()) for p in evidence.glob('fixed-replay-rank-*.json')]
        assert {p['rank'] for p in records}==set(range(8))
        assert all(p['counts'].get('fixed_eager_dispatch',0)>0 and p['counts'].get('breakable_replays',0)>0 for p in records)
        if shadow:assert all(p['counts'].get('shadow_calls',0)>0 for p in records)


def verify_baseline(output):
    args=json.loads((output/'resolved-server-args.json').read_text())
    assert not args['enable_deterministic_inference']
    assert args['cuda_graph_config']['decode']['backend']=='breakable'
    assert args['cuda_graph_config']['prefill']['backend']=='disabled'
    assert args['deepep_mode']=='normal' and args['moe_runner_backend']=='triton'


def main():
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs/baseline-breakable-opt-hgx03-20261012-02';run.mkdir(exist_ok=False)
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if guard.gpu_pids():raise RuntimeError('Other GPU task exists')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        env=guard.environment(run);env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env.update(SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE='0',SGLANG_MEMORY_SAVER_CUDA_GRAPH='0',
            SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH='0',SLIME_QWEN3_RESIDUAL_RMS_MODE='off',
            SLIME_QWEN3_RESIDUAL_RMS_AUDIT='0')
        srt=guard.PROJECT/'experiments/sglang-main-sm80/python/sglang/srt'
        paths=[Path(__file__),*guard.PREP.glob('gpu_tools/baseline_graph_*.py'),
            guard.PREP/'gpu_tools/expert_graph_hooks.py',guard.PREP/'gpu_tools/fixed_capacity_dispatch_hooks.py',
            guard.PREP/'gpu_tools/breakable_experiment.py',guard.PREP/'gpu_tools/breakable_model_gate.py',
            guard.PREP/'gpu_tools/inference_benchmark.py',srt/'layers/layernorm.py',srt/'models/qwen3_moe.py',
            srt/'layers/moe/moe_runner/triton.py',srt/'layers/moe/token_dispatcher/deepep.py',
            srt/'layers/moe/moe_runner/triton_utils/fused_moe.py',srt/'layers/attention/triton_backend.py']
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for name in hashes:
            p=Path(name);dest=run/'source-snapshots'/p.relative_to(guard.PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        guard.save(run/'manifest.json',dict(host=socket.gethostname(),pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
            source_sha256=hashes,modes=list(experiment.MODES),deterministic=False,unified_decode=False,
            residual_rms_candidate=False,qk_candidate=False,restore_keepalive=False,
            correctness_gate_prefill_max_requests=1,benchmark_prefill_max_requests=None,
            protocol=dict(rounds=1,warmup=3,steps=20,response_tokens=128)))
        guard.save(run/'resources.json',resources);guard.p2p_preflight(run)
        for mode in experiment.MODES:
            output=run/('exact-'+mode)
            command=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/baseline_graph_model_gate.py'),
                'capture','--mode',mode,'--output',str(output)]
            if mode!='original':command.append('--shadow')
            guard.command(run,'exact-'+mode,command,env,groups[0]/'memory.usage_in_bytes',timeout=1800)
            verify_baseline(output);proof(output,mode,shadow=mode!='original')
            if mode!='original':
                guard.command(run,'compare-'+mode,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/baseline_graph_model_gate.py'),
                    'compare','--reference',str(run/'exact-original'),'--candidate',str(output),
                    '--output',str(run/('compare-'+mode+'.json'))],env,groups[0]/'memory.usage_in_bytes')
        results={}
        for mode in experiment.MODES:
            output=run/mode;output.mkdir();worker=dict(env);worker['SLIME_BASELINE_GRAPH_MODE']=mode
            guard.command(run,mode,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/baseline_graph_inference_benchmark.py'),
                '--run',str(output),'--variant','baseline','--residual-rms-mode','off','--router-tile-mode','off',
                '--warmup','3','--steps','20','--response-tokens','128'],worker,groups[0]/'memory.usage_in_bytes',timeout=1800)
            verify_baseline(output);proof(output,mode)
            summary=json.loads((output/'summary.json').read_text())
            assert summary['formal_batches']==20 and not summary['profiled']
            results[mode]=summary;guard.save(run/'performance-progress.json',results)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        original=results['original']['median_s']
        result=dict(model_exact_passed=True,summaries=results,
            delay_reduction_percent={mode:100*(1-s['median_s']/original) for mode,s in results.items()},
            limitation='One sequential Engine launch per mode; non-deterministic baseline, no batch-invariance guarantee')
        guard.save(run/'results.json',result);guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:guard.stop_worker();print('BASELINE_OPT_REMAINING_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':main()
