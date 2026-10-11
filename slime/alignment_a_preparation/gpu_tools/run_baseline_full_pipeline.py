"""Gate baseline full decode and compare with the best baseline Breakable variant."""
import argparse
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
from run_baseline_graph_optimizations import proof as breakable_proof
from run_fixed_capacity_pipeline import full_proof


def verify(output,backend,gate=False):
    config=json.loads((output/'resolved-server-args.json').read_text())
    assert config['enable_deterministic_inference'] is False
    assert config['cuda_graph_config']['decode']['backend']==backend
    assert config['cuda_graph_config']['prefill']['backend']=='disabled'
    assert config['deepep_mode']=='normal' and config['moe_runner_backend']=='triton'
    assert config['prefill_max_requests']==(1 if gate else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='baseline-full-hgx03-20261012-01')
    args=parser.parse_args()
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    if Path(args.run_id).name!=args.run_id:raise ValueError('Use a plain run ID')
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if guard.gpu_pids():raise RuntimeError('Other GPU task exists')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        memory=groups[0]/'memory.usage_in_bytes'
        env=guard.environment(run);env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env.update(SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE='0',SGLANG_MEMORY_SAVER_CUDA_GRAPH='0',
            SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH='0',SLIME_QWEN3_RESIDUAL_RMS_MODE='off',
            SLIME_QWEN3_RESIDUAL_RMS_AUDIT='0',SLIME_QWEN3_ROUTER_TILE_AUDIT='0')
        srt=guard.PROJECT/'experiments/sglang-main-sm80/python/sglang/srt'
        paths=[Path(__file__),*guard.PREP.glob('gpu_tools/baseline_full_*.py'),
            *guard.PREP.glob('gpu_tools/baseline_graph_*.py'),
            *guard.PREP.glob('gpu_tools/fixed_capacity_*.py'),
            guard.PREP/'gpu_tools/expert_graph_hooks.py',guard.PREP/'gpu_tools/inference_benchmark.py',
            guard.PREP/'gpu_tools/breakable_model_gate.py',guard.PREP/'gpu_tools/breakable_experiment.py',
            srt/'layers/layernorm.py',srt/'models/qwen3_moe.py',srt/'models/utils.py',srt/'server_args.py',
            srt/'layers/moe/moe_runner/triton.py',srt/'layers/moe/token_dispatcher/deepep.py',
            srt/'layers/moe/moe_runner/triton_utils/fused_moe.py',srt/'layers/attention/triton_backend.py',
            srt/'model_executor/runner_backend/full_cuda_graph_backend.py']
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for name in hashes:
            p=Path(name);dest=run/'source-snapshots'/p.relative_to(guard.PROJECT)
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        guard.save(run/'manifest.json',dict(host=socket.gethostname(),pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
            source_sha256=hashes,capacity=16,deterministic=False,unified_decode=False,
            residual_rms_candidate=False,qk_candidate=False,restore_keepalive=False,
            correctness_gate_prefill_max_requests=1,benchmark_prefill_max_requests=None,
            protocol=dict(warmup=3,steps=20,response_tokens=128),
            scope='Full decode model forward; prefill eager; timing is end-to-end Engine.generate'))
        guard.save(run/'resources.json',resources);guard.p2p_preflight(run)
        reference=run/'exact-original'
        guard.command(run,'exact-original',[str(guard.PYTHON),str(guard.PREP/'gpu_tools/baseline_graph_model_gate.py'),
            'capture','--mode','original','--output',str(reference)],env,memory,timeout=1800)
        verify(reference,'breakable',gate=True);breakable_proof(reference,'original')
        for backend in ('eager','full'):
            output=run/('exact-'+backend)
            command=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/baseline_full_model_gate.py'),
                '--backend',backend,'--output',str(output)]
            if backend=='eager':command.append('--shadow')
            guard.command(run,'exact-'+backend,command,env,memory,timeout=1800)
            verify(output,'disabled' if backend=='eager' else 'full',gate=True)
            if backend=='full':full_proof(output)
            else:
                stats=[json.loads(p.read_text()) for p in (output/'graph-evidence').glob('fixed-shadow-rank-*.json')]
                assert {p['rank'] for p in stats}==set(range(8))
                assert all(p['counts'].get('shadow_calls',0)>0 for p in stats)
                guard.save(run/'fixed-shadow-summary.json',dict(passed=True,stats=stats))
            guard.command(run,'compare-'+backend,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/baseline_graph_model_gate.py'),
                'compare','--reference',str(reference),'--candidate',str(output),
                '--output',str(run/('compare-'+backend+'.json'))],env,memory)
        results={}
        for mode in ('fixed-experts','full'):
            output=run/mode;output.mkdir();worker=dict(env)
            worker['SLIME_BASELINE_GRAPH_MODE']=mode
            script='baseline_full_inference_benchmark.py' if mode=='full' else 'baseline_graph_inference_benchmark.py'
            guard.command(run,mode,[str(guard.PYTHON),str(guard.PREP/'gpu_tools'/script),
                '--run',str(output),'--variant','baseline','--residual-rms-mode','off','--router-tile-mode','off',
                '--warmup','3','--steps','20','--response-tokens','128'],worker,memory,timeout=1800)
            verify(output,'full' if mode=='full' else 'breakable')
            if mode=='full':full_proof(output)
            else:breakable_proof(output,mode)
            summary=json.loads((output/'summary.json').read_text())
            assert summary['formal_batches']==20 and not summary['profiled']
            results[mode]=summary;guard.save(run/'performance-progress.json',results)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        baseline=results['fixed-experts']['median_s']
        result=dict(model_exact_passed=True,fixed_capacity_shadow_passed=True,full_capture_replay_passed=True,
            summaries=results,delay_reduction_vs_breakable_percent=100*(1-results['full']['median_s']/baseline),
            limitation='One sequential launch per mode; baseline batch-sensitive; sampled outputs may differ under natural scheduling')
        guard.save(run/'results.json',result);guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:guard.stop_worker();print('BASELINE_FULL_REMAINING_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':main()
