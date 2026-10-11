"""Gate RMS/QK graph stacking, then compare three current Breakable modes."""
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

import residual_rms_performance as guard
from resource_preflight import snapshot, validate
from run_breakable_model_and_performance import graph_proof


def fusion_proof(output, mode):
    records = [json.loads(p.read_text()) for p in (output/'graph-evidence').glob('fusion-*.json')]
    assert {(r['rank'], r['size']) for r in records} == {(r,s) for r in range(8) for s in (2,4)}
    assert all(r['rms_captured_calls'] > 0 and r['segments'] == 49 and r['breaks'] == 48 for r in records)
    assert all(r['qk_captured_calls'] == (48 if mode == 'breakable-rms-qk' else 0) for r in records)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='breakable-fusions-hgx03-20261011-03')
    args=parser.parse_args()
    if not args.run_id.startswith('breakable-fusions-') or '/' in args.run_id:
        raise ValueError('Invalid run ID')
    if socket.gethostname() != 'ai_hgx_03':
        raise RuntimeError('Use ai_hgx_03')
    signal.signal(signal.SIGTERM, guard.interrupted)
    signal.signal(signal.SIGINT, guard.interrupted)
    run = guard.PREP/'gpu_runs'/args.run_id
    run.mkdir(exist_ok=False)
    lock = open(guard.PROJECT/'.a-tmp/router-tile-performance.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if guard.gpu_pids():
            raise RuntimeError('Other GPU task exists')
        resources = snapshot(str(guard.PROJECT/'.a-tmp'))
        validate(resources, 512, 20)
        groups = [Path(x['path']) for x in resources['cgroups']
                  if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups) != 1:
            raise RuntimeError('Allocation memory cgroup missing')
        guard.save(run/'resources.json', resources)
        env = guard.environment(run)
        env['PYTHONPATH'] = str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env['SGLANG_MEMORY_SAVER_CUDA_GRAPH'] = '0'
        env['SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH'] = '0'
        srt = guard.PROJECT/'experiments/sglang-main-sm80/python/sglang/srt'
        paths = [Path(__file__), *guard.PREP.glob('gpu_tools/breakable_*.py'),
            guard.PREP/'gpu_tools/check_graph_fusions.py', guard.PREP/'gpu_tools/graph_fusion_proof.py',
            guard.PREP/'gpu_tools/qk_runtime_hooks.py', guard.PREP/'gpu_tools/qk_norm_prototype.py',
            guard.PREP/'gpu_tools/inference_benchmark.py', srt/'layers/attention/triton_backend.py',
            srt/'model_executor/model_runner_components/cuda_graph_setup.py',
            srt/'batch_invariant_ops/fused_rmsnorm_candidate.py']
        hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for name in hashes:
            p = Path(name)
            target = run/'source-snapshots'/p.relative_to(guard.PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
        guard.save(run/'manifest.json',dict(host=socket.gethostname(),pid=os.getpid(),
            job_id=os.environ.get('SLURM_JOB_ID'),source_sha256=hashes,restore_keepalive=False,
            protocol=dict(rounds=1,warmup=3,steps=20,response_tokens=128),
            scope='Current KV-once version; three deterministic Breakable graph variants'))
        guard.p2p_preflight(run)
        guard.command(run,'graph-tensor-gate',[str(guard.PYTHON),str(guard.PREP/'gpu_tools/check_graph_fusions.py'),
            '--output',str(run/'graph-tensor-gate.json')],env,groups[0]/'memory.usage_in_bytes')
        reference=guard.PREP/'gpu_runs/kv-once-baseline-graph-hgx03-20261011-01/exact-breakable'
        for mode in ('breakable-rms','breakable-rms-qk'):
            output=run/('exact-'+mode)
            guard.command(run,'exact-'+mode,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_model_gate.py'),
                'capture','--mode',mode,'--output',str(output)],env,groups[0]/'memory.usage_in_bytes')
            graph_proof(output)
            fusion_proof(output,mode)
            guard.command(run,'compare-'+mode,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_model_gate.py'),
                'compare','--reference',str(reference),'--candidate',str(output),
                '--output',str(run/('compare-'+mode+'.json'))],env,groups[0]/'memory.usage_in_bytes')
        results={}
        for mode in ('breakable','breakable-rms','breakable-rms-qk'):
            output=run/mode
            output.mkdir()
            worker=dict(env)
            worker['SLIME_BREAKABLE_EXPERIMENT_MODE']=mode
            worker['SLIME_GRAPH_FUSION_PREWARM']='1'
            worker['SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH']='0' if mode=='breakable' else '1'
            guard.command(run,mode,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_inference_benchmark.py'),
                '--run',str(output),'--variant','aligned','--residual-rms-mode','off' if mode=='breakable' else 'candidate',
                '--router-tile-mode','off','--warmup','3','--steps','20','--response-tokens','128'],worker,
                groups[0]/'memory.usage_in_bytes')
            graph_proof(output)
            if mode!='breakable':
                fusion_proof(output,mode)
            summary=json.loads((output/'summary.json').read_text())
            assert summary['formal_batches']==20 and not summary['profiled']
            resolved=json.loads((output/'resolved-server-args.json').read_text())
            assert resolved['enable_deterministic_inference'] and resolved['cuda_graph_config']['decode']['backend']=='breakable'
            results[mode]=summary
            guard.save(run/'performance-progress.json',results)
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
            raise RuntimeError('Source changed during experiment')
        base=results['breakable']['median_s']
        rms=results['breakable-rms']['median_s']
        qk=results['breakable-rms-qk']['median_s']
        result=dict(exact_passed=True,summaries=results,rms_reduction_percent=100*(1-rms/base),
            rms_qk_reduction_percent=100*(1-qk/base),qk_on_rms_reduction_percent=100*(1-qk/rms),
            limitation='One sequential engine launch per variant; no repeated-start paired estimate')
        guard.save(run/'results.json',result)
        guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()))
        raise
    finally:
        guard.stop_worker()
        print('FUSION_RUN_REMAINING_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':
    main()
