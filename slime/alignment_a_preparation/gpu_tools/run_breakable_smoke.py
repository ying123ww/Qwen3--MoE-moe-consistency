"""One guarded Plan 2 smoke Engine on the existing HGX03 allocation."""
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
from resource_preflight import snapshot, validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    if not args.run_id.startswith('breakable-smoke-') or '/' in args.run_id:raise ValueError('Invalid run ID')
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    try:
        lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if guard.gpu_pids():raise RuntimeError('Other GPU task exists')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        guard.save(run/'resources.json',resources)
        env=guard.environment(run)
        env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env['SLIME_BREAKABLE_EXPERIMENT_MODE']='breakable'
        env['SGLANG_MEMORY_SAVER_CUDA_GRAPH']='0'
        srt=guard.PROJECT/'experiments/sglang-main-sm80/python/sglang/srt'
        paths=[Path(__file__),guard.PREP/'gpu_tools/breakable_experiment.py',guard.PREP/'gpu_tools/breakable_inference_benchmark.py',
               srt/'server_args.py',srt/'layers/moe/ep_moe/layer.py',srt/'models/qwen3_moe.py',
               srt/'layers/attention/triton_backend.py',srt/'model_executor/runner_backend/breakable_cuda_graph_backend.py']
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        guard.save(run/'manifest.json',dict(host=socket.gethostname(),pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
            source_sha256=hashes,restore_keepalive=False,scope='Smoke only; no speedup claim'))
        guard.p2p_preflight(run)
        output=run/'smoke';output.mkdir()
        argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_inference_benchmark.py'),'--run',str(output),
              '--variant','aligned','--residual-rms-mode','off','--router-tile-mode','off',
              '--warmup','1','--steps','1','--response-tokens','16']
        guard.command(run,'smoke',argv,env,groups[0]/'memory.usage_in_bytes',timeout=1800)
        proof=[json.loads(p.read_text()) for p in (output/'graph-evidence').glob('rank-*-size-*.json')]
        if {p['rank'] for p in proof}!=set(range(8)):raise RuntimeError('Replay evidence missing ranks')
        if any(p['segments']!=49 or p['breaks']!=48 or not p['successful_replay'] for p in proof):
            raise RuntimeError('Unexpected graph split; inspect evidence')
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise RuntimeError('Source changed')
        guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time(),replay_ranks=8))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()))
        raise
    finally:guard.stop_worker()


if __name__=='__main__':main()
