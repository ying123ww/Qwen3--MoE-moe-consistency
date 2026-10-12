import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import residual_rms_performance as guard
from resource_preflight import snapshot,validate
from run_fixed_capacity_pipeline import full_proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--variants',nargs='+',default=['reference','dense']);p.add_argument('--reference',type=Path);args=p.parse_args()
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    if args.reference is not None:
        reference=args.reference.resolve()
        if not (reference/'complete.json').is_file():raise RuntimeError('Reference gate incomplete')
        (run/'reference').symlink_to(reference,target_is_directory=True)
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    folder=Path(__file__).parent;paths=sorted(folder.glob('*.py'));hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    try:
        if guard.gpu_pids():raise RuntimeError('Other GPU task active')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup unavailable')
        env=guard.environment(run);env['PYTHONPATH']=str(folder)+':'+str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env['FLASHINFER_WORKSPACE_BASE']=str(Path('/dev/shm')/run.name)
        nccl=guard.ENVROOT/'lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2';env['SGLANG_NCCL_SO_PATH']=str(nccl);env['LD_LIBRARY_PATH']=str(nccl.parent)+':'+env.get('LD_LIBRARY_PATH','')
        guard.save(run/'resources.json',resources);guard.save(run/'manifest.json',dict(variants=args.variants,source_sha256=hashes,scope='Finite full model gates; no timing claims',restore_keepalive=False));guard.p2p_preflight(run)
        for variant in args.variants:
            output=run/variant
            guard.command(run,variant,[str(guard.PYTHON),str(folder/'p12_model_gate.py'),'--variant',variant,'--output',str(output)],env,groups[0]/'memory.usage_in_bytes',timeout=2400)
            full_proof(output)
            proofs=[json.loads(x.read_text()) for x in (output/'graph-evidence').glob('p12-capture-size-*-rank-*.json')]
            if {(x['rank'],x['size']) for x in proofs}!={(r,s) for r in range(8) for s in (2,4)}:raise RuntimeError('Incomplete P12 capture proofs')
            runtime=[json.loads(x.read_text()) for x in (output/'graph-evidence').glob('runtime-gate-rank-*.json')]
            if len(runtime)!=8 or any(x['counts'].get('eager_decode',0) or x['counts'].get('full_replays',0)!=64 for x in runtime):raise RuntimeError('Native runtime replay/fallback gate failed')
            if variant.startswith('dense'):
                dense=[json.loads(x.read_text()) for x in (output/'graph-evidence').glob('dense-capture-size-*-rank-*.json')]
                if len(dense)!=16 or any(any(x['layers'][k]!=list(range(48)) for k in ('qkv','output_projection')) for x in dense):raise RuntimeError('Incomplete all48 dense coverage')
            if variant!='reference':
                guard.command(run,'compare-'+variant,[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_model_gate.py'),'compare','--reference',str(run/'reference'),'--candidate',str(output),'--output',str(run/f'compare-{variant}.json')],env,groups[0]/'memory.usage_in_bytes',timeout=300)
            guard.save(run/'progress.json',dict(completed_variant=variant,time=time.time()))
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:guard.stop_worker();print('P12_REMAINING_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':main()
