"""P14 cap16/cap8 finite multibatch model gates with resident GPU sampling."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import residual_rms_performance as guard
from resource_preflight import snapshot,validate
from run_fixed_capacity_pipeline import full_proof


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--timing-run',type=Path,required=True);args=parser.parse_args()
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    timing=json.loads((args.timing_run/'results.json').read_text())
    if not timing['complete']:raise RuntimeError('P12 prerequisite missing')
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    try:
        if guard.gpu_pids():raise RuntimeError('Other GPU task active')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Memory cgroup missing')
        env=guard.environment(run);folder=Path(__file__).parent.resolve()
        env['PYTHONPATH']=str(folder)+':'+str(guard.PREP/'gpu_tools')+':'+str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env['FLASHINFER_WORKSPACE_BASE']=str(Path('/dev/shm')/run.name)
        nccl=guard.ENVROOT/'lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2';env['SGLANG_NCCL_SO_PATH']=str(nccl);env['LD_LIBRARY_PATH']=str(nccl.parent)+':'+env.get('LD_LIBRARY_PATH','')
        paths=sorted(folder.glob('*.py'));hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        guard.save(run/'manifest.json',dict(variant=timing['winner'],source_sha256=hashes,scope='Ten finite workloads cap16/cap8; globalmaxRunning8=>DP2 workermax4=>CP2send2. Static guard beforeDeepEP; no crossnode claim',restore_keepalive=False))
        guard.p2p_preflight(run);models={};resident={}
        for cap in (16,8):
            output=run/f'cap-{cap}';output.mkdir();stopped=threading.Event();observations=[];errors=[]
            def monitor():
                while not stopped.is_set():
                    try:
                        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15)
                        rows=[dict(index=int(a.strip()),used_mib=int(b.strip()),total_mib=int(c.strip())) for a,b,c in (x.split(',') for x in text.strip().splitlines())]
                        observations.append(dict(time=time.time(),gpus=rows))
                    except Exception as exc:errors.append(str(exc))
                    stopped.wait(1)
            watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
            try:
                guard.command(run,f'cap-{cap}',[str(guard.PYTHON),str(folder/'capacity_model_probe.py'),'--run',str(output),'--capacity',str(cap),'--variant',timing['winner']],env,groups[0]/'memory.usage_in_bytes',timeout=2400)
            finally:stopped.set();watcher.join(timeout=20)
            guard.save(output/'nvml-resident.json',dict(observations=observations,errors=errors,scope='1s periodic whole-device resident sample, includes non-Torch allocations; sampled peak can miss transients'))
            if not observations:raise RuntimeError('No resident GPU observations')
            full_proof(output);models[cap]=json.loads((output/'results.json').read_text())
            memory=[json.loads(p.read_text()) for p in (output/'memory-evidence').glob('memory-rank-*.json')]
            if {x['rank'] for x in memory}!=set(range(8)):raise RuntimeError('Incomplete memory ranks')
            if any(x['counts'].get('unsafe_physical_send_rejected',0)!=1 or set(x['replay_shapes'])!={'2','4'} for x in memory):raise RuntimeError('Capacity boundary/bucket gate not covered')
            if any(x['counts'].get('pure_eager_decode',0)!=0 for x in memory):raise RuntimeError('Supported pure decode unexpectedly fell back to eager')
            proofs=[json.loads(p.read_text()) for p in (output/'graph-evidence').glob('fixed-capture-size-*-rank-*.json')]
            for proof in proofs:
                if proof['capacity']!=cap or any(int(k)>cap//4 for k in proof['source_rows']):raise RuntimeError('Any rank physical send bound unproved')
            resident[cap]=dict(memory=memory,nvml_peak_mib={i:max(g['used_mib'] for x in observations for g in x['gpus'] if g['index']==i) for i in range(8)})
        if len(models[16]['cases'])!=10 or len(models[8]['cases'])!=10:raise RuntimeError('Incomplete ten-workload model gate')
        for left,right in zip(models[16]['cases'],models[8]['cases']):
            if left['case']!=right['case'] or len(left['records'])!=len(right['records']):raise RuntimeError('Case identity differs')
            for a,b in zip(left['records'],right['records']):
                for key in ('prompt','response_length','tokens','logprob_bits','route_sha256','route_shape'):
                    if a[key]!=b[key]:raise RuntimeError(f'Capacity model mismatch case={left["case"]} field={key}')
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        guard.save(run/'results.json',dict(complete=True,capacity_model_exact=True,cases=10,memory=resident,scope='Finite cap8/16model equivalence and actualsendguard/memory; standalone routingzero/idle/shadowexpert checks remain separate'))
        guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:guard.stop_worker();print('P14_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':main()
