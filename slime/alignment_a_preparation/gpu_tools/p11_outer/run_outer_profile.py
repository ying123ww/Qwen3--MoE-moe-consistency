import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import residual_rms_performance as guard
from resource_preflight import snapshot,validate
from run_fixed_capacity_pipeline import full_proof

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--nsys',type=Path,required=True);a=p.parse_args()
    run=guard.PREP/'gpu_runs'/a.run_id;run.mkdir(exist_ok=False);folder=Path(__file__).parent.resolve()
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    try:
        if guard.gpu_pids():raise RuntimeError('GPU task active')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Memory cgroup missing')
        env=guard.environment(run);env['PYTHONPATH']=str(folder)+':'+str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        nccl=guard.ENVROOT/'lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2';env['SGLANG_NCCL_SO_PATH']=str(nccl);env['LD_LIBRARY_PATH']=str(nccl.parent)+':'+env.get('LD_LIBRARY_PATH','')
        sources=sorted(folder.glob('*.py'))+[guard.PREP/'gpu_tools/full_graph_profile_hooks.py'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        guard.save(run/'manifest.json',dict(source_sha256=hashes,scope='P11 actual graph-outside prefill/sampling complement, NCU still waived',restore_keepalive=False));guard.p2p_preflight(run);results={}
        for mode in ('A','C'):
            out=run/mode;out.mkdir();(out/'nsys').mkdir()
            cmd=[str(a.nsys),'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output='+str(out/'nsys/timeline'),str(guard.PYTHON),str(folder/'outer_profile_benchmark.py'),'--mode',mode,'--run',str(out)]
            guard.command(run,mode,cmd,env,groups[0]/'memory.usage_in_bytes',timeout=1800)
            full_proof(out);records=[json.loads(p.read_text()) for p in (out/'profile-evidence').glob('outer-rank-*.json')]
            if {x['rank'] for x in records}!=set(range(8)) or any(x['counts'].get('prefill',0)<=0 for x in records):raise RuntimeError('Actual prefill evidence missing')
            reports=list((out/'nsys').glob('*.nsys-rep'))
            if len(reports)!=1:raise RuntimeError('Expected one actual prefill report')
            with (out/'nsys/export.log').open('w') as log:subprocess.run([str(a.nsys),'export','--type=sqlite','--force-overwrite=true','--output='+str(out/'nsys/prefill.sqlite'),str(reports[0])],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            results[mode]=records
        if not all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()):raise RuntimeError('Source changed')
        guard.save(run/'results.json',dict(complete=True,records=results));guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as e:guard.save(run/'state.json',dict(status='failed',pid=os.getpid(),error=str(e),time=time.time()));raise
    finally:guard.stop_worker();print('P11_OUTER_GPU_PIDS='+guard.gpu_pids(),flush=True)

if __name__=='__main__':main()
