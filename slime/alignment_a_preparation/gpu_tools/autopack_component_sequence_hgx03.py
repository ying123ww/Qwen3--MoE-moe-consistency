"""Bounded Native/RoutePack component attribution; restore keeper on exit."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import autopack_gpu_compare_sequence as seq

BASE=seq.ROOT/'alignment_a_preparation/gpu_runs/autopack-components-hgx03-20260916-02'
PLANS=seq.ROOT/'alignment_a_preparation/gpu_runs/autopack-gpu-compare128-20260914/plans'
original=seq.command_for

def command_for(mode,method):
    run,_,cmd=original(mode,method)
    profile=Path(f'/tmp/ap-components-{method}-20260916-02')
    options={'--rollout':'/tmp/ap-real-capture128-20260914/rollout.pt',
             '--global-batch-size':'128','--rollout-batch-size':'16','--expected-samples':'128',
             '--max-tokens-per-gpu':'10240','--warmups':'1','--repeats':'1',
             '--cuda-home':'/tmp/slime-cuda-alignment-a-20260909',
             '--profile-directory':str(profile),'--schedule-manifest':str(PLANS/method/'runtime.json')}
    for key,value in options.items():cmd[cmd.index(key)+1]=value
    cmd+=['--expected-host','ai_hgx_03','--component-profile']
    return run,profile,cmd

def gpu_pids():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()

def restore():
    for _ in range(12):
        if not gpu_pids():break
        time.sleep(5)
    if gpu_pids():
        return {'status':'not_started_GPU_processes_remain','pids':gpu_pids()}
    script=seq.ROOT/'alignment_a_preparation/gpu_runs/autopack-real-hgx05-20260914/restore_keepalive.sh'
    log=BASE/'keepalive.log'
    with log.open('x') as f:
        env=os.environ.copy()
        for key in ('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF'):
            env.pop(key,None)
        env['CUDA_LAUNCH_BLOCKING']='1'
        p=subprocess.Popen(['bash',str(script)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    return {'status':'submitted_requires_verification','pid':p.pid,'pgid':p.pid,'log':str(log)}

def main():
    if socket.gethostname().split('.')[0]!='ai_hgx_03':raise RuntimeError('ai_hgx_03 only')
    BASE.mkdir(exist_ok=False);(BASE/'runs').mkdir()
    seq.BASE=BASE;seq.ATTEMPT="02";seq.command_for=command_for
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    os.environ['ROUTEPACK_CHECKPOINT_CPU_OFFLOAD']='0'
    status={'status':'running','completed':[],'scope':'1 warmup, 1 formal pilot, 1 component attribution and 1 gradient validation per method; not a new stable speed benchmark'}
    (BASE/'plan.json').write_text(json.dumps([command_for('timing',m)[2] for m in ('native','routepack')],indent=2))
    try:
        for method in ('native','routepack'):
            print('starting',method,flush=True)
            status['completed'].append(seq.run_stage('timing',method))
            print('passed',method,flush=True)
        status['status']='passed'
    except Exception as exc:
        status.update(status='failed',error=str(exc));raise
    finally:
        status['keeper']=restore()
        (BASE/'status.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':main()
