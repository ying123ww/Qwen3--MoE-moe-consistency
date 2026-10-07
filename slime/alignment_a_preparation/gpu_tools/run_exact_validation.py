"""Guarded router model gate or isolated Ray B1 forward-only validation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler, Request

from launch import ROOT, PREP, ENV, check_memory_guard
from resource_preflight import snapshot, validate, GIB
from run_matmul_attribution import safe_run_id


def ray_job(argv, env, run, resources):
    ray_tmp = Path('/dev/shm') / ('rp-' + run.name[-14:])
    if ray_tmp.exists():
        raise RuntimeError('Ray scratch exists; reconcile before reuse')
    for port in (6387, 8267):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    # No pre-existing user Ray runtime may be stopped by this isolated run.
    ps = subprocess.check_output(['ps','-u',str(os.getuid()),'-o','comm='], text=True)
    if any(name.strip() in ('raylet','gcs_server') for name in ps.splitlines()):
        raise RuntimeError('Existing Ray service; refusing to own its lifecycle')
    ray_env = dict(env, RAY_TMPDIR=str(ray_tmp))
    opener = build_opener(ProxyHandler({}))
    address = 'http://127.0.0.1:8267/api/jobs/'
    started = False
    try:
        started = True
        with (run/'ray-start.log').open('w') as log:
            subprocess.run([str(ENV/'bin/ray'),'start','--head','--node-ip-address','127.0.0.1','--port','6387',
                            '--num-gpus','8','--num-cpus',str(resources['ray_cpus']),
                            '--object-store-memory',str(4*GIB),'--disable-usage-stats',
                            '--dashboard-host','127.0.0.1','--dashboard-port','8267','--temp-dir',str(ray_tmp)],
                           env=ray_env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        body = {'entrypoint':shlex.join(argv),'submission_id':run.name,'runtime_env':{'env_vars':env}}
        with opener.open(Request(address,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=30) as response:
            (run/'submission.json').write_text(json.dumps(json.load(response),indent=2)+'\n')
        deadline=time.monotonic()+2400
        while True:
            with opener.open(address+run.name,timeout=15) as response:
                state=json.load(response)
            (run/'job-status.json').write_text(json.dumps(state,indent=2)+'\n')
            with opener.open(address+run.name+'/logs',timeout=15) as response:
                (run/'job.log').write_text(json.load(response)['logs'])
            if state['status'] in ('SUCCEEDED','FAILED','STOPPED'):
                return 0 if state['status']=='SUCCEEDED' else 1
            if time.monotonic()>deadline:
                with opener.open(Request(address+run.name+'/stop',data=b'{}',headers={'Content-Type':'application/json'}),timeout=15):
                    pass
                raise TimeoutError('B1 job exceeded 40 minute bound')
            time.sleep(5)
    finally:
        if started:
            with (run/'ray-stop.log').open('w') as log:
                subprocess.run([str(ENV/'bin/ray'),'stop'],env=ray_env,stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
            if ray_tmp.exists():
                shutil.rmtree(ray_tmp)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind',choices=['router','b1'],required=True)
    ap.add_argument('--mode',choices=['off','shadow','candidate'],default='off')
    ap.add_argument('--run-id',type=safe_run_id,required=True)
    ap.add_argument('--rollout-path',type=Path)
    args=ap.parse_args()
    if args.kind=='b1' and (args.rollout_path is None or args.mode!='off'):
        ap.error('B1 requires a prepared rollout and router mode off')
    assert socket.gethostname().split('.')[0]=='ai_hgx_03'
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lock=open('/tmp/alignment-a-launch.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=PREP/'gpu_runs'/args.run_id
    run.mkdir(exist_ok=False)
    tmp=ROOT.parent/'.a-tmp'/run.name
    tmp.mkdir(exist_ok=False)
    resources=snapshot(str(tmp))
    validate(resources,512,20)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU compute processes are active')
    jobs=[Path(g['path']) for g in resources['cgroups'] if Path(g['path']).name.startswith('job_') and 'memory' in g['path']]
    if len(jobs)!=1:
        raise RuntimeError('Cannot identify allocation memory cgroup')
    (run/'resource-preflight.json').write_text(json.dumps(resources,indent=2)+'\n')
    frozen=json.loads((PREP/'gpu_runs/a-restore-20260909-06/runtime_env.json').read_text())['env_vars']
    env=dict(frozen)
    env.update({k:str(tmp) for k in ('TMPDIR','TMP','TEMP')})
    cache=Path('/dev/shm')/run.name
    cache.mkdir(exist_ok=False)
    for key,sub in {'TVM_FFI_CACHE_DIR':'tvm','TRITON_CACHE_DIR':'triton','TORCHINDUCTOR_CACHE_DIR':'inductor','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg'}.items():
        dest=cache/sub
        dest.mkdir()
        env[key]=str(dest)
    env.update(ALIGN_A_RUN_DIR=str(run),SLIME_QWEN3_ROUTER_TILE_MODE='off')
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        env[key]=''
    env['NO_PROXY']=env['no_proxy']='*'
    for folder in (tmp,cache,run):
        p=folder/'write-probe'
        p.write_text('ok')
        p.unlink()
    if args.kind=='router':
        for key in list(env):
            if key.startswith('MEGATRON_USE_SGLANG_'):
                env.pop(key)
        argv=[str(ENV/'bin/python'),str(PREP/'gpu_tools/router_tile_exact.py'),'capture','--mode',args.mode,
              '--reference-manifest',str(PREP/'gpu_runs/a-matmul-attribution-20260912-01/manifest.json'),
              '--frozen-rollout','/public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt',
              '--output',str(run/'capture'),'--response-tokens','16']
    else:
        sys.path.insert(0,str(ROOT/'tools'))
        from routepack.b1_gate import configure_launch
        options=json.loads((PREP/'v89_cli_static.json').read_text())['options_last_value']
        substitutions={'$SLIME_DIR':str(ROOT),'$HF_MODEL':'/storage/data/yingxy2024/Models/Qwen3-30B-A3B',
                       '$CKPT':'/storage/data/yingxy2024/Models/Qwen3-30B-A3B_torch_dist','$ALIGN_ROOT':str(run),
                       '$DEEPEP_CONFIG':env['SLIME_QWEN3_FIXED_DEEPEP_CONFIG']}
        for key,values in options.items():
            for old,new in substitutions.items():
                values=[v.replace(old,new) for v in values]
            options[key]=values
        options,env=configure_launch(options,env,run=run,rollout_path=args.rollout_path.resolve())
        argv=[str(ENV/'bin/python'),str(ROOT/'train.py')]
        for key,values in options.items():
            argv.extend([key,*values])
    if any('$' in arg for arg in argv):
        raise RuntimeError('Unexpanded command argument')
    (run/'argv.json').write_text(json.dumps(argv,indent=2)+'\n')
    (run/'runtime_env.json').write_text(json.dumps({'env_vars':env},indent=2)+'\n')
    srt=ROOT.parent/'experiments/sglang-main-sm80/python/sglang/srt'
    sources=[Path(__file__),PREP/'gpu_tools/router_tile_exact.py',srt/'models/qwen3_router_tile.py',srt/'models/qwen3_moe.py',ROOT/'tools/routepack/b1_gate.py']
    if args.kind=='b1':
        sources.extend([ROOT/'tools/routepack/prepare_b1.py', ROOT/'tools/routepack/runtime_topology.py',
                        ROOT/'slime/backends/megatron_utils/cp_utils.py', ROOT/'slime/utils/routing_replay.py',
                        Path('/public/home/yingxy2024/slime-deps/Megatron-LM/megatron/core/transformer/moe/router.py'),
                        Path('/public/home/yingxy2024/slime-deps/Megatron-LM/megatron/core/transformer/moe/moe_utils.py')])
        (run/'input.json').write_text(json.dumps({'path':str(args.rollout_path.resolve()),
             'sha256':hashlib.sha256(args.rollout_path.read_bytes()).hexdigest()},indent=2)+'\n')
    (run/'source-hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    guard_state=run/'memory-guard.json'
    os.environ['ALIGN_A_MEMORY_GUARD']=str(guard_state)
    guard=subprocess.Popen([str(ENV/'bin/python'),str(PREP/'gpu_tools/memory_guard.py'),'--cgroup',str(jobs[0]),'--state',str(guard_state)])
    process_env=dict(os.environ,**env)
    try:
        for _ in range(50):
            if guard_state.exists():
                break
            time.sleep(.1)
        check_memory_guard(resources)
        if args.kind=='b1':
            code=ray_job(argv,env,run,resources)
        else:
            with (run/'job.log').open('w') as log:
                code=subprocess.run(argv,env=process_env,stdout=log,stderr=subprocess.STDOUT,timeout=2400).returncode
        (run/'exit.json').write_text(json.dumps({'returncode':code})+'\n')
        if code:
            raise RuntimeError(f'Validation failed: {run.name}; inspect job.log')
        if args.kind=='b1':
            gates=[json.loads((run/'b1'/f'rank-{rank}'/'gate.json').read_text()) for rank in range(8)]
            if any(g['rank']!=rank or not g['passed'] or g['optimizer_steps']!=0 or not g['r3_all_48_layers_consumed']
                   for rank,g in enumerate(gates)):
                raise RuntimeError('B1 rank evidence is incomplete or failed')
            (run/'gate-complete.json').write_text(json.dumps({'ranks':8,'passed':True,'optimizer_steps':0})+'\n')
        else:
            completed=json.loads((run/'capture'/'complete.json').read_text())
            if completed != {'mode':args.mode,'cases':4}:
                raise RuntimeError('Router capture completion does not match requested mode/workloads')
        shutil.rmtree(cache)
        shutil.rmtree(tmp)
    finally:
        guard.terminate()
        guard.wait(timeout=10)
    print('VALIDATION_COMPLETE '+run.name,flush=True)


if __name__=='__main__':
    main()
