"""Sequential isolated inference processes under the existing 900 GiB guard."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import resource
import shutil
import urllib.request

from launch import check_memory_guard, ROOT, PREP, ENV
from resource_preflight import snapshot, validate


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    assert socket.gethostname().split('.')[0] == 'ai_hgx_03'
    lock = open('/tmp/alignment-a-launch.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runs = PREP/'gpu_runs'
    state = runs/'p1-inference-20260910-state.json'
    plan = [(1,'baseline'), (1,'aligned')]
    for pair, variant in plan:
        run = runs/f'a-p1-inference-{variant}-20260910-{pair:02d}'
        if (run/'exit.json').exists() and json.loads((run/'exit.json').read_text())['returncode'] == 0 and (run/'summary.json').exists():
            continue
        resources = snapshot(os.environ['ALIGN_A_RAY_TMP'])
        validate(resources, 512, 20)
        check_memory_guard(resources)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:8267/api/jobs/', timeout=10) as response:
            jobs = json.load(response)
        if any(j['status'] in ('RUNNING', 'PENDING') for j in jobs):
            raise RuntimeError('Ray job active')
        deadline = time.monotonic()+300
        while True:
            try:
                gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True, timeout=15)
                if not gpu.strip():
                    break
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic()>deadline:
                raise RuntimeError('GPUs did not become free')
            time.sleep(5)
        if run.exists():
            if not (run/'exit.json').exists() or json.loads((run/'exit.json').read_text())['returncode'] == 0:
                raise RuntimeError('Existing unverified run: reconcile before retry')
            run.rename(run.with_name(run.name + '-failed-' + str(int(time.time()))))
        run.mkdir(exist_ok=False)
        (run/'resource-preflight.json').write_text(json.dumps(resources, indent=2)+'\n')
        env = dict(os.environ)
        env.update(json.loads((runs/'a-restore-20260909-06/runtime_env.json').read_text())['env_vars'])
        tmp = ROOT.parent/'.a-tmp'/run.name
        tmp.mkdir(exist_ok=True)
        env.update({k:str(tmp) for k in ['TMPDIR','TMP','TEMP']})
        cache = Path('/dev/shm')/('p1-cache-'+run.name)
        cache.mkdir(exist_ok=False)
        for key, subdir in {'TVM_FFI_CACHE_DIR':'tvm','TRITON_CACHE_DIR':'triton', 'TORCHINDUCTOR_CACHE_DIR':'inductor','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg'}.items():
            target=cache/subdir
            target.mkdir()
            env[key]=str(target)
        for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
            env[key]=''
        env['NO_PROXY']=env['no_proxy']='*'
        env['SGLANG_PROFILE_WITH_STACK']='0'
        # Verify all writable destinations before importing torch.
        for folder in (tmp,cache,run):
            probe=folder/'write-probe'
            probe.write_bytes(b'ok')
            probe.unlink()
        env['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE'] = '1' if variant == 'aligned' else '0'
        for key in list(env):
            if key.startswith('MEGATRON_USE_SGLANG_'):
                env.pop(key)
        (run/'runtime_env.json').write_text(json.dumps({'env_vars': {k:env[k] for k in json.loads((runs/'a-restore-20260909-06/runtime_env.json').read_text())['env_vars'] if k in env}}, indent=2)+'\n')
        argv = [str(ENV/'bin/python'),str(PREP/'gpu_tools/inference_benchmark.py'),'--run',str(run),'--variant',variant,'--warmup','3','--steps','1','--profile']
        (run/'argv.json').write_text(json.dumps(argv)+'\n')
        state.write_text(json.dumps({'status':'running','active':run.name})+'\n')
        check_memory_guard(resources)
        with (run/'job.log').open('w') as log:
            result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        (run/'exit.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
        if result.returncode == 0:
            shutil.rmtree(cache)
            shutil.rmtree(tmp)
        if result.returncode:
            raise RuntimeError(f'{run.name} failed: {result.returncode}')
    state.write_text(json.dumps({'status':'complete','active':None})+'\n')
    print('P1_INFERENCE_PAIR_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
