"""Sequential isolated inference processes under the existing 900 GiB guard."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from launch import check_memory_guard, ROOT, PREP, ENV
from resource_preflight import snapshot, validate


def main():
    assert socket.gethostname().split('.')[0] == 'ai_hgx_03'
    lock = open('/tmp/alignment-a-launch.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runs = PREP/'gpu_runs'
    state = runs/'inference-pairs-20260910-state.json'
    plan = [(1,'baseline'), (1,'aligned'), (2,'aligned'), (2,'baseline'), (3,'baseline'), (3,'aligned')]
    for pair, variant in plan:
        run = runs/f'a-inference-{variant}-20260910-{pair:02d}'
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
        env['TVM_FFI_CACHE_DIR'] = '/dev/shm/alignment-a-tvm-ffi-999484'
        env['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE'] = '1' if variant == 'aligned' else '0'
        for key in list(env):
            if key.startswith('MEGATRON_USE_SGLANG_'):
                env.pop(key)
        (run/'runtime_env.json').write_text(json.dumps({'env_vars': {k:env[k] for k in json.loads((runs/'a-restore-20260909-06/runtime_env.json').read_text())['env_vars'] if k in env}}, indent=2)+'\n')
        argv = [str(ENV/'bin/python'),str(PREP/'gpu_tools/inference_benchmark.py'),'--run',str(run),'--variant',variant]
        (run/'argv.json').write_text(json.dumps(argv)+'\n')
        state.write_text(json.dumps({'status':'running','active':run.name})+'\n')
        check_memory_guard(resources)
        with (run/'job.log').open('w') as log:
            result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        (run/'exit.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
        if result.returncode:
            raise RuntimeError(f'{run.name} failed: {result.returncode}')
    state.write_text(json.dumps({'status':'complete','active':None})+'\n')
    print('INFERENCE_PAIRS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
