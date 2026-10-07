"""Sequential A-stage gates; stop on any failure, never overlap GPU runs."""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import build_opener, ProxyHandler

BASE = Path('/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation')
PYTHON = '/public/home/yingxy2024/.conda/envs/slime/bin/python'
HTTP = build_opener(ProxyHandler({}))


def wait_job(job):
    while True:
        with HTTP.open('http://127.0.0.1:8267/api/jobs/'+job, timeout=15) as response:
            state = json.load(response)
        (BASE/'gpu_runs'/job/'job-status.json').write_text(json.dumps(state,indent=2)+'\n')
        if state['status'] == 'SUCCEEDED':
            return
        if state['status'] in ('FAILED','STOPPED'):
            raise RuntimeError(job+': '+str(state.get('message')))
        time.sleep(10)


def run(*args):
    subprocess.run([PYTHON, str(BASE/'gpu_tools'/args[0]), *args[1:]],check=True)


if __name__ == '__main__':
    native='a-native-correctness-20260910-01'
    wait_job(native)
    run('status.py',native)
    run('check_run.py',str(BASE/'gpu_runs'/native),'--mode','native')
    for variant in ('aligned','native'):
        job='a-'+variant+'-timing-20260910-02'
        # Allow actor teardown after Ray reports completion; launch independently checks GPU emptiness.
        deadline = time.monotonic() + 300
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip():
            if time.monotonic() >= deadline:
                raise RuntimeError('GPU processes remain after completion; inspect before retry')
            time.sleep(5)
        run('launch.py','--mode',variant,'--run-id',job,'--steps','26','--timing')
        wait_job(job)
        run('status.py',job)
        run('summarize.py',str(BASE/'gpu_runs'/job),'--warmup','5')
    print('PAIR_COMPLETE',flush=True)
