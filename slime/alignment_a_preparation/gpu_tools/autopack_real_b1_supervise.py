"""Bounded supervisor for one real-length B1 run on ai_hgx_05."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT = Path('/public/home/yingxy2024/moe-consistency/slime')
RUN = ROOT / 'alignment_a_preparation/gpu_runs/autopack-real-hgx05-b1-20260914-02'
PROFILE = Path('/tmp/ap-real-b1-arrays-20260914-02')
PYTHON = '/public/home/yingxy2024/.conda/envs/slime/bin/python'


def main():
    if socket.gethostname().split('.')[0] != 'ai_hgx_05':
        raise RuntimeError('ai_hgx_05 only')
    import psutil
    if os.environ.get('PYTORCH_ALLOC_CONF'):
        raise RuntimeError('Conflicting PYTORCH_ALLOC_CONF must be reviewed')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    logpath = RUN.with_suffix('.log')
    metrics = RUN.with_suffix('.memory.jsonl')
    command = [PYTHON, str(ROOT/'alignment_a_preparation/gpu_tools/autopack_b1_hgx05.py'),
        '--rollout','/tmp/ap-real-capture-20260914-02/capture/rollout.pt',
        '--output', str(RUN), '--cuda-home','/tmp/ap-cuda-20260914',
        '--global-batch-size','256','--rollout-batch-size','32','--n-samples-per-prompt','8',
        '--expected-samples','256','--max-tokens-per-gpu','20480','--balance-data',
        '--rollout-max-response-len','8192','--rollout-max-context-len','16384',
        '--profile-directory',str(PROFILE)]
    started = time.monotonic()
    reason = None
    with logpath.open('x') as log, metrics.open('x') as track:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned = psutil.Process(child.pid)
        while child.poll() is None:
            procs = [owned] + owned.children(recursive=True)
            rss = 0
            for proc in procs:
                try:
                    rss += proc.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            available = psutil.virtual_memory().available
            tmp_free = psutil.disk_usage('/tmp').free
            row = dict(time=time.time(), elapsed_s=time.monotonic()-started, pid=child.pid,
                rss_gib=rss/2**30, available_gib=available/2**30, tmp_free_gib=tmp_free/2**30,
                result_free_gib=psutil.disk_usage(RUN.parent).free/2**30)
            track.write(json.dumps(row)+'\n')
            track.flush()
            if rss > 1024*2**30: reason = 'owned_process_tree_rss_exceeds_1TiB'
            elif available < 128*2**30: reason = 'system_available_below_128GiB'
            elif tmp_free < 10*2**30: reason = 'temporary_disk_free_below_10GiB'
            elif row['result_free_gib'] < 1: reason = 'result_disk_free_below_1GiB'
            elif row['elapsed_s'] > 5400: reason = 'walltime_exceeds_5400s'
            if reason:
                # Exact children of our own driver only, never other GPU jobs.
                for proc in reversed(procs):
                    try: proc.terminate()
                    except psutil.NoSuchProcess: pass
                _, alive = psutil.wait_procs(procs, timeout=20)
                for proc in alive:
                    try: proc.kill()
                    except psutil.NoSuchProcess: pass
                break
            time.sleep(5)
        rc = child.wait()
    RUN.mkdir(exist_ok=True)
    (RUN/'supervisor-exit.json').write_text(json.dumps(dict(returncode=rc, stop_reason=reason,
        elapsed_s=time.monotonic()-started, command=command, log=str(logpath), metrics=str(metrics)),indent=2)+'\n')
    raise SystemExit(rc if rc else (1 if reason else 0))


if __name__ == '__main__':
    main()
