"""Sequential frozen-data training capture; keep evidence, clean successful-run scratch."""
import json
import shutil
import subprocess
import time
from pathlib import Path
from continue_a import BASE, wait_job, run

if __name__ == '__main__':
    for variant in ('native','aligned'):
        job=f'a-p1-train-{variant}-20260910-01'
        root=BASE/'gpu_runs'/job
        run('launch.py','--mode',variant,'--timing','--p1-train-profile','--steps','3','--run-id',job)
        wait_job(job)
        run('status.py',job)
        for rank in (0,4):
            assert (root/f'profiles/train/train-rollout3-rank{rank}.json.gz').is_file()
        run('summarize_p1.py',str(root))
        deadline=time.monotonic()+300
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip():
            if time.monotonic()>deadline:
                raise RuntimeError('GPU workers remain after completed training')
            time.sleep(5)
        for folder in (BASE.parent.parent/'.a-tmp'/job, Path('/dev/shm')/('p1-cache-'+job)):
            shutil.rmtree(folder)
        (root/'scratch-cleaned.json').write_text(json.dumps({'cleaned':True})+'\n')
    print('P1_TRAIN_PAIR_COMPLETE',flush=True)
