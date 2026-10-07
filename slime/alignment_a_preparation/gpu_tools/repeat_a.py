"""Two further independent pairs, reversing the previous pair's order first."""
import json
import subprocess
import time
from continue_a import BASE, run, wait_job

if __name__ == '__main__':
    plan = [(3, 'native'), (3, 'aligned'), (4, 'aligned'), (4, 'native')]
    state_path = BASE/'gpu_runs/repeats-20260910-state.json'
    completed = []
    for pair, variant in plan:
        job = 'a-%s-timing-20260910-%02d' % (variant, pair)
        root = BASE/'gpu_runs'/job
        if (root/'measurement-summary.json').exists():
            assert json.loads((root/'job-status.json').read_text())['status'] == 'SUCCEEDED'
            completed.append(job)
            continue
        state_path.write_text(json.dumps({'active': job, 'completed': completed, 'status': 'waiting_for_free_gpus'})+'\n')
        deadline = time.monotonic() + 300
        while True:
            if time.monotonic() >= deadline:
                raise RuntimeError('GPU status remains occupied or unavailable; refusing overlap')
            try:
                occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip()
            except subprocess.TimeoutExpired:
                time.sleep(5)
                continue
            if not occupied:
                break
            time.sleep(5)
        run('launch.py','--mode',variant,'--run-id',job,'--steps','26','--timing')
        state_path.write_text(json.dumps({'active': job, 'completed': completed, 'status': 'running'})+'\n')
        wait_job(job)
        run('status.py',job)
        run('summarize.py',str(BASE/'gpu_runs'/job),'--warmup','5')
        completed.append(job)
    state_path.write_text(json.dumps({'active': None,'completed': completed,'status': 'complete'})+'\n')
    print('REPEATS_COMPLETE',flush=True)
