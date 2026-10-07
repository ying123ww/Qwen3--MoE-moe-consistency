"""Finish the submitted aligned online run, then launch its configured baseline."""
import json
from pathlib import Path
import subprocess
import time
from continue_a import BASE, PYTHON, wait_job, run


def main():
    aligned='a-online-aligned-20260910-01'
    wait_job(aligned)
    run('status.py', aligned)
    rows=[json.loads(x) for x in (BASE/'gpu_runs'/aligned/'iterations.jsonl').read_text().splitlines()]
    assert len(rows)==6 and sum(not x['warmup'] for x in rows)==5
    deadline=time.monotonic()+300
    while True:
        try:
            occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip()
            if not occupied:
                break
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic()>deadline:
            raise RuntimeError('Previous GPU workers not released')
        time.sleep(5)
    native='a-online-native-20260910-01'
    run('launch.py','--mode','native','--online-timing','--steps','6','--run-id',native)
    wait_job(native)
    run('status.py',native)
    print('ONLINE_PAIR_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
