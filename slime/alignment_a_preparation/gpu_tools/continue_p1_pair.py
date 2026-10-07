"""Bounded sequential P1 pair; fail closed if aligned capture is incomplete."""
import subprocess
import time
from continue_a import BASE, wait_job, run


def validate(job):
    root=BASE/'gpu_runs'/job
    for rank in (0,4):
        for stage in ('train','weight_sync'):
            assert (root/f'profiles/train/{stage}-rollout3-rank{rank}.json.gz').is_file()
    assert list((root/'profiles/rollout').glob('*.json.gz'))
    run('summarize_p1.py',str(root))


if __name__ == '__main__':
    aligned='a-p1-aligned-20260910-02'
    wait_job(aligned)
    run('status.py',aligned)
    validate(aligned)
    deadline=time.monotonic()+300
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip():
        if time.monotonic()>deadline:
            raise RuntimeError('GPU workers still present')
        time.sleep(5)
    native='a-p1-native-20260910-01'
    run('launch.py','--mode','native','--online-timing','--p1-profile','--steps','3','--run-id',native)
    wait_job(native)
    run('status.py',native)
    validate(native)
    print('P1_PAIR_COMPLETE',flush=True)
