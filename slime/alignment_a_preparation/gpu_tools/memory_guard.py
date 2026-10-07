"""Independent job-scoped memory cutoff. No Ray, torch, or GPU imports."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def targets(job, proc=Path('/proc'), uid=None, own=None):
    uid = os.getuid() if uid is None else uid
    own = os.getpid() if own is None else own
    result = []
    for p in proc.glob('[0-9]*'):
        try:
            if int(p.name) == own or p.stat().st_uid != uid:
                continue
            groups = (p/'cgroup').read_text().splitlines()
            if not any(job in line.split(':', 2)[-1].split('/') for line in groups):
                continue
            name = (p/'comm').read_text().strip()
            if name.startswith(('python', 'ray::', 'sglang')) or name in ('raylet', 'gcs_server', 'pt_main_thread'):
                result.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def cutoff(usage, threshold):
    return usage >= threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cgroup', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--threshold-gib', type=int, default=900)
    args = ap.parse_args()
    group, state = Path(args.cgroup), Path(args.state)
    job = group.name
    if not job.startswith('job_'):
        raise ValueError('Expected job-level cgroup')
    threshold = args.threshold_gib * 1024**3
    limit = int((group/'memory.limit_in_bytes').read_text())
    if threshold <= 0 or threshold >= limit:
        raise ValueError('Cutoff must be below the allocation limit')
    fired = False
    while group.exists():
        error = None
        try:
            usage = int((group/'memory.usage_in_bytes').read_text())
        except Exception as exc:
            usage, error = None, str(exc)
        fired = fired or error is not None or cutoff(usage, threshold)
        killed = []
        if fired:
            for pid in targets(job):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
                except ProcessLookupError:
                    pass
        record = {'pid': os.getpid(), 'time': time.time(), 'job': job,
                  'usage_bytes': usage, 'threshold_bytes': threshold, 'limit_bytes': limit,
                  'triggered': fired, 'signal_sent_to': killed, 'error': error}
        tmp = state.with_suffix('.new')
        tmp.write_text(json.dumps(record)+'\n')
        tmp.replace(state)
        time.sleep(1)


if __name__ == '__main__':
    main()
