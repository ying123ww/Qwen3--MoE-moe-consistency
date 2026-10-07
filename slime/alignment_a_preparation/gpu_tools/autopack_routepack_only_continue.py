"""Wait for the already running RoutePack gate, then time only RoutePack."""
import json
import os
from pathlib import Path
import subprocess
import time

import autopack_gpu_compare128_sequence as case


def main():
    import psutil
    sequence = case.sequence
    sequence.command_for = case.command_for
    os.environ['ROUTEPACK_CHECKPOINT_CPU_OFFLOAD'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    base = sequence.BASE
    gate = Path('/tmp/ap-exchange128-gate-routepack-20260914-01')
    gate_run = base / 'runs/gate_routepack-01'
    plan = dict(scope='User requested RoutePack only; Random cancelled',
                existing_gate_pid=69395, stages=['wait_existing_gate_routepack', 'timing_routepack'])
    (base/'routepack-only-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    started = time.monotonic()
    # Read actual gate artifacts; the stopped former supervisor is not evidence
    # that the still-running gate child failed.
    while True:
        files = [gate/f'rank-{i}/gate.json' for i in range(8)]
        alive = psutil.pid_exists(69395)
        if alive:
            proc = psutil.Process(69395)
            alive = proc.status() != psutil.STATUS_ZOMBIE
            if alive and 'autopack_b1_hgx05.py' not in ' '.join(proc.cmdline()):
                raise RuntimeError('Gate PID reused; inspect task ownership')
        if all(p.exists() for p in files) and not alive:
            break
        if not alive and not all(p.exists() for p in files):
            raise RuntimeError('Gate process exited without all eight result files')
        if time.monotonic()-started > 5400:
            raise TimeoutError('Existing gate did not finish within 5400 seconds')
        time.sleep(15)
    ranks=[]
    for i,path in enumerate(files):
        g=json.loads(path.read_text())
        s=json.loads((gate/f'rank-{i}/schedule.json').read_text())
        row=dict(rank=i,passed=g['passed'],slots=len(s['micro_batch_indices']),
                 response_tokens=sum(x['response_tokens'] for x in g['logprobs']),
                 mismatched_route_rows=sum(x['mismatched_rows'] for x in g['observations']))
        if g['rank'] != i or not g['passed'] or row['slots'] != 13 or row['mismatched_route_rows'] or g['optimizer_steps']:
            raise RuntimeError(f'RoutePack gate rank {i} failed acceptance')
        ranks.append(row)
    if sum(x['response_tokens'] for x in ranks) != 813356:
        raise RuntimeError('Response token coverage mismatch')
    (gate_run/'gate-summary.json').write_text(json.dumps(dict(passed=True,method='routepack',
        rank_summaries=ranks,scope='Exchanged schedule forward gate; Random cancelled'),indent=2)+'\n')
    for attempt in range(12):
        if not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            break
        time.sleep(5)
    else:
        raise RuntimeError('GPUs not empty after gate; do not overlap timing')
    result=sequence.run_stage('timing','routepack')
    (base/'routepack-only-status.json').write_text(json.dumps(dict(status='passed',timing=result,
        random='cancelled_by_user'),indent=2)+'\n')


if __name__=='__main__':
    main()
