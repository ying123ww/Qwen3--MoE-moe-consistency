"""Serial exchange correctness gates followed by bounded real-step timing runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/public/home/yingxy2024/moe-consistency/slime')
BASE = ROOT / 'alignment_a_preparation/gpu_runs/autopack-gpu-compare-20260914'
PYTHON = '/public/home/yingxy2024/.conda/envs/slime/bin/python'
STAGES = [('gate', 'routepack'), ('gate', 'random'), ('timing', 'native'),
          ('timing', 'random'), ('timing', 'routepack')]


ATTEMPT = '01'

def command_for(mode, method):
    name = f'{mode}_{method}-{ATTEMPT}'
    run = BASE / 'runs' / name
    profile = Path(f'/tmp/ap-exchange-{mode}-{method}-20260914-{ATTEMPT}')
    cmd = [PYTHON, str(ROOT/'alignment_a_preparation/gpu_tools/autopack_b1_hgx05.py'),
        '--rollout','/tmp/ap-real-capture-20260914-02/capture/rollout.pt',
        '--output',str(run),'--cuda-home','/tmp/ap-cuda-20260914',
        '--global-batch-size','256','--rollout-batch-size','32','--n-samples-per-prompt','8',
        '--expected-samples','256','--max-tokens-per-gpu','20480','--balance-data',
        '--rollout-max-response-len','8192','--rollout-max-context-len','16384',
        '--profile-directory',str(profile),'--schedule-manifest',str(BASE/'plans'/method/'runtime.json'),
        '--execution-mode',mode,'--warmups','2','--repeats','5']
    return run, profile, cmd


def run_stage(mode, method):
    import psutil
    run, profile, cmd = command_for(mode, method)
    if run.exists() or profile.exists():
        raise FileExistsError(f'Run/profile already exists: {run}, {profile}')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU process exists; refusing to overlap')
    started, reason = time.monotonic(), None
    with run.with_suffix('.log').open('x') as log, run.with_suffix('.memory.jsonl').open('x') as track:
        child = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owner = psutil.Process(child.pid)
        while child.poll() is None:
            procs = [owner] + owner.children(recursive=True)
            rss = 0
            for proc in procs:
                try: rss += proc.memory_info().rss
                except psutil.NoSuchProcess: pass
            row = dict(time=time.time(),elapsed_s=time.monotonic()-started,pid=child.pid,
                rss_gib=rss/2**30,available_gib=psutil.virtual_memory().available/2**30,
                tmp_free_gib=psutil.disk_usage('/tmp').free/2**30,
                result_free_gib=psutil.disk_usage(BASE).free/2**30)
            track.write(json.dumps(row)+'\n'); track.flush()
            if row['rss_gib'] > 1024: reason='owned_rss_above_1TiB'
            elif row['available_gib'] < 128: reason='system_available_below_128GiB'
            elif row['tmp_free_gib'] < 10: reason='tmp_free_below_10GiB'
            elif row['result_free_gib'] < 1: reason='result_free_below_1GiB'
            elif row['elapsed_s'] > 5400: reason='walltime_above_5400s'
            if reason:
                for proc in reversed(procs):
                    try: proc.terminate()
                    except psutil.NoSuchProcess: pass
                _, alive = psutil.wait_procs(procs,timeout=20)
                for proc in alive:
                    try: proc.kill()
                    except psutil.NoSuchProcess: pass
                break
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: pass
        rc = child.wait()
    run.mkdir(exist_ok=True)
    status=dict(mode=mode,method=method,returncode=rc,stop_reason=reason,
                elapsed_s=time.monotonic()-started,profile=str(profile),command=cmd)
    (run/'supervisor-exit.json').write_text(json.dumps(status,indent=2)+'\n')
    if rc or reason:
        raise RuntimeError(f'{mode}/{method} failed: rc={rc}, reason={reason}; inspect {run}.log')
    if mode == 'gate':
        ranks=[]
        for rank in range(8):
            gate=json.loads((profile/f'rank-{rank}/gate.json').read_text())
            schedule=json.loads((profile/f'rank-{rank}/schedule.json').read_text())
            if gate['rank'] != rank or not gate['passed'] or gate['optimizer_steps'] != 0:
                raise RuntimeError(f'Rank {rank} correctness gate failed')
            ranks.append(dict(rank=rank,passed=True,slots=len(schedule['micro_batch_indices']),
                response_tokens=sum(x['response_tokens'] for x in gate['logprobs']),
                mismatched_route_rows=sum(x['mismatched_rows'] for x in gate['observations'])))
        summary=dict(passed=True,method=method,rank_summaries=ranks,
                     scope='Exchanged schedule, two forward-only passes; no gradient equality claim')
        (run/'gate-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    else:
        result=json.loads((profile/'timing.json').read_text())
        (run/'timing.json').write_text(json.dumps(result,indent=2)+'\n')
        if not result['passed']:
            raise RuntimeError(f'{method} weight/gradient validity failed')
    # Let only our Ray cleanup finish before submitting the next stage.
    for attempt in range(12):
        if not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            return status
        time.sleep(5)
    raise RuntimeError('GPU processes remain after completed stage; refusing next stage')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume-timing',action='store_true')
    p.add_argument('--attempt',default='01')
    args=p.parse_args()
    global ATTEMPT
    if not args.attempt.isdigit():
        raise ValueError('Attempt must be numeric')
    ATTEMPT=args.attempt
    stages=STAGES
    if args.resume_timing:
        for method in ('routepack','random'):
            gate=json.loads((BASE/'runs'/f'gate_{method}-01'/'gate-summary.json').read_text())
            ranks=gate['rank_summaries']
            if not gate['passed'] or len(ranks)!=8 or {x['rank'] for x in ranks}!=set(range(8)) or not all(x['passed'] and x['mismatched_route_rows']==0 for x in ranks):
                raise RuntimeError('Saved exchanged-schedule gates do not pass')
        stages=[('timing', method) for method in ('native','random','routepack')]
    suffix=f'-timing-{ATTEMPT}' if args.resume_timing else ''
    if socket.gethostname().split('.')[0] != 'ai_hgx_05':
        raise RuntimeError('ai_hgx_05 only')
    if os.environ.get('PYTORCH_ALLOC_CONF'):
        raise RuntimeError('Conflicting allocator environment')
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    (BASE/'runs').mkdir(parents=True,exist_ok=True)
    plan=[dict(mode=m,method=n,command=command_for(m,n)[2]) for m,n in stages]
    (BASE/f'sequence-plan{suffix}.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.dry_run:
        return
    done=[]
    try:
        for mode,method in stages:
            print(json.dumps(dict(event='starting',mode=mode,method=method)),flush=True)
            done.append(run_stage(mode,method))
            print(json.dumps(dict(event='passed',mode=mode,method=method)),flush=True)
    except Exception as exc:
        (BASE/f'sequence-status{suffix}.json').write_text(json.dumps(dict(status='failed',completed=done,error=str(exc)),indent=2)+'\n')
        raise
    (BASE/f'sequence-status{suffix}.json').write_text(json.dumps(dict(status='passed',completed=done),indent=2)+'\n')


if __name__=='__main__':
    main()
