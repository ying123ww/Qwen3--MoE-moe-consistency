"""Persistent salloc client with the user's exact resource options and an srun payload."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
from zoneinfo import ZoneInfo

import residual_rms_performance as shared


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if not re.fullmatch(r'remaining-profile-[a-z0-9-]+',args.run_id):
        parser.error('Use a fresh remaining-profile- ID')
    queue=shared.PROJECT/'.a-tmp'/(args.run_id+'-queue')
    queue.mkdir(exist_ok=False)
    prior=shared.PREP/'gpu_runs/rms-perf-hgx03-20261008-02/exact-off/manifest.json'
    sources=list(json.loads(prior.read_text())['source_sha256'])
    sources += [str(shared.PREP/'gpu_tools'/name) for name in (
        'inference_benchmark.py','profiling_annotations.py','residual_rms_performance.py',
        'resource_preflight.py','summarize_matmul.py','summarize_remaining_inference.py',
        'run_remaining_inference_profile.py','queue_remaining_profile.py')]
    digests={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}
    expected=queue/'expected-sources.json'
    shared.save(expected,digests)
    snapshot=queue/'source-snapshots'
    for name in digests:
        path=Path(name)
        try:
            relative=path.relative_to(shared.PROJECT)
        except ValueError:
            relative=Path('external')/path.relative_to('/')
        target=snapshot/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    argv=['salloc','-p','a100','-A','shaozy-a100','-N','1','--ntasks-per-node=1',
          '--cpus-per-task=64','--mem=1T','--gres=gpu:8','-x','ai_hgx_04','-t','2-00:00:00',
          'srun','-N','1','-n','1','-c','64','--kill-on-bad-exit=1',
          '--chdir='+str(shared.PROJECT),str(shared.PYTHON),
          str(shared.PREP/'gpu_tools/run_remaining_inference_profile.py'),
          '--run-id',args.run_id,'--expected-sources',str(expected)]
    state=dict(status='submitting',queue_pid=os.getpid(),queue_pgid=os.getpgrp(),
               submitted_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),argv=argv,
               report=str(shared.PREP/'gpu_runs'/args.run_id/'REPORT.md'))
    shared.save(queue/'queue-state.json',state)
    process=None
    def stop(signum,frame):
        state['status']='stopping'
        shared.save(queue/'queue-state.json',state)
        if process is not None: process.terminate()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    with (queue/'allocation.log').open('w') as log:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,start_new_session=True)
        state.update(client_pid=process.pid,client_pgid=process.pid)
        shared.save(queue/'queue-state.json',state)
        for line in process.stdout:
            log.write(line)
            log.flush()
            match=re.search(r'(?:Pending job allocation|Granted job allocation)\s+(\d+)',line)
            if match:
                state['job_id']=int(match.group(1))
                state['status']='allocated' if 'Granted' in line else 'queued'
                shared.save(queue/'queue-state.json',state)
            if line.startswith('REMAINING_PROFILE_COMPLETE'):
                state['status']='report-ready'
                shared.save(queue/'queue-state.json',state)
        code=process.wait()
    state.update(status='complete' if code==0 else 'failed',returncode=code,
                 finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    shared.save(queue/'queue-state.json',state)


if __name__=='__main__': main()
