"""Persistent sequential follow-up: profile analysis, then standalone exact gate."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

import residual_rms_performance as guard

RUN = guard.PREP / 'gpu_runs/remaining-profile-hgx03-qk-20261011-01'


def analyze(item):
    mode, trace, output = item
    from summarize_qk_detail import summarize
    with gzip.open(trace, 'rt') as handle:
        data = json.load(handle)
    result = summarize(data['traceEvents'])
    result.update(trace=str(trace), distributed_info=data.get('distributedInfo'))
    output.write_text(json.dumps(result, indent=2) + '\n')
    return dict(mode=mode, output=str(output), gpu_event_categories=result['gpu_event_categories'])


def main():
    if socket.gethostname() != 'ai_hgx_03':
        raise RuntimeError('Wrong follow-up host')
    state = RUN / 'followup-state.json'
    def save(status, **extra):
        guard.save(state, dict(status=status,pid=os.getpid(),pgid=os.getpgrp(),time=time.time(),**extra))
    try:
        save('waiting-for-profile-completion')
        while True:
            status = json.loads((RUN / 'state.json').read_text())
            if status['status'] == 'failed':
                raise RuntimeError('Profile failed: ' + status.get('error',''))
            if status['status'] == 'complete':
                driver = status['driver_pid']
                p = Path(f'/proc/{driver}/stat')
                if not p.exists() or p.read_text().split()[2] == 'Z':
                    break
            time.sleep(5)
        save('analyzing-detail',completed=0)
        jobs=[]
        for mode in ('baseline','candidate'):
            directory=RUN/f'profile-{mode}'/'detail-summary'
            directory.mkdir(exist_ok=False)
            for trace in sorted((RUN/f'profile-{mode}').glob('profiles/**/*.json.gz')):
                rank=int(re.search(r'-TP-(\d+)-',trace.name).group(1))
                jobs.append((mode,trace,directory/f'rank-{rank}.json'))
        if len(jobs)!=16:
            raise RuntimeError('Expected sixteen worker traces')
        done=[]
        with ProcessPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(analyze,item) for item in jobs]):
                done.append(future.result())
                save('analyzing-detail',completed=len(done))
                print(json.dumps(done[-1]),flush=True)
        guard.save(RUN/'detail-index.json',dict(complete=True,rows=done))
        argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/qk_microbenchmark.py'),
              '--evidence',str(RUN/'profile-candidate/qk-evidence'),'--run-id','qk-micro-hgx03-20261011-01']
        log=RUN/'qk-micro-driver.log'
        with log.open('x') as handle:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            save('running-micro-gate',child_pid=child.pid,child_pgid=child.pid,log=str(log))
            code=child.wait()
        if code:
            raise RuntimeError('Micro gate/benchmark failed; see ' + str(log))
        save('complete',detail_index=str(RUN/'detail-index.json'),micro_run='qk-micro-hgx03-20261011-01')
        print('QK_DETAIL_AND_MICRO_COMPLETE',flush=True)
    except BaseException as exc:
        save('failed',error=str(exc))
        raise


if __name__=='__main__':
    main()
