"""Lossless recovery from PyTorch's complete JSON, then single-pass analysis."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import residual_rms_performance as guard

RUN = guard.PREP / 'gpu_runs/remaining-profile-hgx03-qk-20261011-01'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        while data:=handle.read(8*1024*1024):h.update(data)
    return h.hexdigest()


def analyze(item):
    mode,rank,trace=item
    from summarize_remaining_inference import summarize
    from summarize_qk_detail import summarize as detail
    with gzip.open(trace,'rt') as handle:data=json.load(handle)
    basic=summarize(data['traceEvents']);basic['trace']=str(trace)
    basic['distributed_rank']=data.get('distributedInfo',{}).get('rank')
    basic_path=RUN/f'profile-{mode}'/f'rank-{rank}-basic.json'
    basic_path.write_text(json.dumps(basic,indent=2)+'\n')
    detailed=detail(data['traceEvents']);detailed.update(trace=str(trace),distributed_info=data.get('distributedInfo'))
    output=RUN/f'profile-{mode}'/'detail-summary'/f'rank-{rank}.json'
    output.write_text(json.dumps(detailed,indent=2)+'\n')
    return dict(mode=mode,rank=rank,basic=basic,detail_path=str(output),categories=detailed['gpu_event_categories'])


def main():
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Wrong host')
    state=RUN/'recovery-state.json'
    def save(status,**extra):guard.save(state,dict(status=status,pid=os.getpid(),pgid=os.getpgrp(),time=time.time(),**extra))
    try:
        if guard.gpu_pids():raise RuntimeError('GPU task exists before recovery analysis')
        save('validating-raw-json')
        previously_recovered=RUN/'trace-recovery.json'
        already_complete=previously_recovered.exists() and json.loads(previously_recovered.read_text()).get('complete')
        integrity=[] if already_complete else json.loads((RUN/'trace-integrity.json').read_text())
        raw_files=list((guard.PROJECT/'.a-tmp'/RUN.name).glob('*.json'))
        recovered=[]
        for item in integrity:
            if item['valid']:continue
            damaged=Path(item['path'])
            if not damaged.is_absolute():damaged=guard.PROJECT/damaged
            with gzip.open(damaged,'rb') as handle:prefix=handle.read(1024*1024)
            matches=[]
            for raw in raw_files:
                with raw.open('rb') as handle:
                    if handle.read(len(prefix))==prefix:matches.append(raw)
            if len(matches)!=1:raise RuntimeError('Cannot uniquely pair the raw trace: '+str(damaged))
            raw=matches[0]
            with raw.open() as handle:data=json.load(handle)
            if not data.get('traceEvents') or data['traceEvents'][-1].get('name')!='Record Window End':
                raise RuntimeError('Raw JSON has no complete record window')
            event_count=len(data['traceEvents']);del data
            old_sha=digest(damaged);raw_sha=digest(raw)
            backup=damaged.with_suffix('.gz.truncated-original')
            temporary=damaged.with_suffix('.gz.recovered-new')
            with raw.open('rb') as source,gzip.open(temporary,'wb',compresslevel=9) as target:
                shutil.copyfileobj(source,target,length=8*1024*1024)
            h=hashlib.sha256()
            with gzip.open(temporary,'rb') as handle:
                while block:=handle.read(8*1024*1024):h.update(block)
            if h.hexdigest()!=raw_sha:raise RuntimeError('Recovered compressed bytes differ from raw JSON')
            damaged.rename(backup);temporary.rename(damaged)
            recovered.append(dict(trace=str(damaged),original_damaged=str(backup),original_sha256=old_sha,
                                  raw_json=str(raw),raw_sha256=raw_sha,recovered_sha256=digest(damaged),events=event_count))
            guard.save(RUN/'trace-recovery.json',dict(complete=False,rows=recovered))
            print('RECOVERED '+str(damaged),flush=True)
        if not already_complete:
            guard.save(RUN/'trace-recovery.json',dict(complete=True,rows=recovered,
                       method='Complete original JSON validated, paired by identical 1 MiB prefix, recompressed with exact full raw SHA256 equality; no events synthesized'))
        import re
        jobs=[]
        for mode in ('baseline','candidate'):
            (RUN/f'profile-{mode}'/'detail-summary').mkdir(exist_ok=True)
            for trace in sorted((RUN/f'profile-{mode}').glob('profiles/**/*.json.gz')):
                jobs.append((mode,int(re.search(r'-TP-(\d+)-',trace.name).group(1)),trace))
        if len(jobs)!=16:raise RuntimeError('Expected 16 worker traces')
        results=[]
        pending=[]
        cache_by_mode={}
        for mode in ('baseline','candidate'):
            path=RUN/f'profile-{mode}'/'component-summary.json'
            cache_by_mode[mode]=json.loads(path.read_text()) if path.exists() else []
        for mode,rank,trace in jobs:
            cached=next((row for row in cache_by_mode[mode]
                         if row.get('distributed_rank',row['gpu_devices'][0])==rank),None)
            if cached is None and mode=='candidate':
                debug=RUN/f'profile-{mode}'/f'rank-{rank}-basic-debug.json'
                if debug.exists():cached=json.loads(debug.read_text())
            detail_path=RUN/f'profile-{mode}'/'detail-summary'/f'rank-{rank}.json'
            if cached and not cached['unlinked_persistent_matmul'] and detail_path.exists():
                if cached['gpu_devices'] != [rank]:raise RuntimeError('Cached GPU/rank mismatch')
                cached.setdefault('distributed_rank',rank)
                cached.setdefault('trace',str(trace))
                cached.setdefault('recovered_matrix_links',[])
                categories=json.loads(detail_path.read_text())['gpu_event_categories']
                results.append(dict(mode=mode,rank=rank,basic=cached,detail_path=str(detail_path),categories=categories,cached=True))
            else:pending.append((mode,rank,trace))
        save('analyzing-complete-traces',completed=len(results))
        with ProcessPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(analyze,item) for item in pending]):
                results.append(future.result());save('analyzing-complete-traces',completed=len(results))
                print(json.dumps(dict(mode=results[-1]['mode'],rank=results[-1]['rank'],categories=results[-1]['categories'])),flush=True)
        profiles={mode:[x['basic'] for x in sorted(results,key=lambda x:x['rank']) if x['mode']==mode] for mode in ('baseline','candidate')}
        for mode,rows in profiles.items():
            if len(rows)!=8 or any(x['unlinked_persistent_matmul'] for x in rows):raise RuntimeError('Incomplete rank/matmul attribution')
            guard.save(RUN/f'profile-{mode}'/'component-summary.json',rows)
        from run_hgx03_qk_profile import report
        report(RUN,profiles)
        guard.save(RUN/'detail-index.json',dict(complete=True,rows=[{k:v for k,v in x.items() if k!='basic'} for x in results]))
        guard.save(RUN/'state.json',dict(status='complete',driver_pid=os.getpid(),report=str(RUN/'REPORT.md'),time=time.time(),
                 recovered_from_complete_raw_json=True,original_sampling_driver_pid=123861))
        argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/qk_microbenchmark.py'),'--evidence',str(RUN/'profile-candidate/qk-evidence'),
              '--run-id','qk-micro-hgx03-20261011-01']
        log=RUN/'qk-micro-driver.log'
        with log.open('x') as handle:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            save('running-micro-gate',child_pid=child.pid,child_pgid=child.pid,log=str(log))
            code=child.wait()
        if code:raise RuntimeError('Micro gate/benchmark failed; see '+str(log))
        save('complete',micro_run='qk-micro-hgx03-20261011-01')
        print('QK_RECOVERY_DETAIL_MICRO_COMPLETE',flush=True)
    except BaseException as exc:
        save('failed',error=str(exc));raise


if __name__=='__main__':main()
