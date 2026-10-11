"""Wait for whole timing, verify the joint trace, then investigate dense tiles."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import residual_rms_performance as guard
from resource_preflight import snapshot,validate


def analyze(item):
    rank,trace,output=item
    from summarize_remaining_inference import summarize
    from summarize_qk_detail import summarize as detail
    with gzip.open(trace,'rt') as handle:data=json.load(handle)
    basic=summarize(data['traceEvents']);basic.update(distributed_rank=rank,trace=str(trace))
    extra=detail(data['traceEvents']);extra.update(trace=str(trace),distributed_info=data.get('distributedInfo'))
    guard.save(output/f'rank-{rank}-detail.json',extra)
    return basic


def main():
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Wrong follow-up host')
    run=guard.PREP/'gpu_runs/qk-joint-profile-hgx03-20261011-01';run.mkdir(exist_ok=False)
    parent=guard.PREP/'gpu_runs/qk-model-perf-hgx03-20261011-01'
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    def state(status,**extra):guard.save(run/'state.json',dict(status=status,pid=os.getpid(),pgid=os.getpgrp(),time=time.time(),**extra))
    try:
        state('waiting-for-180-batch-timing')
        while True:
            s=json.loads((parent/'state.json').read_text())
            if s['status']=='failed':raise RuntimeError('Full model/performance stage failed: '+s.get('error',''))
            p=Path(f"/proc/{s.get('pid',s.get('driver_pid',0))}/stat")
            if s['status']=='complete' and (not p.exists() or p.read_text().split()[2]=='Z'):break
            if guard.STOP:raise RuntimeError('Stop requested')
            time.sleep(5)
        lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if guard.gpu_pids():raise RuntimeError('Other GPU task exists before verification profile')
        resource=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resource,512,20)
        groups=[Path(x['path']) for x in resource['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory group missing')
        guard.save(run/'resources.json',resource);env=guard.environment(run)
        env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']='1';env['SGLANG_PROFILE_WITH_STACK']='0'
        output=run/'profile-joint';output.mkdir()
        env['SLIME_QK_PROFILE_DETAIL_DIR']=str(output/'qk-evidence')
        sources=[Path(__file__)]+[guard.PREP/'gpu_tools'/x for x in ('profile_joint_and_capture_dense.py','profile_export_guard.py','qk_runtime_hooks.py','qk_norm_prototype.py','qk_profile_annotations.py','dense_projection_capture.py','dense_tile_probe.py')]
        hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
        guard.save(run/'manifest.json',dict(source_sha256=hashes,parent_timing=str(parent),restore_keepalive=False,protocol=dict(warmup=3,profile_batches=1,response_tokens=128)))
        argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/profile_joint_and_capture_dense.py'),'--run',str(output),
              '--variant','aligned','--residual-rms-mode','candidate','--router-tile-mode','off','--warmup','3','--steps','1',
              '--profile','--profile-shapes','--profile-attribution','--response-tokens','128']
        guard.command(run,'profile-joint',argv,env,groups[0]/'memory.usage_in_bytes')
        evidence=list((output/'dense-evidence').glob('rank-*-layer-*.pt'))
        if len(evidence)!=8:raise RuntimeError('Missing real QKV/O operands')
        log=(run/'profile-joint.log').read_text()
        if 'COMPLETE_PROFILE_EXPORTS_VERIFIED 8' not in log:raise RuntimeError('Trace completion not verified')
        import re
        traces=sorted(output.glob('profiles/**/*.json.gz'))
        if len(traces)!=8:raise RuntimeError('Missing joint worker traces')
        state('analyzing-joint-profile',completed=0)
        details=output/'detail-summary';details.mkdir()
        rows=[]
        with ProcessPoolExecutor(max_workers=2) as pool:
            jobs=[(int(re.search(r'-TP-(\d+)-',p.name).group(1)),p,details) for p in traces]
            for future in as_completed([pool.submit(analyze,x) for x in jobs]):
                rows.append(future.result());state('analyzing-joint-profile',completed=len(rows))
        rows.sort(key=lambda x:x['distributed_rank'])
        if any(x['unlinked_persistent_matmul'] for x in rows):raise RuntimeError('Unlinked joint matmul')
        guard.save(output/'component-summary.json',rows)
        reference=guard.PREP/'gpu_runs/remaining-profile-hgx03-qk-20261011-01/profile-candidate/component-summary.json'
        refs=json.loads(reference.read_text());pairs=[]
        for a,b in zip(refs,rows):
            pairs.append(dict(rank=b['distributed_rank'],rms_events=a['gpu_events'],joint_events=b['gpu_events'],
                              event_reduction=a['gpu_events']-b['gpu_events'],rms_union_s=a['gpu_union_s'],joint_union_s=b['gpu_union_s']))
        guard.save(run/'event-comparison.json',dict(complete=True,rows=pairs))
        # Standalone probe owns the same lock itself; release after all analysis.
        fcntl.flock(lock,fcntl.LOCK_UN)
        argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/dense_tile_probe.py'),'--evidence',str(output/'dense-evidence'),
              '--run-id','dense-tiles-hgx03-20261011-01']
        with (run/'dense-probe.log').open('x') as handle:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            state('running-dense-probe',child_pid=child.pid,child_pgid=child.pid)
            code=child.wait()
        if code:raise RuntimeError('Dense tile probe failed; see dense-probe.log')
        state('complete',dense_run='dense-tiles-hgx03-20261011-01')
        print('JOINT_PROFILE_DENSE_PROBE_COMPLETE',flush=True)
    except BaseException as exc:
        state('failed',error=str(exc));raise
    finally:guard.stop_worker()


if __name__=='__main__':main()
