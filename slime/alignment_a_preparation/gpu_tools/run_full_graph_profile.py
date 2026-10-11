"""P11 source-marked full-graph profile; requires the completed P10 protocol first."""
import argparse
import fcntl
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
from run_fixed_capacity_pipeline import full_proof

NSYS=Path('/public/home/yingxy2024/.conda/envs/slime/nsight-compute-2025.2.1/host/target-linux-x64/nsys')


def validate_profile(output,mode):
    full_proof(output)
    completed=json.loads((output/'complete.json').read_text())
    assert completed['complete'] and completed['steps']==32
    records=[json.loads(p.read_text()) for p in (output/'profile-evidence').glob('components-rank-*.json')]
    assert {p['rank'] for p in records}==set(range(8))
    expected={'model_forward','qkv','output_projection','attention_core','qk_rope','router',
              'router_select','experts','deepep_dispatch','deepep_combine','logits'}
    for row in records:
        assert row['steps']==32 and row['mode']==mode
        assert expected<={x['component'] for x in row['components']}
        assert all(x['last_replay_gpu_ms']>=0 for x in row['components'])
        for component in ('qkv','output_projection','attention_core','router','experts','deepep_dispatch','deepep_combine'):
            assert {x['layer'] for x in row['components'] if x['component']==component}==set(range(48))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('smoke','nsys'),required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--p10-run',type=Path,required=True)
    parser.add_argument('--nsys-path',type=Path,default=NSYS)
    args=parser.parse_args()
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    if Path(args.run_id).name!=args.run_id:raise ValueError('Plain run ID required')
    p10=json.loads((args.p10_run/'results.json').read_text())
    assert p10['independent_engines']==9 and p10['formal_batches']==180
    assert json.loads((args.p10_run/'state.json').read_text())['status']=='complete'
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if guard.gpu_pids():raise RuntimeError('Another GPU task is active')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        env=guard.environment(run);env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env['FLASHINFER_WORKSPACE_BASE']=str(Path('/dev/shm')/run.name)
        nccl=guard.ENVROOT/'lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2'
        env['SGLANG_NCCL_SO_PATH']=str(nccl)
        env['LD_LIBRARY_PATH']=str(nccl.parent)+':'+env.get('LD_LIBRARY_PATH','')
        paths=[Path(__file__),*guard.PREP.glob('gpu_tools/full_graph_profile_*.py'),
               *guard.PREP.glob('gpu_tools/full_graph_ab_*.py'),guard.PREP/'gpu_tools/fixed_capacity_dispatch_hooks.py']
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        guard.save(run/'manifest.json',dict(phase=args.phase,p10_run=str(args.p10_run),source_sha256=hashes,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            modes=['C'] if args.phase=='smoke' else ['A','C'],steps=32,restore_keepalive=False,
            timing_use='Nsight/event instrumentation for attribution; never a normal latency benchmark'))
        guard.save(run/'resources.json',resources);guard.p2p_preflight(run)
        results={}
        for mode in (('C',) if args.phase=='smoke' else ('A','C')):
            output=run/mode;output.mkdir()
            worker=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/full_graph_profile_benchmark.py'),
                    '--mode',mode,'--run',str(output)]
            if args.phase=='nsys':
                worker.append('--nsys');(output/'nsys').mkdir()
                worker=[str(args.nsys_path),'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                    '--cuda-event-trace=false',
                    '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=repeat:2',
                    '--kill=sigterm','--force-overwrite=true','--output='+str(output/'nsys/timeline'),*worker]
            guard.command(run,mode,worker,env,groups[0]/'memory.usage_in_bytes',timeout=2400)
            records=validate_profile(output,mode);results[mode]=records
            if args.phase=='nsys':
                reports=sorted((output/'nsys').glob('*.nsys-rep'))
                if len(reports)!=2:raise RuntimeError(f'Expected capture and steady reports, got {reports}')
                for index,report in enumerate(reports):
                    target=output/'nsys'/f'window-{index}.sqlite'
                    with (output/'nsys'/f'export-{index}.log').open('w') as log:
                        subprocess.run([str(args.nsys_path),'export','--type=sqlite','--force-overwrite=true',
                            '--output='+str(target),str(report)],env=env,stdout=log,stderr=subprocess.STDOUT,
                            check=True,timeout=600)
                guard.save(output/'nsys/reports.json',dict(reports=list(map(str,reports))))
            guard.save(run/'profile-progress.json',results)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        guard.save(run/'results.json',dict(complete=True,records=results,phase=args.phase))
        guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:guard.stop_worker();print('P11_REMAINING_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':main()
