"""Gate-first seven-axis pilot, then two independent confirmations of selected winner."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import residual_rms_performance as guard
from resource_preflight import snapshot,validate
from run_fixed_capacity_pipeline import full_proof


def gate_passed(path,variant):
    folder=path/variant
    full_proof(folder)
    proofs=[json.loads(x.read_text()) for x in (folder/'graph-evidence').glob('p12-capture-size-*-rank-*.json')]
    if {(x['rank'],x['size']) for x in proofs}!={(r,s) for r in range(8) for s in (2,4)}:raise RuntimeError('Capture gate incomplete')
    runtime=[json.loads(x.read_text()) for x in (folder/'graph-evidence').glob('runtime-gate-rank-*.json')]
    if len(runtime)!=8 or any(x['counts'].get('eager_decode',0) or x['counts'].get('full_replays',0)!=64 for x in runtime):raise RuntimeError('Runtime gate failed')
    if variant!='reference':
        comparison=json.loads((path/f'compare-{variant}.json').read_text())
        if not comparison['model_exact'] or not comparison['batch_exact']:raise RuntimeError('Numerical gate failed')
    return folder


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gate1',type=Path,required=True);p.add_argument('--gate2',type=Path,required=True);p.add_argument('--wait-pid',type=int);args=p.parse_args()
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    try:
        guard.save(run/'state.json',dict(status='queued',waiting_for=args.wait_pid,pid=os.getpid(),time=time.time()))
        if args.wait_pid:
            while Path(f'/proc/{args.wait_pid}').exists():
                command=Path(f'/proc/{args.wait_pid}/cmdline').read_bytes()
                if b'run_p12_model_gates.py' not in command:break
                if guard.STOP:raise RuntimeError('Stop requested')
                time.sleep(5)
        state=json.loads((args.gate2/'state.json').read_text())
        if state['status']!='complete':raise RuntimeError('All requested ablation gates must finish before timing; inspect gate failure')
        variants=['reference','dense','rms-off','dense-rms-off','dense-qk','dense-router','dense-qk-router']
        gates={v:str(gate_passed(args.gate1 if v in ('reference','dense') else args.gate2,v)) for v in variants}
        lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if guard.gpu_pids():raise RuntimeError('Other GPU task active')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        folder=Path(__file__).parent
        paths=sorted(folder.glob('*.py'))+list(guard.PREP.glob('gpu_tools/full_graph_ab_*.py'))
        hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
        env=guard.environment(run);env['PYTHONPATH']=str(folder)+':'+str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env['FLASHINFER_WORKSPACE_BASE']=str(Path('/dev/shm')/run.name)
        nccl=guard.ENVROOT/'lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2';env['SGLANG_NCCL_SO_PATH']=str(nccl);env['LD_LIBRARY_PATH']=str(nccl.parent)+':'+env.get('LD_LIBRARY_PATH','')
        guard.save(run/'resources.json',resources);guard.save(run/'manifest.json',dict(gates=gates,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=hashes,protocol=dict(warmup=3,formal=20,output_tokens=1024,route_return_timed=True,pilot=variants,confirmation='Two more paired launches of reference and pilot winner; reverse order in round2'),restore_keepalive=False));guard.p2p_preflight(run)
        records=[]
        def execute(variant,round_id):
            label=f'round-{round_id}-{variant}';output=run/label;output.mkdir()
            worker_env=dict(env);worker_env['SLIME_QWEN3_P12_VARIANT']=variant
            guard.command(run,label,[str(guard.PYTHON),str(folder/'p12_benchmark.py'),'--variant',variant,'--mode','C','--run',str(output)],worker_env,groups[0]/'memory.usage_in_bytes',timeout=2400)
            full_proof(output)
            measured=[json.loads(line) for line in (output/'measurements.jsonl').read_text().splitlines()]
            if len(measured)!=23 or sum(not x['warmup'] for x in measured)!=20:raise RuntimeError('Incomplete generation measurement')
            if any(x['output_tokens']!=1024 or any(c!=0 for c in x['cached_tokens']) for x in measured):raise RuntimeError('Workload/cache protocol drift')
            runtime=[json.loads(x.read_text()) for x in (output/'graph-evidence').glob('runtime-formal-rank-*.json')]
            if len(runtime)!=8 or any(x['counts'].get('eager_decode',0) or x['counts'].get('full_replays',0)!=2560 for x in runtime):raise RuntimeError('Formal decode replay/fallback gate failed')
            summary=json.loads((output/'summary.json').read_text());row=dict(round=round_id,variant=variant,output=str(output),summary=summary)
            records.append(row);guard.save(run/'progress.json',records);return row
        pilot={v:execute(v,1) for v in variants}
        winner=min((v for v in variants if v!='reference'),key=lambda v:pilot[v]['summary']['median_s'])
        guard.save(run/'selection.json',dict(winner=winner,pilot_selection_only=True))
        execute(winner,2);execute('reference',2)
        execute('reference',3);execute(winner,3)
        paired=[]
        for r in (1,2,3):
            ref=next(x['summary']['median_s'] for x in records if x['round']==r and x['variant']=='reference')
            cand=next(x['summary']['median_s'] for x in records if x['round']==r and x['variant']==winner)
            paired.append(dict(round=r,reference_s=ref,winner_s=cand,change_percent=(cand/ref-1)*100))
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        guard.save(run/'results.json',dict(complete=True,records=records,winner=winner,paired=paired,median_paired_change_percent=statistics.median(x['change_percent'] for x in paired),scope='Whole generate includes route return; all modes numerical-gated. Winner selected in pilot; two independent confirmation pairs, not seven fully repeated axes.'))
        guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:guard.stop_worker();print('P12_TIMING_REMAINING_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':main()
