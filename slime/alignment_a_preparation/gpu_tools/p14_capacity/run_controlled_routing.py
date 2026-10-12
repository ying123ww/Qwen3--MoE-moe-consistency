"""Guarded8-GPU normal EP4 controlled graph routing complement."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import residual_rms_performance as guard
from resource_preflight import snapshot,validate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    try:
        if guard.gpu_pids():raise RuntimeError('AnotherGPUtask active')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,128,20)
        group=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(group)!=1:raise RuntimeError('Memory cgroup missing')
        env=guard.environment(run);folder=Path(__file__).parent.resolve();env['PYTHONPATH']=str(folder)+':'+env.get('PYTHONPATH','')
        nccl=guard.ENVROOT/'lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2';env['SGLANG_NCCL_SO_PATH']=str(nccl);env['LD_LIBRARY_PATH']=str(nccl.parent)+':'+env.get('LD_LIBRARY_PATH','')
        sources=[Path(__file__),folder/'controlled_routing_probe.py'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        guard.save(run/'manifest.json',dict(source_sha256=hashes,scope='Synthetic identityexpert controllednormalEP4routing complementsmodelgate',restore_keepalive=False));guard.p2p_preflight(run)
        command=[str(guard.PYTHON),'-m','torch.distributed.run','--standalone','--nproc-per-node','8',str(folder/'controlled_routing_probe.py'),'--run',str(run)]
        guard.command(run,'controlled',command,env,group[0]/'memory.usage_in_bytes',timeout=600)
        records=[json.loads(p.read_text()) for p in run.glob('controlled-rank-*.json')]
        if {x['rank'] for x in records}!=set(range(8)) or any(not x['complete'] or len(x['cases'])!=5 for x in records):raise RuntimeError('Incomplete controlled routing ranks')
        for record in records:
            for case in record['cases']:
                for cap in ('8','16'):
                    row=case['capacities'][cap]
                    if row['output_bit_mismatches'] or row['real_receive_rows']>int(cap):raise RuntimeError('Controlledgraph bounds/bit gate failed')
        if not any(row['capacities']['8']['zero_receive'] for x in records for row in x['cases']):raise RuntimeError('Zero receive not actuallytested')
        if not any(row['logical_send_rows']==0 for x in records for row in x['cases']):raise RuntimeError('Idle sender not actuallytested')
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        guard.save(run/'results.json',dict(complete=True,records=records,scope='Actual normalfixed8/16graphdispatch/combine all-rank/skew/balance/spread/idle,identityexperts; not fullmodelproof'))
        guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:guard.stop_worker();print('P14_CONTROLLED_GPU_PIDS='+guard.gpu_pids(),flush=True)


if __name__=='__main__':main()
