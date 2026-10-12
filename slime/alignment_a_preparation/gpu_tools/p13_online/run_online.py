"""Own isolated Ray cluster for the real v89 full-graph online gate."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import tempfile
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import residual_rms_performance as guard
from resource_preflight import snapshot,validate,GIB
from prepare_online_config import prepare


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--timing-run',type=Path,required=True);args=parser.parse_args()
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    timing=json.loads((args.timing_run/'results.json').read_text())
    if not timing['complete'] or timing['median_paired_change_percent']>=0:raise RuntimeError('P12 confirmed winner unavailable')
    variant=timing['winner'];run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    head=None;client=None;submitted=False;other_active=False
    try:
        if guard.gpu_pids():raise RuntimeError('Other GPU task active')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        memory_file=groups[0]/'memory.usage_in_bytes'
        env=guard.environment(run);short_tmp=tempfile.mkdtemp(prefix='p13_',dir='/tmp')
        env.update({key:short_tmp for key in ('TMPDIR','TMP','TEMP')});guard.save(run/'short-temp.json',dict(path=short_tmp))
        folder=Path(__file__).parent.resolve();p12=guard.PREP/'gpu_tools'
        env['PYTHONPATH']=str(folder)+':'+str(p12)+':'+str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env.update(SLIME_P13_SERIAL_CHECKPOINT_READ='1',SLIME_P13_CHECKPOINT_LOCK=str(Path(short_tmp)/'checkpoint-reader.lock'),SLIME_P13_FULL_BOOTSTRAP='1',SLIME_P13_RUN_DIR=str(run),SGLANG_MEMORY_SAVER_CUDA_GRAPH='0',SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT='0',SLIME_QWEN3_RESIDUAL_RMS_MODE='candidate',SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH='1',SLIME_QWEN3_RESIDUAL_RMS_AUDIT='0',SLIME_QWEN3_ROUTER_TILE_MODE='off',SLIME_QWEN3_ROUTER_TILE_AUDIT='0',ENABLE_ROUTING_REPLAY='0',OPEN_TRAINING_INT4_FAKE_QAT_FLAG='0',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',CUDA_DEVICE_MAX_CONNECTIONS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NVTE_ALLOW_NONDETERMINISTIC_ALGO='0',TE_DISABLE_FA3='TRUE',TORCH_COMPILE_DISABLE='1')
        import yaml
        original=yaml.safe_load((guard.PREP/'source_snapshots/v89_runtime_env.yaml.txt').read_text())['env_vars']
        for key,value in original.items():
            if key.startswith(('MEGATRON_','SLIME_LAYERWISE_','SGLANG_TENSOR_DUMP_')):env[key]=str(value)
        for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env[key]=''
        env['NO_PROXY']=env['no_proxy']='*'
        nccl=guard.ENVROOT/'lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2';env['SGLANG_NCCL_SO_PATH']=str(nccl);env['LD_LIBRARY_PATH']=str(nccl.parent)+':'+env.get('LD_LIBRARY_PATH','')
        config=prepare(run,variant);argv=config['argv']
        guard.save(run/'prepared-online-config.json',config);guard.save(run/'argv.json',argv);guard.save(run/'resources.json',resources)
        source_paths=sorted(folder.glob('*.py'))+sorted(p12.glob('*.py'))
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
        guard.save(run/'manifest.json',dict(variant=variant,p12_run=str(args.timing_run),source_sha256=hashes,protocol='V89 lr1e-6 StatelessAdam/reset states,controlled0/1diagnosticreward,3realrollouts,full colocated tensor sync (training NCCL collection, Gloo/IPC delivery),1024targets perversion',restore_keepalive=False))
        cache_keys=('PATH','PYTHONPATH','CUDA_HOME','CUDACXX','CPATH','LD_LIBRARY_PATH','CUDNN_HOME','TMPDIR','TMP','TEMP','TVM_FFI_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','XDG_CACHE_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','PYTHONUNBUFFERED','PYTHONFAULTHANDLER','CUDA_DEVICE_MAX_CONNECTIONS','CUBLAS_WORKSPACE_CONFIG','NVTE_ALLOW_NONDETERMINISTIC_ALGO','TE_DISABLE_FA3','TORCH_COMPILE_DISABLE','NO_PROXY','no_proxy')
        runtime_env={key:str(value) for key,value in env.items() if key in cache_keys or key.startswith(('SLIME_','SGLANG_','MEGATRON_','NCCL_'))}
        for key in ('ENABLE_ROUTING_REPLAY','OPEN_TRAINING_INT4_FAKE_QAT_FLAG'):runtime_env[key]=env[key]
        guard.save(run/'runtime_env.json',dict(env_vars=runtime_env))
        ray_dir=Path(tempfile.mkdtemp(prefix='rp13_',dir='/tmp'))
        for port in (6391,8269):
            with socket.socket() as probe:probe.bind(('127.0.0.1',port))
        head_command=[str(guard.ENVROOT/'bin/ray'),'start','--head','--block','--node-ip-address','127.0.0.1','--port','6391','--dashboard-host','127.0.0.1','--dashboard-port','8269','--num-gpus','8','--num-cpus',str(resources['ray_cpus']),'--object-store-memory',str(4*GIB),'--disable-usage-stats','--temp-dir',str(ray_dir)]
        stream=(run/'ray-head.log').open('w');head=subprocess.Popen(head_command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        guard.save(run/'ray-head.json',dict(pid=head.pid,pgid=head.pid,command=head_command,temp_dir=str(ray_dir)))
        from ray.job_submission import JobSubmissionClient
        class BoundedClient(JobSubmissionClient):
            def _do_request(self,method,endpoint,**kwargs):
                kwargs.setdefault('timeout',(5,15));return super()._do_request(method,endpoint,**kwargs)
        deadline=time.monotonic()+180
        while True:
            if head.poll() is not None:raise RuntimeError('Owned Ray head exited during startup')
            try:
                client=BoundedClient('http://127.0.0.1:8269');client.list_jobs();break
            except Exception:
                if time.monotonic()>deadline:raise
                time.sleep(2)
        import shlex
        job_id=client.submit_job(entrypoint=shlex.join(argv),submission_id=run.name,runtime_env={'env_vars':runtime_env});submitted=True
        guard.save(run/'submission.json',dict(job_id=job_id,address='http://127.0.0.1:8269',head_pid=head.pid))
        started=time.monotonic()
        while True:
            status=str(client.get_job_status(job_id));usage=int(memory_file.read_text())
            memory_stat={k:int(v) for k,v in (line.split() for line in (groups[0]/'memory.stat').read_text().splitlines())}
            observation=dict(status=status,pid=os.getpid(),ray_job_id=job_id,head_pid=head.pid,memory_usage_bytes=usage,memory_stat=memory_stat,time=time.time())
            guard.save(run/'state.json',observation)
            with (run/'memory-observations.jsonl').open('a') as file:file.write(json.dumps(observation)+'\n')
            if usage>900*GIB or guard.STOP or time.monotonic()-started>10800:
                client.stop_job(job_id);raise RuntimeError('Owned online job stopped by memory/stop/time guard')
            if status.endswith('SUCCEEDED'):break
            if status.endswith(('FAILED','STOPPED')):raise RuntimeError('Real online job '+status)
            time.sleep(5)
        (run/'job.log').write_text(client.get_job_logs(job_id))
        subprocess.run([str(guard.PYTHON),str(folder/'validate_online_bits.py'),'--run',str(run)],env=env,check=True,timeout=300)
        updates=[json.loads(line) for path in (run/'training-update-evidence').glob('updates-rank-*.jsonl') for line in path.read_text().splitlines()]
        for rank in range(8):
            rows=[x for x in updates if x['rank']==rank and x['rollout_id'] in (0,1)]
            if {x['rollout_id'] for x in rows}!={0,1} or any(x['gradient_norm']<=0 or x['sampled_master_changed_elements']<=0 for x in rows):raise RuntimeError('Two real nonzero updates not proved on all ranks')
        comparisons=[json.loads((run/'replay-validation'/f'version-{v}.json').read_text()) for v in (0,1,2)]
        if not all(x['model_equal'] for x in comparisons) or not comparisons[2]['replacement_recovered']:raise RuntimeError('Updated replay/recapture/replacement recovery gate missing')
        lifecycle=[json.loads(line) for path in (run/'lifecycle').glob('lifecycle-rank-*.jsonl') for line in path.read_text().splitlines()]
        for rank in range(8):
            rows=[x for x in lifecycle if x['rank']==rank]
            if not any(x['event']=='replacement_negative' and x['old_replay_rejected'] for x in rows):raise RuntimeError('Storage replacement rejection absent')
            if not any(x['event']=='release' for x in rows) or not any(x['event']=='resume' for x in rows):raise RuntimeError('Real sleep/wake lifecycle absent')
        guard.save(run/'state.json',dict(status='online-logprob-complete',pid=os.getpid(),time=time.time(),remaining='Audit lifecycle logs, graph identities, full sync and perversion comparisons before marking P13 complete'))
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    except BaseException as exc:
        if client is not None and submitted:
            try:(run/'job.log').write_text(client.get_job_logs(run.name))
            except Exception:pass
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise
    finally:
        if client is not None and submitted:
            try:
                active=[x for x in client.list_jobs() if str(x.status).endswith(('RUNNING','PENDING'))]
                other_active=any(x.submission_id!=run.name for x in active)
                if any(x.submission_id==run.name for x in active):client.stop_job(run.name)
            except Exception:pass
        if head is not None and not other_active:
            import psutil
            try:
                owned=psutil.Process(head.pid);children=owned.children(recursive=True)
                for process in reversed(children):
                    try:process.terminate()
                    except psutil.NoSuchProcess:pass
                owned.terminate();_,alive=psutil.wait_procs(children+[owned],timeout=30)
                for process in alive:
                    try:process.kill()
                    except psutil.NoSuchProcess:pass
            except psutil.NoSuchProcess:pass
        guard.save(run/'handover.json',dict(remaining_gpu_pids=guard.gpu_pids(),other_active_jobs=other_active,restore_keepalive=False))


if __name__=='__main__':main()
