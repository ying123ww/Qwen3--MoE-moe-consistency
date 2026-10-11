"""Balanced three-mode timing under inherited vs cleaned scheduler environment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import threading
import time

import residual_rms_performance as guard
from resource_preflight import snapshot,validate


def worker_environment(env,condition,name):
    if condition not in ('inherited','clean') or name not in ('baseline','aligned','candidate'):
        raise ValueError('Unknown reproduction condition or variant')
    result=dict(env)
    if condition=='clean':
        for key in list(result):
            if key.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MPICH_')):
                result.pop(key)
    result['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']='0' if name=='baseline' else '1'
    return result


def telemetry(run,stop):
    query='timestamp,index,pstate,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw,power.limit'
    while not stop.is_set():
        row=dict(time=time.time(),loadavg=os.getloadavg())
        try:
            row['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu='+query,'--format=csv,noheader'],text=True,timeout=10)
            cpus=sorted(os.sched_getaffinity(0))
            row['cpu_frequency_khz']={str(cpu):(Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq').read_text().strip())
                                      for cpu in cpus if Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq').exists()}
        except Exception as exc: row['telemetry_error']=str(exc)
        with (run/'telemetry.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
        stop.wait(10)


def report(run,rows):
    text='# 推理剩余开销复现实验\n\n'
    text+=f"状态：{'完整对照完成' if len(rows)==6 else '部分结果，实验仍在进行'}；已完成{len(rows)}/6个配对组。\n\n"
    text+=f'节点：{socket.gethostname()}。所有组通过同一srun分配下的同一启动器执行，CPU affinity/cgroup不改变；clean仅去掉子进程的Slurm/PMI类环境变量，不是SSH启动。每Engine3预热+5正式，固定8 prompts/1024输出tokens，无profiler。\n\n'
    text+='| 条件 | 轮次 | baseline s | aligned-off s | RMS candidate s | candidate比baseline慢 | RMS融合延迟下降 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for row in rows:
        text+=f"| {row['condition']} | {row['round']} | {row['baseline']:.6f} | {row['aligned']:.6f} | {row['candidate']:.6f} | {row['remaining_percent']:.3f}% | {row['fusion_reduction_percent']:.3f}% |\n"
    text+='\n条件分组及轮次顺序交错；worker指纹在第1批预热采集一次，正式区间没有逐调用hook I/O。telemetry每10秒在独立线程记录GPU状态/CPU频率/load，原始观测用于诊断而非将相关性当作因果。\n\n每轮生成输出哈希、完整配置、worker环境/线程/cgroup与所有批次均保存。若clean与inherited不同，下一步逐个环境变量组隔离；若不不同，则不支持Slurm环境这一假说。跨节点剩余比例不是固定常数，不能用当前profile累计时间直接拆分旧1.35s。\n'
    (run/'REPORT.md').write_text(text)
    (guard.PREP/'P1_INFERENCE_REPRO_20261009.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--expected-sources',type=Path,required=True)
    args=parser.parse_args()
    if not args.run_id.startswith('remaining-profile-repro-') or not args.run_id.replace('-','').isalnum():
        parser.error('Use a fresh remaining-profile-repro- ID')
    signal.signal(signal.SIGTERM,guard.interrupted)
    signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs'/args.run_id
    run.mkdir(exist_ok=False)
    stop=threading.Event()
    monitor=None
    try:
        expected=json.loads(args.expected_sources.read_text())
        for name,digest in expected.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Source changed while queued: '+name)
        import fcntl
        lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=snapshot(str(guard.PROJECT/'.a-tmp'))
        validate(state,512,20)
        groups=[Path(x['path']) for x in state['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1: raise RuntimeError('Cannot identify allocation memory group')
        guard.save(run/'resources.json',state)
        env=guard.environment(run)
        env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        guard.save(run/'manifest.json',dict(host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),
            pid=os.getpid(),source_sha256=expected,versions={k:importlib.metadata.version(k) for k in ('torch','triton','sglang')},
            protocol=dict(rounds=3,warmup=3,steps=5,conditions=['inherited','clean'],variants=['baseline','aligned','candidate'])))
        guard.p2p_preflight(run)
        monitor=threading.Thread(target=telemetry,args=(run,stop),daemon=True)
        monitor.start()
        rows=[]
        orders=[('baseline','aligned','candidate'),('aligned','candidate','baseline'),('candidate','baseline','aligned')]
        for index,order in enumerate(orders,1):
            conditions=('inherited','clean') if index%2 else ('clean','inherited')
            for condition in conditions:
                row=dict(round=index,condition=condition)
                for name in order:
                    label=f'{condition}-{index}-{name}'
                    output=run/label
                    output.mkdir()
                    child_env=worker_environment(env,condition,name)
                    argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/inference_benchmark.py'),'--run',str(output),
                          '--variant','baseline' if name=='baseline' else 'aligned','--residual-rms-mode','candidate' if name=='candidate' else 'off',
                          '--router-tile-mode','off','--warmup','3','--steps','5','--runtime-diagnostics']
                    guard.command(run,label,argv,child_env,groups[0]/'memory.usage_in_bytes')
                    summary=json.loads((output/'summary.json').read_text())
                    if summary['formal_batches']!=5 or summary['profiled']: raise RuntimeError('Invalid formal measurements')
                    if len(list((output/'worker-diagnostics').glob('rank-*.json')))!=8: raise RuntimeError('Missing worker fingerprints')
                    row[name]=summary['median_s']
                row['remaining_percent']=(row['candidate']/row['baseline']-1)*100
                row['fusion_reduction_percent']=(1-row['candidate']/row['aligned'])*100
                rows.append(row)
                guard.save(run/'comparison.json',dict(complete=False,rows=rows))
                report(run,rows)
        guard.save(run/'comparison.json',dict(complete=True,rows=rows))
        report(run,rows)
        guard.save(run/'state.json',dict(status='complete',report=str(run/'REPORT.md'),pid=os.getpid(),time=time.time()))
        print('REMAINING_PROFILE_COMPLETE '+str(run/'REPORT.md'),flush=True)
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()))
        failure=f'# 复现实验失败\n\n实验：{run.name}。\n\n错误：{exc}\n\n原始样本和阶段日志保留，部分结果不能视为完整对照。\n'
        try:
            (run/'FAILURE.md').write_text(failure)
            (guard.PREP/'P1_INFERENCE_REPRO_20261009.md').write_text(failure)
        except OSError:
            pass
        raise
    finally:
        stop.set()
        if monitor: monitor.join(timeout=12)
        guard.stop_worker()


if __name__=='__main__': main()
