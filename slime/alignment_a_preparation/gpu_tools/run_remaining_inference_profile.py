"""One allocation: matched sanity timings, component profiles and a saved report."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time

import residual_rms_performance as guard
from resource_preflight import snapshot, validate
from summarize_remaining_inference import collect

PRIOR = guard.PREP/'gpu_runs/rms-perf-hgx03-20261008-02'


def worker_environment(env, name):
    if name not in ('baseline','candidate'):
        raise ValueError('Unknown profiling variant: '+name)
    result=dict(env)
    result['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']='0' if name=='baseline' else '1'
    return result


def failure_report(run, error):
    text=f'# 剩余推理开销采样失败\n\n实验：{run.name}。\n\n原因：{error}\n\n'
    text+='这不是已完成的归因报告，不能据此发布新的性能占比。实际阶段、退出状态和详细错误分别见本实验目录的state.json、*-exit.json和阶段日志。\n'
    (run/'FAILURE.md').write_text(text)
    (guard.PREP/'P1_REMAINING_INFERENCE_PROFILE_20261009.md').write_text(text)


def report(run, profiles):
    def rank(row):
        return row['distributed_rank'] if row['distributed_rank'] is not None else row['gpu_devices'][0]
    by_rank = {name:{rank(row):row for row in rows} for name,rows in profiles.items()}
    pairs = []
    for index in range(8):
        base, candidate = by_rank['baseline'][index], by_rank['candidate'][index]
        families = set(base['gpu_families']) | set(candidate['gpu_families'])
        delta = []
        for family in families:
            a=base['gpu_families'].get(family,dict(count=0,seconds=0.))
            b=candidate['gpu_families'].get(family,dict(count=0,seconds=0.))
            delta.append(dict(family=family,count_delta=b['count']-a['count'],gpu_sum_delta_s=b['seconds']-a['seconds'],baseline=a,candidate=b))
        pairs.append(dict(rank=index,gpu_event_delta=candidate['gpu_events']-base['gpu_events'],
                          gpu_union_delta_s=candidate['gpu_union_s']-base['gpu_union_s'],
                          families=sorted(delta,key=lambda row:row['gpu_sum_delta_s'],reverse=True)))
    result=dict(complete=True,pairs=pairs,profiles=profiles,
                limitation='Profiled CPU/GPU event costs are not a partition of the unprofiled latency gap. Different generated content can change MoE loads.')
    guard.save(run/'attribution-comparison.json',result)
    text='# RMS融合后剩余推理开销归因\n\n'
    text+=f"实验目录：{run.name}；节点：{socket.gethostname()}。baseline和RMS candidate均router off，固定8 prompts/1218输入、1024输出tokens。每Engine3批预热；sanity每组5批正式无profiler计时，profile每组1批完整128-token采样，启用相同CPU组件标记与shape记录。\n\n"
    text+='## 本节点无profiler检查\n\n'
    for name in ('baseline','candidate'):
        value=json.loads((run/f'sanity-{name}/summary.json').read_text())
        text+=f"- {name}: 中位数{value['median_s']:.6f}s，正式{value['formal_batches']}批。\n"
    text+='\n## 两个代表rank的GPU事件构成\n\n'
    for index in (0,4):
        a,b=by_rank['baseline'][index],by_rank['candidate'][index]
        text+=f"### rank {index}\n\nGPU事件：{a['gpu_events']} → {b['gpu_events']}；GPU区间并集：{a['gpu_union_s']:.3f} → {b['gpu_union_s']:.3f}s；GPU观察窗口：{a['gpu_window_s']:.3f} → {b['gpu_window_s']:.3f}s。\n\n"
        text+='| GPU family | baseline次数 | candidate次数 | baseline累计s | candidate累计s | 累计差值s |\n|---|---:|---:|---:|---:|---:|\n'
        for row in pairs[index]['families']:
            x,y=row['baseline'],row['candidate']
            text+=f"| {row['family']} | {x['count']} | {y['count']} | {x['seconds']:.4f} | {y['seconds']:.4f} | {row['gpu_sum_delta_s']:+.4f} |\n"
        text+='\nCPU组件区间为最内层标记在各CPU线程上的exclusive区间累计，不能与GPU时间相加，也不能直接解释为纯CPU计算。\n\n| CPU scope | baseline s | candidate s |\n|---|---:|---:|\n'
        for key in sorted(set(a['cpu_scope_exclusive_s'])|set(b['cpu_scope_exclusive_s'])):
            text+=f"| {key} | {a['cpu_scope_exclusive_s'].get(key,0):.4f} | {b['cpu_scope_exclusive_s'].get(key,0):.4f} |\n"
        text+=f"\nGPU空隙与下一事件enqueue关系：baseline {a['gpu_gap_s']}；candidate {b['gpu_gap_s']}。before_next_launch表示该GPU空隙中CUDA API尚未开始的部分，可包含调度或跨rank等待；不是纯CPU耗时。\n\n"
    text+='## 解释边界\n\n组件标记只包裹原方法，未修改张量数学、候选tile或CUDA graph配置。标记通过本次Engine的forward_hooks安装，sanity和普通Engine不安装。原RMS逐bit gate的数值源码及PyTorch/Triton版本必须保持一致才能采样。\n\nGPU累计时间、CPU区间和launch API会重叠，不能把这些差值加成约1.35s的分解。profile观察窗口受profiler扰动，性能比例引用本次无profiler sanity及此前180批完整计时。MoE输入内容可不同，kernel时长差异不是配置单因素证明。\n\n原始trace、全部8 ranks、CPU API统计与矩阵形状关联见本目录component-summary.json和attribution-comparison.json。\n'
    (run/'REPORT.md').write_text(text)
    (guard.PREP/'P1_REMAINING_INFERENCE_PROFILE_20261009.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--expected-sources',type=Path,required=True)
    args=parser.parse_args()
    if not args.run_id.startswith('remaining-profile-') or not args.run_id.replace('-','').isalnum():
        parser.error('Use a fresh remaining-profile- prefixed ID')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    signal.signal(signal.SIGTERM,guard.interrupted)
    signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs'/args.run_id
    run.mkdir(exist_ok=False)
    try:
        expected=json.loads(args.expected_sources.read_text())
        for name,digest in expected.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Queued source changed before execution: '+name)
        prior=json.loads((PRIOR/'exact-comparison.json').read_text())
        if not prior['model_exact'] or not prior['batch_exact']:
            raise RuntimeError('Missing passing original RMS gate')
        capture=json.loads((PRIOR/'exact-off/manifest.json').read_text())
        for name,digest in capture['source_sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Numerical source differs from passing gate: '+name)
        if hashlib.sha256(guard.FROZEN.read_bytes()).hexdigest()!=capture['source_rollout_sha256']:
            raise RuntimeError('Frozen workload changed')
        versions={name:importlib.metadata.version(name) for name in ('torch','triton','sglang')}
        old=json.loads((PRIOR/'runtime-versions.json').read_text())
        if any(versions[name]!=old[name] for name in ('torch','triton')):
            raise RuntimeError('Torch/Triton versions differ from passing RMS gate')
        import fcntl
        lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resources=snapshot(str(guard.PROJECT/'.a-tmp'))
        validate(resources,512,20)
        groups=[Path(item['path']) for item in resources['cgroups']
                if Path(item['path']).name.startswith('job_') and 'memory' in item['path']]
        if len(groups)!=1: raise RuntimeError('Missing unique allocation memory cgroup')
        guard.save(run/'resources.json',resources)
        env=guard.environment(run)
        env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env['SGLANG_PROFILE_WITH_STACK']='0'
        guard.save(run/'manifest.json',dict(host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),
            driver_pid=os.getpid(),driver_pgid=os.getpgrp(),source_sha256=expected,versions=versions,
            passing_gate=str(PRIOR/'exact-comparison.json'),numerical_gate_sources_unchanged=True,
            protocol=dict(warmup=3,sanity_steps=5,profile_steps=1,response_tokens=128)))
        guard.p2p_preflight(run)
        for phase in ('sanity','profile'):
            for name in ('baseline','candidate'):
                label=f'{phase}-{name}'
                output=run/label
                output.mkdir()
                argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/inference_benchmark.py'),
                      '--run',str(output),'--variant','baseline' if name=='baseline' else 'aligned',
                      '--residual-rms-mode','off' if name=='baseline' else 'candidate',
                      '--router-tile-mode','off','--warmup','3','--steps','5' if phase=='sanity' else '1']
                if phase=='profile': argv+=['--profile','--profile-shapes','--profile-attribution']
                worker_env=worker_environment(env,name)
                guard.save(output/'launch-environment.json',dict(
                    variant=name,unified_decode=worker_env['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']))
                guard.command(run,label,argv,worker_env,groups[0]/'memory.usage_in_bytes')
                resolved=json.loads((output/'resolved-server-args.json').read_text())
                if any(resolved['cuda_graph_config'][part]['backend']!='disabled' for part in ('prefill','decode')):
                    raise RuntimeError('Actual CUDA graph backend is not disabled')
        guard.save(run/'state.json',dict(status='analyzing',driver_pid=os.getpid(),time=time.time()))
        profiles={name:collect(run/f'profile-{name}') for name in ('baseline','candidate')}
        report(run,profiles)
        guard.save(run/'state.json',dict(status='complete',driver_pid=os.getpid(),report=str(run/'REPORT.md'),time=time.time()))
        print('REMAINING_PROFILE_COMPLETE '+str(run/'REPORT.md'),flush=True)
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),driver_pid=os.getpid(),time=time.time()))
        try:
            failure_report(run,str(exc))
        except OSError:
            pass
        raise
    finally:
        guard.stop_worker()


if __name__=='__main__': main()
