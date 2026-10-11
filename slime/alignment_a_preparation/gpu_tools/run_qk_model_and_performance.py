"""Exact-gated nine-Engine timing on the user's existing allocation."""
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import time

import residual_rms_performance as guard
from resource_preflight import snapshot,validate
from run_inference_repro import telemetry
import threading


def report(run,rows,complete=False):
    text='# Q/K 布局与归一化融合：模型门禁和整轮计时\n\n'
    text+=f"状态：{'完成' if complete else '进行中'}；完成 {len(rows)}/3 轮计时。\n\n"
    text+='三组均为同一 SSH/extern 条件；baseline、当前 RMS 版本、RMS＋联合 Q/K。每 Engine 3 预热＋20 正式，固定 8 prompts、每请求 128 输出 tokens，无 profiler/审计。所有组使用同一个 hook 工厂，off 模式透传原路径。\n\n'
    text+='| 轮次 | baseline 秒 | RMS 秒 | RMS＋Q/K 秒 | Q/K 延迟下降 | 最终比 baseline 慢 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in rows:
        text+=f"| {r['round']} | {r['baseline']:.6f} | {r['rms']:.6f} | {r['joint']:.6f} | {r['qk_reduction_percent']:.3f}% | {r['remaining_percent']:.3f}% |\n"
    if complete:
        text+=f"\n配对 Q/K 延迟下降中位数：{statistics.median(r['qk_reduction_percent'] for r in rows):.3f}%；最终额外开销中位数：{statistics.median(r['remaining_percent'] for r in rows):.3f}%。\n"
    text+='\n候选由显式实验 hook 安装，上游模型与 RMSNorm 文件未改动、默认关闭。只支持已验证的 BF16、小 M=1..4、head_dim=128、相同权重 dtype；其他路径回退原方法。保留上游 1024-wide 归约、sqrt 后取倒数、FP32 算术和末尾 BF16 舍入。\n\n有限模型/批次门禁结果见 exact-comparison.json；未做跨框架训练验证。\n'
    (run/'REPORT.md').write_text(text)
    (guard.PREP/'P1_QK_NORM_PERFORMANCE_20261011.md').write_text(text)


def main():
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs/qk-model-perf-hgx03-20261011-01';run.mkdir(exist_ok=False)
    monitor=None;stop=threading.Event()
    try:
        micro=guard.PREP/'gpu_runs/qk-micro-hgx03-20261011-02'
        if not json.loads((micro/'state.json').read_text())['status']=='complete':raise RuntimeError('Standalone gate incomplete')
        micro_manifest=json.loads((micro/'manifest.json').read_text())
        for p,v in micro_manifest['source_sha256'].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=v:raise RuntimeError('Prototype changed after passing micro gate')
        lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if guard.gpu_pids():raise RuntimeError('Other GPU task exists before model gate')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        guard.save(run/'resources.json',resources);env=guard.environment(run)
        env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        sources=list(guard.PREP.glob('gpu_tools/qk_*.py'))+[Path(__file__),guard.PREP/'gpu_tools/inference_benchmark.py']
        frozen=json.loads((guard.PREP/'gpu_runs/rms-repro-hgx03-20261011-01/manifest.json').read_text())['source_sha256']
        sources += [Path(p) for p in frozen if '/experiments/' in p]
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        for name in hashes:
            p=Path(name);target=run/'source-snapshots'/p.relative_to(guard.PROJECT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        guard.save(run/'manifest.json',dict(host=socket.gethostname(),pid=os.getpid(),pgid=os.getpgrp(),job_id=os.environ.get('SLURM_JOB_ID'),
            source_sha256=hashes,versions={k:importlib.metadata.version(k) for k in ('torch','triton','sglang')},
            protocol=dict(rounds=3,warmup=3,steps=20,response_tokens=128),restore_keepalive=False))
        guard.p2p_preflight(run)
        for mode in ('off','shadow','joint'):
            argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/qk_model_gate.py'),'capture','--mode',mode,'--output',str(run/f'exact-{mode}')]
            guard.command(run,'exact-'+mode,argv,env,groups[0]/'memory.usage_in_bytes')
        argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/qk_model_gate.py'),'compare','--reference',str(run/'exact-off'),
              '--shadow',str(run/'exact-shadow'),'--candidate',str(run/'exact-joint'),'--output',str(run/'exact-comparison.json')]
        guard.command(run,'exact-compare',argv,env,groups[0]/'memory.usage_in_bytes')
        rows=[];report(run,rows)
        monitor=threading.Thread(target=telemetry,args=(run,stop),daemon=True);monitor.start()
        for index,order in enumerate((('baseline','rms','joint'),('rms','joint','baseline'),('joint','baseline','rms')),1):
            row=dict(round=index);samples={}
            for name in order:
                label=f'perf-{index}-{name}';output=run/label;output.mkdir()
                worker=dict(env);worker['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']='0' if name=='baseline' else '1'
                worker['SLIME_QK_EXPERIMENT_MODE']='joint' if name=='joint' else 'off'
                argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/qk_inference_benchmark.py'),'--run',str(output),
                      '--variant','baseline' if name=='baseline' else 'aligned','--residual-rms-mode','off' if name=='baseline' else 'candidate',
                      '--router-tile-mode','off','--warmup','3','--steps','20','--response-tokens','128']
                guard.command(run,label,argv,worker,groups[0]/'memory.usage_in_bytes')
                summary=json.loads((output/'summary.json').read_text())
                if summary['formal_batches']!=20 or summary['profiled']:raise RuntimeError('Incomplete timing samples')
                samples[name]=[json.loads(x) for x in (output/'measurements.jsonl').read_text().splitlines()]
                if len(samples[name])!=23 or any(x['output_tokens']!=1024 for x in samples[name]):raise RuntimeError('Workload changed')
                if name=='joint' and (run/(label+'.log')).read_text().count('QK_NORM_CANDIDATE_FIRST_USE')!=8:
                    raise RuntimeError('Candidate not observed on eight ranks')
                row[name]=summary['median_s']
            matches=sum(a['output_sha256']==b['output_sha256'] for a,b in zip(samples['rms'],samples['joint']))
            if matches!=23:raise RuntimeError('Sampling tokens differ after Q/K fusion')
            row.update(sampling_hash_matches=matches,qk_reduction_percent=(1-row['joint']/row['rms'])*100,
                       remaining_percent=(row['joint']/row['baseline']-1)*100)
            rows.append(row);guard.save(run/'comparison.json',dict(complete=False,rows=rows));report(run,rows)
            print(json.dumps(row),flush=True)
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=v for p,v in hashes.items()):raise RuntimeError('Source changed during formal experiment')
        guard.save(run/'comparison.json',dict(complete=True,rows=rows,formal_batches=180,sampling_hash_matches=69))
        report(run,rows,True);guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()))
        (run/'FAILURE.md').write_text(f'# Q/K 实验失败\n\n{exc}\n\n不发布未通过门禁的收益。\n')
        raise
    finally:
        stop.set()
        if monitor:monitor.join(timeout=12)
        guard.stop_worker()


if __name__=='__main__':main()
