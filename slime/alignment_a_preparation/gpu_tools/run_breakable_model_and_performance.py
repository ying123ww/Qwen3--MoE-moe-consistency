"""Exact-gated Plan 2 versus original eager, Plan 1 and baseline on HGX03."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import threading
import time

import residual_rms_performance as guard
from resource_preflight import snapshot, validate
from run_inference_repro import telemetry


def graph_proof(output):
    rows=[json.loads(p.read_text()) for p in (output/'graph-evidence').glob('rank-*-size-*.json')]
    if {p['rank'] for p in rows}!=set(range(8)):raise RuntimeError('Missing successful replay ranks')
    if any(p['segments']!=49 or p['breaks']!=48 or not p['successful_replay'] or
           any('_a2a_forward_with_output_impl' not in name for name in p['break_functions']) for p in rows):
        raise RuntimeError('Unexpected graph boundaries')
    return rows


def report(run,rows,complete=False):
    text='# Plan 2：DeepEP normal 与 Breakable CUDA Graph\n\n'
    text+=f"状态：{'完成' if complete else '进行中'}；完成 {len(rows)}/3 轮。\n\n"
    text+='四组在同一 ai_hgx_03 / SSH extern / Slurm 分配运行。Plan 2 使用原始确定性算子、decode breakable、prefill eager；整个 MoE eager。Plan 1 是 RMS＋联合 Q/K，graph disabled。每 Engine 3 预热＋20 正式、固定 8 prompts / 1218 输入 tokens / 1024 输出 tokens，整轮 generate 无 profiler/逐调用审计。\n\n'
    text+='| 轮次 | baseline 秒 | 原确定性 eager 秒 | Plan 1 秒 | Plan 2 秒 | Plan 2 相对原确定性延迟下降 | Plan 2 相对 Plan 1 延迟下降 | Plan 2 比 baseline 慢 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        text+=f"| {r['round']} | {r['baseline']:.6f} | {r['eager']:.6f} | {r['plan1']:.6f} | {r['breakable']:.6f} | {r['vs_eager_percent']:.3f}% | {r['vs_plan1_percent']:.3f}% | {r['remaining_percent']:.3f}% |\n"
    if complete:
        text+=f"\n配对下降中位数：对原确定性 {statistics.median(r['vs_eager_percent'] for r in rows):.3f}%，对 Plan 1 {statistics.median(r['vs_plan1_percent'] for r in rows):.3f}%；相对 baseline 差值 {statistics.median(r['remaining_percent'] for r in rows):.3f}%。\n"
    text+='\n有限 greedy 四组输入的 tokens/routes/logprob bits 与 batch 划分门禁见 exact-comparison.json。各 rank 成功 replay 的分段结构见每组 graph-evidence。启动、cache flush 与模型加载不计入 generate 延迟；不宣称跨框架训练一致。\n'
    (run/'REPORT.md').write_text(text)
    (guard.PREP/'P2_BREAKABLE_CUDA_GRAPH_PERFORMANCE_20261011.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    if not args.run_id.startswith('breakable-model-perf-') or '/' in args.run_id:raise ValueError('Invalid run ID')
    signal.signal(signal.SIGTERM,guard.interrupted);signal.signal(signal.SIGINT,guard.interrupted)
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    monitor=None;stop=threading.Event()
    try:
        lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if guard.gpu_pids():raise RuntimeError('Other GPU task exists')
        resources=snapshot(str(guard.PROJECT/'.a-tmp'));validate(resources,512,20)
        groups=[Path(x['path']) for x in resources['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups)!=1:raise RuntimeError('Allocation memory cgroup missing')
        guard.save(run/'resources.json',resources);env=guard.environment(run)
        env['PYTHONPATH']=str(guard.PREP/'gpu_tools')+':'+env.get('PYTHONPATH','')
        env['SGLANG_MEMORY_SAVER_CUDA_GRAPH']='0'
        srt=guard.PROJECT/'experiments/sglang-main-sm80/python/sglang/srt'
        paths=[Path(__file__),*guard.PREP.glob('gpu_tools/breakable_*.py'),*guard.PREP.glob('gpu_tools/qk_*.py'),
               guard.PREP/'gpu_tools/inference_benchmark.py',srt/'server_args.py',srt/'models/qwen3_moe.py',
               srt/'layers/moe/ep_moe/layer.py',srt/'layers/layernorm.py',srt/'layers/attention/triton_backend.py',srt/'batch_invariant_ops/batch_invariant_ops.py',
               srt/'batch_invariant_ops/fused_rmsnorm_candidate.py']
        paths+=list((srt/'model_executor/runner_backend_utils/breakable_cuda_graph').glob('*.py'))
        paths+=[srt/'model_executor/runner_backend/breakable_cuda_graph_backend.py',srt/'model_executor/runner/decode_cuda_graph_runner.py']
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for name in hashes:
            p=Path(name);target=run/'source-snapshots'/p.relative_to(guard.PROJECT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        guard.save(run/'manifest.json',dict(host=socket.gethostname(),pid=os.getpid(),pgid=os.getpgrp(),job_id=os.environ.get('SLURM_JOB_ID'),
            source_sha256=hashes,versions={k:importlib.metadata.version(k) for k in ('torch','triton','sglang')},
            protocol=dict(rounds=3,warmup=3,steps=20,response_tokens=128),restore_keepalive=False))
        guard.p2p_preflight(run)
        for mode in ('eager','breakable'):
            argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_model_gate.py'),'capture','--mode',mode,'--output',str(run/f'exact-{mode}')]
            guard.command(run,'exact-'+mode,argv,env,groups[0]/'memory.usage_in_bytes')
            if mode=='breakable':graph_proof(run/f'exact-{mode}')
        argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_model_gate.py'),'compare','--reference',str(run/'exact-eager'),
              '--candidate',str(run/'exact-breakable'),'--output',str(run/'exact-comparison.json')]
        guard.command(run,'exact-compare',argv,env,groups[0]/'memory.usage_in_bytes')
        rows=[];report(run,rows)
        monitor=threading.Thread(target=telemetry,args=(run,stop),daemon=True);monitor.start()
        orders=(('baseline','eager','plan1','breakable'),('eager','plan1','breakable','baseline'),('breakable','baseline','eager','plan1'))
        for index,order in enumerate(orders,1):
            row=dict(round=index);samples={}
            for mode in order:
                label=f'perf-{index}-{mode}';output=run/label;output.mkdir()
                worker=dict(env);worker['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']='0' if mode=='baseline' else '1'
                worker['SLIME_BREAKABLE_EXPERIMENT_MODE']=mode
                argv=[str(guard.PYTHON),str(guard.PREP/'gpu_tools/breakable_inference_benchmark.py'),'--run',str(output),
                      '--variant','baseline' if mode=='baseline' else 'aligned','--residual-rms-mode','candidate' if mode=='plan1' else 'off',
                      '--router-tile-mode','off','--warmup','3','--steps','20','--response-tokens','128']
                guard.command(run,label,argv,worker,groups[0]/'memory.usage_in_bytes')
                summary=json.loads((output/'summary.json').read_text())
                if summary['formal_batches']!=20 or summary['profiled']:raise RuntimeError('Incomplete timing samples')
                samples[mode]=[json.loads(x) for x in (output/'measurements.jsonl').read_text().splitlines()]
                if len(samples[mode])!=23 or any(x['output_tokens']!=1024 for x in samples[mode]):raise RuntimeError('Workload changed')
                if mode=='breakable':graph_proof(output)
                if mode=='plan1' and (run/(label+'.log')).read_text().count('QK_NORM_CANDIDATE_FIRST_USE')!=8:raise RuntimeError('Plan 1 inactive')
                resolved=json.loads((output/'resolved-server-args.json').read_text())['cuda_graph_config']
                if resolved['decode']['backend']!=('breakable' if mode=='breakable' else 'disabled') or resolved['prefill']['backend']!='disabled':
                    raise RuntimeError('Unexpected resolved graph configuration')
                row[mode]=summary['median_s']
            matches=sum(a['output_sha256']==b['output_sha256']==c['output_sha256'] for a,b,c in zip(samples['eager'],samples['plan1'],samples['breakable']))
            if matches!=23:raise RuntimeError('Aligned sampling outputs differ')
            row.update(sampling_hash_matches=matches,vs_eager_percent=(1-row['breakable']/row['eager'])*100,
                vs_plan1_percent=(1-row['breakable']/row['plan1'])*100,remaining_percent=(row['breakable']/row['baseline']-1)*100)
            rows.append(row);guard.save(run/'comparison.json',dict(complete=False,rows=rows));report(run,rows)
            print(json.dumps(row),flush=True)
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise RuntimeError('Source changed')
        guard.save(run/'comparison.json',dict(complete=True,rows=rows,formal_batches=240,sampling_hash_matches=69))
        report(run,rows,True);guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()))
        (run/'FAILURE.md').write_text(f'# Plan 2 实验未完成\n\n{exc}\n\n不发布未通过门禁的收益。\n')
        raise
    finally:
        stop.set()
        if monitor:monitor.join(timeout=12)
        guard.stop_worker()


if __name__=='__main__':main()
