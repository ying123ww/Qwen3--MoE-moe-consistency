"""Repeat the original SSH/extern three-mode protocol without numerical changes."""
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


def write_report(run, rows, complete=False):
    lines = ["# ai_hgx_03 三模式复现（2026-10-11）", "",
             f"状态：{'完成' if complete else '进行中'}；完成 {len(rows)}/3 轮。",
             "", "直接从 SSH 的 Slurm step_extern 启动，与旧 ai_hgx_03 实验的 cgroup 类型相同。",
             "每轮三个独立 Engine，顺序轮换；每 Engine 3 批预热、20 批正式。",
             "固定 8 prompts、1218 输入 tokens、每请求 128 输出 tokens。",
             "测量 blocking generate；无 profiler、无审计、无诊断 forward hook。",
             "", "| 轮次 | baseline 秒 | aligned 未融合秒 | RMS 融合秒 | 未融合相对 baseline | 融合相对 baseline | 融合延迟下降 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['round']} | {row['baseline']:.6f} | {row['aligned']:.6f} | {row['candidate']:.6f} | {row['aligned_over_baseline_percent']:.3f}% | {row['remaining_percent']:.3f}% | {row['fusion_reduction_percent']:.3f}% |")
    if complete:
        lines.extend(["", "三轮配对比例中位数："])
        for key in ('aligned_over_baseline_percent', 'remaining_percent', 'fusion_reduction_percent'):
            lines.append(f"- {key}: {statistics.median(row[key] for row in rows):.3f}%")
    lines.extend(["", "所有样本、解析配置、源码快照、版本、资源及每 10 秒遥测均保留。",
                  "复用旧有限 greedy 逐 bit gate 的前提是全部 gate 数值源码和 frozen rollout 未变化；本次另核对各轮 aligned/candidate 的 23 批输出哈希。",
                  "遥测只提供关联线索；本次未直接操纵频率、线程或启动方式，不能单独确证跨节点差异的因果。"])
    (run / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.startswith('rms-repro-hgx03-') or not args.run_id.replace('-', '').isalnum():
        parser.error('Use a fresh rms-repro-hgx03- ID')
    if socket.gethostname() != 'ai_hgx_03':
        raise RuntimeError('This reproduction requires ai_hgx_03')
    signal.signal(signal.SIGTERM, guard.interrupted)
    signal.signal(signal.SIGINT, guard.interrupted)
    run = guard.PREP / 'gpu_runs' / args.run_id
    run.mkdir(exist_ok=False)
    stop = threading.Event()
    monitor = None
    try:
        lock = open(guard.PROJECT / '.a-tmp/router-tile-performance.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if guard.gpu_pids():
            raise RuntimeError('Existing GPU processes; no unrelated job will be stopped')
        old = guard.PREP / 'gpu_runs/rms-perf-hgx03-20261008-02'
        gate_manifest = json.loads((old / 'exact-off/manifest.json').read_text())
        for name, digest in gate_manifest['source_sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Old exact gate source changed: ' + name)
        exact = json.loads((old / 'exact-comparison.json').read_text())
        if not exact['model_exact'] or not exact['batch_exact']:
            raise RuntimeError('Old exact gate did not pass')
        old_measurement = json.loads((old / 'perf-1-candidate/manifest.json').read_text())
        if hashlib.sha256(guard.FROZEN.read_bytes()).hexdigest() != old_measurement['frozen_sha256']:
            raise RuntimeError('Frozen input changed')
        state = snapshot(str(guard.PROJECT / '.a-tmp'))
        validate(state, 512, 20)
        groups = [Path(x['path']) for x in state['cgroups'] if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups) != 1:
            raise RuntimeError('Cannot establish allocation memory group')
        guard.save(run / 'resources.json', state)
        sources = dict(gate_manifest['source_sha256'])
        for name in ('reproduce_hgx03_three_modes.py', 'inference_benchmark.py', 'residual_rms_performance.py', 'resource_preflight.py', 'run_inference_repro.py'):
            path = Path(__file__).with_name(name)
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name in sources:
            path = Path(name)
            relative = path.relative_to(guard.PROJECT) if path.is_relative_to(guard.PROJECT) else Path('external') / path.relative_to('/')
            target = run / 'source-snapshots' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        versions = {k: importlib.metadata.version(k) for k in ('torch', 'triton', 'sglang')}
        if versions['torch'] != '2.11.0+cu129' or versions['triton'] != '3.6.0':
            raise RuntimeError('Old torch/Triton versions differ')
        guard.save(run / 'manifest.json', dict(host=socket.gethostname(), pid=os.getpid(), pgid=os.getpgrp(),
            job_id=os.environ.get('SLURM_JOB_ID'), launch='ssh-step_extern', cgroup=Path('/proc/self/cgroup').read_text(),
            source_sha256=sources, versions=versions, exact_gate_source=str(old / 'exact-comparison.json'),
            numerical_gate_sources_unchanged=True, frozen_unchanged=True,
            protocol=dict(rounds=3, warmup=3, steps=20, response_tokens=128)))
        env = guard.environment(run)
        guard.p2p_preflight(run)
        monitor = threading.Thread(target=telemetry, args=(run, stop), daemon=True)
        monitor.start()
        rows = []
        write_report(run, rows)
        for index, order in enumerate((('baseline', 'aligned', 'candidate'), ('aligned', 'candidate', 'baseline'), ('candidate', 'baseline', 'aligned')), 1):
            row = dict(round=index)
            batches = {}
            for name in order:
                label = f'perf-{index}-{name}'
                output = run / label
                output.mkdir()
                child_env = dict(env)
                child_env['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE'] = '0' if name == 'baseline' else '1'
                argv = [str(guard.PYTHON), str(guard.PREP / 'gpu_tools/inference_benchmark.py'), '--run', str(output),
                        '--variant', 'baseline' if name == 'baseline' else 'aligned', '--residual-rms-mode', 'candidate' if name == 'candidate' else 'off',
                        '--router-tile-mode', 'off', '--warmup', '3', '--steps', '20', '--response-tokens', '128']
                guard.command(run, label, argv, child_env, groups[0] / 'memory.usage_in_bytes')
                summary = json.loads((output / 'summary.json').read_text())
                samples = [json.loads(line) for line in (output / 'measurements.jsonl').read_text().splitlines()]
                if summary['formal_batches'] != 20 or summary['profiled'] or len(samples) != 23 or any(x['output_tokens'] != 1024 for x in samples):
                    raise RuntimeError('Measurement completeness mismatch: ' + label)
                resolved = json.loads((output / 'resolved-server-args.json').read_text())
                if any(resolved['cuda_graph_config'][phase]['backend'] != 'disabled' for phase in ('prefill', 'decode')):
                    raise RuntimeError('CUDA graph protocol changed')
                row[name] = summary['median_s']
                batches[name] = samples
            same = sum(a['output_sha256'] == b['output_sha256'] for a, b in zip(batches['aligned'], batches['candidate']))
            row.update(aligned_candidate_hash_matches=same, compared_batches=23,
                aligned_over_baseline_percent=(row['aligned'] / row['baseline'] - 1) * 100,
                remaining_percent=(row['candidate'] / row['baseline'] - 1) * 100,
                fusion_reduction_percent=(1 - row['candidate'] / row['aligned']) * 100)
            if same != 23:
                raise RuntimeError('Aligned/candidate sampling outputs changed')
            rows.append(row)
            guard.save(run / 'comparison.json', dict(complete=False, rows=rows))
            write_report(run, rows)
            print(json.dumps(row), flush=True)
        if any(hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest for name, digest in sources.items()):
            raise RuntimeError('Source changed during measurement')
        guard.save(run / 'comparison.json', dict(complete=True, rows=rows, formal_batches=180, sampling_hash_matches=69, numerical_sources_unchanged=True))
        write_report(run, rows, True)
        guard.save(run / 'state.json', dict(status='complete', pid=os.getpid(), pgid=os.getpgrp(), report=str(run / 'REPORT.md'), time=time.time()))
        print('THREE_MODE_REPRO_COMPLETE ' + str(run / 'REPORT.md'), flush=True)
    except BaseException as exc:
        guard.save(run / 'state.json', dict(status='failed', error=str(exc), pid=os.getpid(), time=time.time()))
        (run / 'FAILURE.md').write_text(f'# 复现失败\n\n{exc}\n\n部分数据不能视为完成对照。\n')
        raise
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=12)
        guard.stop_worker()


if __name__ == '__main__':
    main()
