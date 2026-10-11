"""Run matched profiling on the verified SSH/extern allocation."""
import json
import os
from pathlib import Path
import socket

import run_remaining_inference_profile as runner
import residual_rms_performance as guard

_COMMAND = guard.command
_ENVIRONMENT = runner.worker_environment


def command(run, label, argv, env, memory_file, **kwargs):
    argv = list(argv)
    env = dict(env)
    if '--profile' in argv:
        argv[1] = str(Path(__file__).with_name('profile_qk_benchmark.py'))
        env['SLIME_QK_PROFILE_DETAIL_DIR'] = str(run / label / 'qk-evidence')
    return _COMMAND(run, label, argv, env, memory_file, **kwargs)


def report(run, profiles):
    def rank_of(row):
        return row['distributed_rank'] if row['distributed_rank'] is not None else row['gpu_devices'][0]
    by_rank = {name: {rank_of(row): row for row in rows} for name, rows in profiles.items()}
    pairs = []
    for rank in range(8):
        a, b = by_rank['baseline'][rank], by_rank['candidate'][rank]
        families = set(a['gpu_families']) | set(b['gpu_families'])
        pairs.append(dict(rank=rank, gpu_event_delta=b['gpu_events'] - a['gpu_events'],
             gpu_union_delta_s=b['gpu_union_s'] - a['gpu_union_s'], families=[dict(family=f,
             baseline=a['gpu_families'].get(f, dict(count=0, seconds=0)),
             candidate=b['gpu_families'].get(f, dict(count=0, seconds=0))) for f in sorted(families)]))
    guard.save(run / 'attribution-comparison.json', dict(complete=True, pairs=pairs, profiles=profiles))
    text = '# ai_hgx_03 Q/K 热点调查（2026-10-11）\n\n'
    text += '同一 SSH/extern 启动器；baseline 与当前 RMS 融合版本均 router off。每组 3 批预热；5 批无 profiler 检查；1 批完整 128-token profile。\n\n'
    for mode in ('baseline', 'candidate'):
        summary = json.loads((run / f'sanity-{mode}/summary.json').read_text())
        text += f"- {mode} 无 profiler 中位数：{summary['median_s']:.6f} 秒。\n"
    text += '\n| rank | baseline GPU 事件 | RMS GPU 事件 | 增量 | baseline GPU 区间并集秒 | RMS GPU 区间并集秒 |\n|---|---:|---:|---:|---:|---:|\n'
    for rank in range(8):
        a, b = by_rank['baseline'][rank], by_rank['candidate'][rank]
        text += f"| {rank} | {a['gpu_events']} | {b['gpu_events']} | {b['gpu_events'] - a['gpu_events']} | {a['gpu_union_s']:.6f} | {b['gpu_union_s']:.6f} |\n"
    text += '\nGPU 事件包含 kernel、memcpy、memset，不等于 kernel 发射次数。CPU、GPU 累计时间与运行时等待不可相加解释无 profiler 延迟。Q/K 细分标记仅用于 profile Engines，张量与布局证据只在预热采集。全部 8 ranks 及细分结果见 JSON 原始数据。\n'
    (run / 'REPORT.md').write_text(text)
    (guard.PREP / 'P1_QK_PROFILE_HGX03_20261011.md').write_text(text)


def failure_report(run, error):
    text = f'# ai_hgx_03 Q/K profile 失败\n\n{error}\n\n部分产物不能视为完成调查。\n'
    (run / 'FAILURE.md').write_text(text)
    (guard.PREP / 'P1_QK_PROFILE_HGX03_20261011.md').write_text(text)


if __name__ == '__main__':
    if socket.gethostname() != 'ai_hgx_03':
        raise RuntimeError('Use the verified ai_hgx_03 allocation')
    guard.command = command
    runner.report = report
    runner.failure_report = failure_report
    runner.main()
