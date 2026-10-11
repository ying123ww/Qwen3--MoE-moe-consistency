"""Check that evidence I/O cannot leak into profiling or plain timing Engines."""
import json
from pathlib import Path
import torch

import qk_profile_annotations as annotations
import run_hgx03_qk_profile as runner


def test_evidence_is_deduplicated_and_disabled_inside_profile(tmp_path, monkeypatch):
    monkeypatch.setenv('SLIME_QK_PROFILE_DETAIL_DIR', str(tmp_path))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 3)
    monkeypatch.setattr(annotations, '_SEEN', set())
    annotations.evidence('layout', 'first', dict(copied=True))
    annotations.evidence('layout', 'first', dict(copied=False))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        annotations.evidence('layout', 'second', dict(copied=True))
    rows = [json.loads(x) for x in (tmp_path / 'rank-3-metadata.jsonl').read_text().splitlines()]
    assert rows == [dict(kind='layout', rank=3, copied=True)]
    assert ('layout', 'second') not in annotations._SEEN


def test_plain_benchmark_is_unchanged_but_profile_has_own_evidence_dir(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, '_COMMAND', lambda *a, **kw: calls.append(a))
    argv = ['python', '/original/inference_benchmark.py', '--steps', '5']
    env = dict(sentinel='retained')
    runner.command(tmp_path, 'sanity-candidate', argv, env, Path('/memory'))
    runner.command(tmp_path, 'profile-candidate', argv + ['--profile'], env, Path('/memory'))
    assert calls[0][2] == argv and calls[0][3] == env
    assert calls[1][2][1].endswith('profile_qk_benchmark.py')
    assert calls[1][3]['SLIME_QK_PROFILE_DETAIL_DIR'] == str(tmp_path / 'profile-candidate/qk-evidence')
    assert argv[1] == '/original/inference_benchmark.py' and env == dict(sentinel='retained')
