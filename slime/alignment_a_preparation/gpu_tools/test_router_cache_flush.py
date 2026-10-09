"""Cache flush is outside timing, but success remains mandatory."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import inference_benchmark as benchmark
import router_tile_performance as performance


@pytest.mark.parametrize('wrapped', [False, True])
def test_waits_for_success_after_transient_rejection(monkeypatch, wrapped):
    results = iter([False, False, True])
    seen = []
    def flush():
        value = next(results)
        seen.append(value)
        return SimpleNamespace(success=value) if wrapped else value
    monkeypatch.setattr(benchmark.time, 'sleep', lambda seconds: None)
    assert benchmark.flush_cache_until_idle(SimpleNamespace(flush_cache=flush)) == 2
    assert seen == [False, False, True]


def test_permanent_flush_failure_is_not_ignored():
    with pytest.raises(RuntimeError, match='idle timeout'):
        benchmark.flush_cache_until_idle(SimpleNamespace(flush_cache=lambda: False), timeout=0)


def resume_fixture(tmp_path, monkeypatch):
    model = tmp_path / 'model.py'
    model.write_text('frozen')
    rollout = tmp_path / 'rollout.pt'
    rollout.write_bytes(b'rollout')
    monkeypatch.setattr(performance, 'FROZEN', rollout)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        host=performance.socket.gethostname(), protocol=dict(warmup=3, steps=20), source_sha256={})))
    (tmp_path / 'state.json').write_text(json.dumps(dict(status='failed')))
    (tmp_path / 'exact-comparison.json').write_text(json.dumps(dict(model_exact=True, batch_exact=True)))
    (tmp_path / 'exact-off').mkdir()
    (tmp_path / 'exact-off/manifest.json').write_text(json.dumps(dict(
        source_sha256={str(model):hashlib.sha256(model.read_bytes()).hexdigest()},
        source_rollout_sha256=hashlib.sha256(rollout.read_bytes()).hexdigest())))
    return model


def test_resume_requires_unchanged_model_source(tmp_path, monkeypatch):
    model = resume_fixture(tmp_path, monkeypatch)
    model.write_text('changed')
    with pytest.raises(RuntimeError, match='Model/gate source changed'):
        performance.verify_resume(tmp_path, SimpleNamespace(warmup=3, steps=20))
    assert not (tmp_path / 'resume-validation.json').exists()


def test_resume_retains_passing_exact_gate(tmp_path, monkeypatch):
    resume_fixture(tmp_path, monkeypatch)
    performance.verify_resume(tmp_path, SimpleNamespace(warmup=3, steps=20))
    assert json.loads((tmp_path / 'resume-validation.json').read_text())['numerical_sources_unchanged']
