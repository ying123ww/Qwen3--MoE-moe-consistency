"""CPU checks for strict router tile experiment gates (no kernel correctness claim)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import router_tile_exact as gate

HELPER = Path(__file__).resolve().parents[3] / 'experiments/sglang-main-sm80/python/sglang/srt/models/qwen3_router_tile.py'


def helper(monkeypatch, tmp_path, mode='off'):
    monkeypatch.setenv('SLIME_QWEN3_ROUTER_TILE_MODE', mode)
    monkeypatch.setenv('SLIME_QWEN3_ROUTER_TILE_AUDIT_DIR', str(tmp_path))
    spec = importlib.util.spec_from_file_location('isolated_router_tile', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_is_exact_torch_mm(monkeypatch, tmp_path):
    module = helper(monkeypatch, tmp_path)
    a, b = torch.randn(2, 4), torch.randn(4, 8)
    assert torch.equal(module.router_matmul(a, b, layer_id=0), torch.mm(a, b))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('a,b,expected', [((1, 2048), (2048, 128), True),
    ((16, 2048), (2048, 128), True), ((17, 2048), (2048, 128), False),
    ((0, 2048), (2048, 128), False), ((2, 2048), (2048, 2560), False),
    ((2, 1024), (1024, 128), False)])
def test_narrow_shape_scope(monkeypatch, tmp_path, a, b, expected):
    assert helper(monkeypatch, tmp_path).eligible_shape(a, b) is expected


def test_invalid_mode_fails(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match='Invalid'):
        helper(monkeypatch, tmp_path, 'typo')


def test_shadow_rejects_cpu(monkeypatch, tmp_path):
    module = helper(monkeypatch, tmp_path, 'shadow')
    with pytest.raises(RuntimeError, match='FP32 CUDA'):
        module.router_matmul(torch.empty(2, 2048), torch.empty(2048, 128), layer_id=0)


def test_shadow_signed_zero_mismatch_is_fatal_and_audited(monkeypatch, tmp_path):
    module = helper(monkeypatch, tmp_path, 'shadow')
    a = SimpleNamespace(dtype=torch.float32, is_cuda=True, device='cuda:0', shape=(2, 2048))
    b = SimpleNamespace(dtype=torch.float32, device='cuda:0', shape=(2048, 128))
    reference = torch.zeros(2, 128)
    candidate = reference.clone()
    candidate[0, 0] = -0.0
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    monkeypatch.setattr(torch, 'mm', lambda a, b: reference)
    monkeypatch.setattr(module, '_candidate', lambda a, b: candidate)
    with pytest.raises(RuntimeError, match='exact gate FAILED'):
        module.router_matmul(a, b, layer_id=7)
    row = json.loads(next(tmp_path.glob('*.jsonl')).read_text())
    assert row['mismatches'] == 1 and row['compared_elements'] == 256
    assert row['rank'] == 0 and row['layer'] == 7


def test_candidate_error_never_falls_back(monkeypatch, tmp_path):
    module = helper(monkeypatch, tmp_path, 'candidate')
    a = SimpleNamespace(dtype=torch.float32, is_cuda=True, device='cuda:0', shape=(2, 2048))
    b = SimpleNamespace(dtype=torch.float32, device='cuda:0', shape=(2048, 128))
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    def fail(a, b):
        raise ValueError('kernel failed')
    monkeypatch.setattr(module, '_candidate', fail)
    with pytest.raises(RuntimeError, match='rank=0 layer=3') as exc:
        module.router_matmul(a, b, layer_id=3)
    assert isinstance(exc.value.__cause__, ValueError)


def test_comparison_checks_bits_and_shape():
    a = [({}, {'response_logprob_path': np.array([0.0], dtype='<f4')})]
    b = [({}, {'response_logprob_path': np.array([-0.0], dtype='<f4')})]
    assert gate.compare_samples(a, b, 'signed-zero')
    assert not gate.compare_samples(a, a, 'equal')


def test_audit_requires_every_rank_layer(tmp_path):
    audit = tmp_path / 'audit'
    audit.mkdir()
    path = audit / 'rank-0.jsonl'
    row = dict(mode='shadow', rank=0, layer=0, eligible=True, a_shape=[2, 2048],
               mismatches=0, nonfinite_elements=0, compared_elements=256)
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='coverage mismatch'):
        gate.audit_summary(tmp_path, 'shadow', expected_ranks=2, expected_layers=1)
    assert gate.audit_summary(tmp_path, 'shadow', expected_ranks=1, expected_layers=1)['compared_elements'] == 256
    row['compared_elements'] = 0
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='no actual bit comparisons'):
        gate.audit_summary(tmp_path, 'shadow', expected_ranks=1, expected_layers=1)


def test_workloads_include_identical_rows_and_variable_lengths():
    prompts = [list(range(150 + i)) for i in range(8)]
    workloads = gate.cases(prompts)
    assert [len(x[2]) for x in workloads] == [8, 3, 1, 3]
    assert workloads[1][2][0] == workloads[2][2][0] == workloads[0][2][0]
    assert list(map(len, workloads[3][2])) == [17, 63, 129]
