"""Strict CPU dispatch/audit checks; kernel bits are checked on A100."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import router_tile_exact as gate

HELPER = Path(__file__).resolve().parents[3] / 'experiments/sglang-main-sm80/python/sglang/srt/batch_invariant_ops/fused_rmsnorm_candidate.py'


def helper(monkeypatch, tmp_path, mode='off', audit='1'):
    monkeypatch.setenv('SLIME_QWEN3_RESIDUAL_RMS_MODE', mode)
    monkeypatch.setenv('SLIME_QWEN3_RESIDUAL_RMS_AUDIT', audit)
    monkeypatch.setenv('SLIME_QWEN3_RESIDUAL_RMS_AUDIT_DIR', str(tmp_path))
    spec = importlib.util.spec_from_file_location('isolated_residual_rms', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(m=2):
    tensor = lambda shape: SimpleNamespace(is_cuda=True, device='cuda:0', dtype=torch.bfloat16,
                                           shape=shape, ndim=len(shape), is_contiguous=lambda: True)
    x, residual, weight = tensor((m,2048)), tensor((m,2048)), tensor((2048,))
    module = SimpleNamespace(_slime_qwen3_rms_kind='pre_mlp', _slime_qwen3_rms_layer=0,
                             hidden_size=2048, variance_size_override=None,
                             cast_x_before_out_mul=False, fp32_residual=False,
                             override_orig_dtype=None, weight=weight, variance_epsilon=1e-6)
    return module, x, residual


def test_default_preserves_native(monkeypatch, tmp_path):
    code = helper(monkeypatch,tmp_path)
    expected = object()
    module = SimpleNamespace(forward_native=lambda *args: expected)
    assert code.forward(module,None,None) is expected
    assert not code.ENABLED and not list(tmp_path.iterdir())


@pytest.mark.parametrize('m,accepted', [(1,True),(2,True),(4,True),(5,False)])
def test_small_M_scope(monkeypatch,tmp_path,m,accepted):
    code=helper(monkeypatch,tmp_path)
    module,x,residual=inputs(m)
    assert code.eligible(module,x,residual,None) is accepted
    module.fp32_residual=True
    assert not code.eligible(module,x,residual,None)


def test_shadow_must_audit(monkeypatch,tmp_path):
    with pytest.raises(ValueError,match='shadow requires audit'):
        helper(monkeypatch,tmp_path,'shadow','0')


def test_timing_path_skips_audit(monkeypatch,tmp_path):
    code=helper(monkeypatch,tmp_path,'candidate','0')
    module,x,residual=inputs()
    expected=object()
    monkeypatch.setattr(code,'_candidate',lambda *args: expected)
    def forbidden(*args):
        raise AssertionError('Timing path must skip audit and rank queries')
    monkeypatch.setattr(code,'Path',forbidden)
    monkeypatch.setattr(torch.distributed,'is_initialized',forbidden)
    assert code.forward(module,x,residual) is expected
    assert not code._CALLS


def test_signed_zero_is_a_fatal_shadow_difference(monkeypatch,tmp_path):
    code=helper(monkeypatch,tmp_path,'shadow')
    module,x,residual=inputs()
    x = torch.zeros(2,2048,dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    module.weight = torch.ones(2048,dtype=torch.bfloat16)
    monkeypatch.setattr(code,'eligible',lambda *args: True)
    monkeypatch.setitem(sys.modules,
        'sglang.srt.batch_invariant_ops.batch_invariant_ops',
        SimpleNamespace(mean_dim=lambda value,dim,keepdim: value.mean(dim=dim,keepdim=keepdim)))
    reference=(torch.zeros(2,2048,dtype=torch.bfloat16),torch.zeros(2,2048,dtype=torch.bfloat16))
    actual=(reference[0].clone(),reference[1].clone())
    actual[0][0,0]=-0.
    module.forward_native=lambda *args: reference
    monkeypatch.setattr(code,'_candidate',lambda *args: actual)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda: False)
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda: False)
    with pytest.raises(RuntimeError,match='exact gate FAILED'):
        code.forward(module,x,residual)
    record=json.loads(next(tmp_path.glob('*.jsonl')).read_text())
    assert record['mismatches']==1 and record['compared_elements']==8192

    dump=torch.load(next(tmp_path.glob('failure-*.pt')),weights_only=True)
    assert dump['context']['mismatches']==1
    assert dump['candidate'][0].view(torch.int16)[0,0] != dump['reference'][0].view(torch.int16)[0,0]


def test_norm_audit_requires_both_target_kinds(tmp_path):
    audit=tmp_path/'audit'
    audit.mkdir()
    rows=[dict(mode='shadow',rank=0,layer=l,kind=k,eligible=True,a_shape=[2,2048],
               compared_elements=8192,mismatches=0,nonfinite_elements=0)
          for l,k in ((0,'pre_mlp'),(1,'input'))]
    p=audit/'rank0.jsonl'
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='coverage mismatch'):
        gate.audit_summary(tmp_path,'shadow',expected_ranks=1,expected_layers=2,component='residual-rms')
    rows.append(dict(rows[-1],kind='pre_mlp'))
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert gate.audit_summary(tmp_path,'shadow',expected_ranks=1,expected_layers=2,component='residual-rms')['compared_elements']==24576
