"""CPU spies run production method bodies without importing GPU SGLang modules."""
import ast
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
QWEN = ROOT / 'experiments/sglang-main-sm80/python/sglang/srt/models/qwen3_moe.py'
TOPK = ROOT / 'experiments/sglang-main-sm80/python/sglang/srt/layers/moe/topk.py'


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


def setup_method(*, allowed=True, available=True, deterministic=True):
    captures, native_calls = [], []
    cap = SimpleNamespace(capture=lambda **kwargs: captures.append(kwargs))
    namespace = {'torch': torch, 'get_global_experts_capturer': lambda: cap if available else None,
                 'get_exec': lambda: SimpleNamespace(deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)),
                 'StandardTopKOutput': namedtuple('StandardTopKOutput', 'topk_weights topk_ids router_logits')}
    load_function(TOPK, 'capture_routed_experts_if_allowed', namespace)
    method = load_function(QWEN, '_select_experts', namespace)
    native_result = object()

    class FakeTopK:
        topk_config = SimpleNamespace(allow_routed_experts_capture=allowed)

        def __call__(self, *args, **kwargs):
            native_calls.append((args, kwargs))
            return native_result

    block = SimpleNamespace(top_k=2, layer_id=7, topk=FakeTopK())
    logits = torch.tensor([[0.1, 2.1, -0.3, 4.2], [3.5, 0.2, 5.1, 0.7]])
    hidden = torch.ones(2, 3)
    output = method(block, hidden, logits, num_token_non_padded=2)
    return output, captures, native_calls, native_result, logits


def test_production_deterministic_selection_captures_exact_returned_ids():
    output, captures, _, _, _ = setup_method()
    assert len(captures) == 1
    assert captures[0]['layer_id'] == 7
    assert captures[0]['topk_indices'] is output.topk_ids
    assert output.topk_ids.dtype == torch.int32


@pytest.mark.parametrize('allowed,available', [(True, True), (False, True), (True, False)])
def test_capture_preserves_math_and_respects_capture_gates(allowed, available):
    output, captures, _, _, logits = setup_method(allowed=allowed, available=available)
    expected_ids = torch.topk(logits, 2, dim=-1, sorted=False).indices
    expected_weights = torch.softmax(logits.gather(-1, expected_ids), dim=-1, dtype=torch.float32)
    torch.testing.assert_close(output.topk_weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(output.topk_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    assert len(captures) == int(allowed and available)
    if captures:
        assert captures[0]['topk_indices'] is output.topk_ids
        assert captures[0]['layer_id'] == 7
        assert output.topk_ids.dtype == torch.int32


def test_nondeterministic_branch_still_delegates_to_topk_once():
    output, captures, native_calls, native_result, _ = setup_method(deterministic=False)
    assert output is native_result
    assert len(native_calls) == 1
    assert captures == []
