import sys

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    TritonRunnerOutput,
    post_permute_triton_to_deepep_ll,
    post_permute_triton_to_deepep_normal,
    pre_permute_deepep_ll_to_triton,
    pre_permute_deepep_normal_to_triton,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLDispatchOutput,
    DeepEPNormalDispatchOutput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def test_deepep_normal_triton_adapter_converts_expert_ids(monkeypatch):
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe

    captured = {}

    def fake_prepare(*args, **kwargs):
        captured["topk_ids"] = args[3]
        return (
            {"BLOCK_SIZE_M": 16},
            None,
            False,
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )

    monkeypatch.setattr(fused_moe, "_prepare_fused_moe_run", fake_prepare)

    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64)
    topk_weights = torch.rand(2, 2, dtype=torch.float32)
    dispatch_output = DeepEPNormalDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_recv_tokens_per_expert=[2, 1],
    )
    quant_info = TritonMoeQuantInfo(
        w13_weight=torch.empty(2, 8, 4, dtype=torch.bfloat16),
        w2_weight=torch.empty(2, 4, 4, dtype=torch.bfloat16),
    )
    running_state = {}

    runner_input = pre_permute_deepep_normal_to_triton(
        dispatch_output, quant_info, MoeRunnerConfig(), running_state
    )

    assert runner_input.topk_ids.dtype == torch.int32
    assert captured["topk_ids"].dtype == torch.int32
    assert torch.equal(runner_input.topk_ids, topk_ids)
    assert running_state["topk_ids"] is topk_ids

    combine_input = post_permute_triton_to_deepep_normal(
        TritonRunnerOutput(hidden_states=hidden_states),
        quant_info,
        MoeRunnerConfig(),
        running_state,
    )
    assert combine_input.hidden_states is hidden_states
    assert combine_input.topk_ids is topk_ids
    assert combine_input.topk_weights is topk_weights


def test_deepep_normal_triton_adapter_rejects_quantized_dispatch():
    dispatch_output = DeepEPNormalDispatchOutput(
        hidden_states=torch.empty(1, 4, dtype=torch.uint8),
        hidden_states_scale=torch.empty(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int64),
        topk_weights=torch.ones(1, 1),
        num_recv_tokens_per_expert=[1],
    )
    quant_info = TritonMoeQuantInfo(
        w13_weight=torch.empty(1, 8, 4, dtype=torch.bfloat16),
        w2_weight=torch.empty(1, 4, 4, dtype=torch.bfloat16),
    )

    with pytest.raises(ValueError, match="BF16 dispatch output"):
        pre_permute_deepep_normal_to_triton(
            dispatch_output, quant_info, MoeRunnerConfig(), {}
        )


def test_deepep_ll_triton_adapter_compacts_and_restores_slab(monkeypatch):
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe

    def fake_prepare(*args, **kwargs):
        return (
            {"BLOCK_SIZE_M": 16},
            None,
            False,
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )

    monkeypatch.setattr(fused_moe, "_prepare_fused_moe_run", fake_prepare)

    hidden_states = torch.arange(3 * 4 * 2, dtype=torch.bfloat16).view(3, 4, 2)
    topk_ids = torch.tensor([[0, 5], [1, 4]], dtype=torch.int64)
    topk_weights = torch.rand(2, 2, dtype=torch.float32)
    dispatch_output = DeepEPLLDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        masked_m=torch.tensor([2, 0, 3], dtype=torch.int32),
        expected_m=2,
    )
    quant_info = TritonMoeQuantInfo(
        w13_weight=torch.empty(3, 8, 2, dtype=torch.bfloat16),
        w2_weight=torch.empty(3, 2, 4, dtype=torch.bfloat16),
    )
    running_state = {}

    runner_input = pre_permute_deepep_ll_to_triton(
        dispatch_output, quant_info, MoeRunnerConfig(), running_state
    )

    expected_rows = torch.cat((hidden_states[0, :2], hidden_states[2, :3]))
    assert torch.equal(runner_input.hidden_states, expected_rows)
    assert torch.equal(
        runner_input.topk_ids,
        torch.tensor([[0], [0], [2], [2], [2]], dtype=torch.int32),
    )
    assert torch.equal(runner_input.topk_weights, torch.ones(5, 1))

    combine_input = post_permute_triton_to_deepep_ll(
        TritonRunnerOutput(hidden_states=runner_input.hidden_states + 100),
        quant_info,
        MoeRunnerConfig(),
        running_state,
    )
    expected_slab = torch.zeros_like(hidden_states)
    expected_slab[0, :2] = hidden_states[0, :2] + 100
    expected_slab[2, :3] = hidden_states[2, :3] + 100
    assert torch.equal(combine_input.hidden_states, expected_slab)
    assert combine_input.topk_ids is topk_ids
    assert combine_input.topk_weights is topk_weights


def test_deepep_ll_triton_adapter_keeps_dummy_for_empty_rank(monkeypatch):
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe

    def fake_prepare(*args, **kwargs):
        return ({}, None, False, torch.empty(0), torch.empty(0), torch.zeros(1))

    monkeypatch.setattr(fused_moe, "_prepare_fused_moe_run", fake_prepare)
    hidden_states = torch.empty(2, 4, 8, dtype=torch.bfloat16)
    dispatch_output = DeepEPLLDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_ids=torch.zeros(1, 2, dtype=torch.int64),
        topk_weights=torch.ones(1, 2),
        masked_m=torch.zeros(2, dtype=torch.int32),
        expected_m=0,
    )
    quant_info = TritonMoeQuantInfo(
        w13_weight=torch.empty(2, 16, 8, dtype=torch.bfloat16),
        w2_weight=torch.empty(2, 8, 8, dtype=torch.bfloat16),
    )
    running_state = {}

    runner_input = pre_permute_deepep_ll_to_triton(
        dispatch_output, quant_info, MoeRunnerConfig(), running_state
    )
    assert runner_input.hidden_states.shape == (1, 8)
    assert runner_input.hidden_states.count_nonzero() == 0
    assert runner_input.topk_weights.item() == 0

    combine_input = post_permute_triton_to_deepep_ll(
        TritonRunnerOutput(hidden_states=torch.ones_like(runner_input.hidden_states)),
        quant_info,
        MoeRunnerConfig(),
        running_state,
    )
    assert torch.equal(combine_input.hidden_states, torch.zeros_like(hidden_states))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
