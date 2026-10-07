import sys
import types
from argparse import Namespace

import pytest
import torch


NUM_GPUS = 0


def test_value_and_greedy_logprob_temperature_semantics(monkeypatch):
    previous_loss = sys.modules.pop("slime.backends.megatron_utils.loss", None)
    previous_cp_utils = sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
    )
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = mpu_stub
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)

    try:
        from slime.backends.megatron_utils.loss import _apply_rollout_temperature, get_responses, get_values

        args = Namespace(rollout_temperature=0.5, allgather_cp=False)
        logits = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.float32)
        tokens = [torch.tensor([10, 11, 12, 13], dtype=torch.long)]

        _, result = get_values(
            logits,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=[4],
            response_lengths=[2],
        )

        torch.testing.assert_close(result["values"][0], torch.tensor([2.0, 3.0]))

        greedy_args = Namespace(rollout_temperature=0.0)
        [(response_logits, response_tokens)] = list(
            get_responses(
                logits,
                args=greedy_args,
                unconcat_tokens=tokens,
                total_lengths=[4],
                response_lengths=[2],
            )
        )
        torch.testing.assert_close(response_logits, torch.tensor([[2.0], [3.0]]))
        torch.testing.assert_close(response_tokens, torch.tensor([12, 13]))

        qwen_logits = torch.tensor([[1.0, 2.0]], dtype=torch.float32, requires_grad=True)
        qwen_args = Namespace(rollout_temperature=0.6, qwen3_alignment_layers=[0])
        scaled = _apply_rollout_temperature(qwen_logits, qwen_args)
        expected = qwen_logits / qwen_logits.new_tensor(0.6)
        torch.testing.assert_close(scaled, expected, rtol=0, atol=0)
        scaled.sum().backward()
        torch.testing.assert_close(
            qwen_logits.grad,
            torch.full_like(qwen_logits, 1.0 / 0.6),
        )
    finally:
        if previous_loss is None:
            sys.modules.pop("slime.backends.megatron_utils.loss", None)
        else:
            sys.modules["slime.backends.megatron_utils.loss"] = previous_loss
        if previous_cp_utils is None:
            sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
        else:
            sys.modules["slime.backends.megatron_utils.cp_utils"] = previous_cp_utils


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
