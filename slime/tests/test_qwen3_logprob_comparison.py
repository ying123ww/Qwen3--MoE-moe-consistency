from pathlib import Path

import pytest
import torch

from slime.utils.compare_qwen3_logprobs import compare_log_probs


def _write_dumps(tmp_path: Path, train_log_probs=None):
    rollout_path = tmp_path / "rollout.pt"
    train_path = tmp_path / "train.pt"
    rollout_samples = [
        {
            "tokens": [10, 11, 12],
            "response_length": 2,
            "rollout_log_probs": [-0.25, -0.5],
        },
        {
            "tokens": [20, 21],
            "response_length": 1,
            "rollout_log_probs": [-0.75],
        },
    ]
    if train_log_probs is None:
        train_log_probs = [[-0.25, -0.5], [-0.75]]
    train_samples = [
        {
            "rollout_position": 1,
            "tokens": torch.tensor([20, 21]),
            "log_probs": torch.tensor(train_log_probs[1]),
        },
        {
            "rollout_position": 0,
            "tokens": torch.tensor([10, 11, 12]),
            "log_probs": torch.tensor(train_log_probs[0]),
        },
    ]
    torch.save({"rollout_id": 1, "samples": rollout_samples}, rollout_path)
    torch.save(
        {"format_version": 2, "rollout_id": 1, "samples": train_samples},
        train_path,
    )
    return rollout_path, train_path


def test_logprob_comparison_maps_rollout_positions(tmp_path):
    rollout_path, train_path = _write_dumps(tmp_path)

    stats = compare_log_probs(rollout_path, train_path)

    assert stats["samples"] == 2
    assert stats["tokens"] == 3
    assert stats["max_abs"] == 0
    assert stats["mean_abs"] == 0
    assert stats["mismatched_tokens"] == 0
    assert stats["exact"]
    assert stats["importance_ratio"]["exact_one"]
    assert stats["ppo_kl"]["exact_zero"]
    assert stats["low_var_kl"]["exact_zero"]


def test_logprob_comparison_reports_difference(tmp_path):
    rollout_path, train_path = _write_dumps(
        tmp_path, train_log_probs=[[-0.25, -0.375], [-0.75]]
    )

    stats = compare_log_probs(rollout_path, train_path)

    assert stats["max_abs"] == pytest.approx(0.125)
    assert stats["mean_abs"] == pytest.approx(0.125 / 3)
    assert stats["mismatched_tokens"] == 1
    assert not stats["exact"]
    assert stats["importance_ratio"]["max_abs_from_one"] == pytest.approx(
        torch.exp(torch.tensor(0.125)).item() - 1
    )
    assert stats["low_var_kl"]["max_abs"] == pytest.approx(torch.exp(torch.tensor(-0.125)).item() - 1 + 0.125)


def test_logprob_comparison_rejects_nonfinite_values(tmp_path):
    rollout_path, train_path = _write_dumps(
        tmp_path, train_log_probs=[[float("nan"), -0.5], [-0.75]]
    )

    stats = compare_log_probs(rollout_path, train_path)

    assert stats["max_abs"] == float("inf")
    assert stats["mismatched_tokens"] == 1
    assert stats["train_nonfinite"] == 1
    assert not stats["exact"]


def test_logprob_comparison_rejects_token_mismatch(tmp_path):
    rollout_path, train_path = _write_dumps(tmp_path)
    train = torch.load(train_path, weights_only=False)
    train["samples"][0]["tokens"][0] = 99
    torch.save(train, train_path)
    with pytest.raises(ValueError, match="token sequence differs"):
        compare_log_probs(rollout_path, train_path)
