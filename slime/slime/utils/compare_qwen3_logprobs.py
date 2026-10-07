#!/usr/bin/env python3
"""Compare Qwen3 rollout and Megatron target-token log-probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load_samples(path: Path) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise TypeError(f"{path} must contain a samples list")
    if not all(isinstance(sample, dict) for sample in payload["samples"]):
        raise TypeError(f"{path} samples must be dictionaries")
    return payload["samples"]


def _tensor(value, *, dtype: torch.dtype, context: str) -> torch.Tensor:
    try:
        return torch.as_tensor(value, dtype=dtype).reshape(-1)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{context} must be tensor-like") from error


def _train_samples_by_position(samples: list[dict], path: Path) -> dict[int, dict]:
    result = {}
    for sample_index, sample in enumerate(samples):
        position = sample.get("rollout_position")
        if position is None:
            raise KeyError(f"{path}: train sample {sample_index} has no rollout_position")
        position = int(position)
        if position in result:
            raise ValueError(f"{path}: duplicate rollout_position {position}")
        result[position] = sample
    return result


def compare_log_probs(rollout_path: Path, train_path: Path) -> dict:
    rollout_samples = _load_samples(rollout_path)
    train_samples = _train_samples_by_position(_load_samples(train_path), train_path)
    expected_positions = set(range(len(rollout_samples)))
    if set(train_samples) != expected_positions:
        missing = sorted(expected_positions - set(train_samples))
        extra = sorted(set(train_samples) - expected_positions)
        raise ValueError(
            "train dump rollout positions do not match rollout dump: "
            f"missing={missing}, extra={extra}"
        )

    total_abs = 0.0
    max_abs = 0.0
    ratio_total_abs_from_one = 0.0
    ratio_max_abs_from_one = 0.0
    low_var_kl_total_abs = 0.0
    low_var_kl_max_abs = 0.0
    tokens = 0
    mismatched = 0
    per_sample = []
    rollout_nonfinite = 0
    train_nonfinite = 0
    for position, rollout_sample in enumerate(rollout_samples):
        train_sample = train_samples[position]
        rollout_tokens = _tensor(
            rollout_sample.get("tokens"), dtype=torch.int64,
            context=f"{rollout_path}: sample {position} tokens",
        )
        train_tokens = _tensor(
            train_sample.get("tokens"), dtype=torch.int64,
            context=f"{train_path}: sample {position} tokens",
        )
        if not torch.equal(rollout_tokens, train_tokens):
            raise ValueError(f"sample {position} token sequence differs between dumps")

        rollout_log_probs = _tensor(
            rollout_sample.get("rollout_log_probs"), dtype=torch.float32,
            context=f"{rollout_path}: sample {position} rollout_log_probs",
        )
        train_log_probs = _tensor(
            train_sample.get("log_probs"), dtype=torch.float32,
            context=f"{train_path}: sample {position} log_probs",
        )
        response_length = int(rollout_sample.get("response_length", -1))
        if rollout_log_probs.numel() != response_length:
            raise ValueError(
                f"rollout sample {position} has {rollout_log_probs.numel()} "
                f"log-probs for response_length={response_length}"
            )
        if train_log_probs.shape != rollout_log_probs.shape:
            raise ValueError(
                f"sample {position} log-prob shapes differ: "
                f"rollout={tuple(rollout_log_probs.shape)}, "
                f"train={tuple(train_log_probs.shape)}"
            )

        finite = torch.isfinite(train_log_probs) & torch.isfinite(rollout_log_probs)
        rollout_nonfinite += int(torch.count_nonzero(~torch.isfinite(rollout_log_probs)).item())
        train_nonfinite += int(torch.count_nonzero(~torch.isfinite(train_log_probs)).item())
        log_ratio = train_log_probs - rollout_log_probs
        safe_log_ratio = torch.where(finite, log_ratio, torch.zeros_like(log_ratio))
        ratio_difference = (safe_log_ratio.exp() - 1).abs()
        low_var_kl = ((-safe_log_ratio).exp() - 1 + safe_log_ratio).abs()
        nonfinite_metric = torch.full_like(log_ratio, float("inf"))
        ratio_difference = torch.where(finite, ratio_difference, nonfinite_metric)
        low_var_kl = torch.where(finite, low_var_kl, nonfinite_metric)
        if ratio_difference.numel():
            ratio_max_abs_from_one = max(ratio_max_abs_from_one, float(ratio_difference.max().item()))
            low_var_kl_max_abs = max(low_var_kl_max_abs, float(low_var_kl.max().item()))
        ratio_total_abs_from_one += float(ratio_difference.sum().item())
        low_var_kl_total_abs += float(low_var_kl.sum().item())
        difference = (train_log_probs - rollout_log_probs).abs()
        difference = torch.where(
            finite, difference, torch.full_like(difference, float("inf"))
        )
        sample_max = float(difference.max().item()) if difference.numel() else 0.0
        sample_sum = float(difference.sum().item())
        sample_mismatched = int(torch.count_nonzero(difference).item())
        max_abs = max(max_abs, sample_max)
        total_abs += sample_sum
        tokens += difference.numel()
        mismatched += sample_mismatched
        per_sample.append(
            {
                "rollout_position": position,
                "tokens": difference.numel(),
                "max_abs": sample_max,
                "mean_abs": sample_sum / difference.numel()
                if difference.numel() else float("nan"),
                "mismatched_tokens": sample_mismatched,
            }
        )

    return {
        "samples": len(rollout_samples),
        "tokens": tokens,
        "max_abs": max_abs,
        "mean_abs": total_abs / tokens if tokens else float("nan"),
        "mismatched_tokens": mismatched,
        "rollout_nonfinite": rollout_nonfinite,
        "train_nonfinite": train_nonfinite,
        "exact": tokens > 0 and mismatched == 0,
        "importance_ratio": {
            "definition": "exp(train_log_prob - rollout_log_prob)",
            "max_abs_from_one": ratio_max_abs_from_one,
            "mean_abs_from_one": ratio_total_abs_from_one / tokens if tokens else float("nan"),
            "exact_one": tokens > 0 and ratio_max_abs_from_one == 0,
        },
        "ppo_kl": {
            "definition": "rollout_log_prob - train_log_prob",
            "max_abs": max_abs,
            "mean_abs": total_abs / tokens if tokens else float("nan"),
            "exact_zero": tokens > 0 and mismatched == 0,
        },
        "low_var_kl": {
            "max_abs": low_var_kl_max_abs,
            "mean_abs": low_var_kl_total_abs / tokens if tokens else float("nan"),
            "exact_zero": tokens > 0 and low_var_kl_max_abs == 0,
        },
        "per_sample": per_sample,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-data", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--max-logprob-diff", type=float, default=0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = compare_log_probs(args.rollout_data, args.train_data)
    print(
        f"log_probs samples={stats['samples']} tokens={stats['tokens']} "
        f"max_abs={stats['max_abs']:.12g} "
        f"mean_abs={stats['mean_abs']:.12g} "
        f"mismatched_tokens={stats['mismatched_tokens']}"
    )
    result = {
        "rollout_data": str(args.rollout_data),
        "train_data": str(args.train_data),
        "max_logprob_diff": args.max_logprob_diff,
        "log_probs": stats,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if not stats["exact"] or stats["max_abs"] > args.max_logprob_diff:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
