#!/usr/bin/env python3
"""Compare Qwen3 Megatron and SGLang final RMSNorm outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from slime.utils.compare_glm52_layerwise import (
    _TP_RANK_RE,
    _as_tensor,
    _load_records,
    _sglang_dump_files,
    _sglang_segments,
    _token_rows,
    map_requests_to_train_sequences,
)


@dataclass
class TrainTailSequence:
    tokens: torch.Tensor
    final_norm: torch.Tensor
    source: str


def _find_unique_suffix(values, suffix: str, context: str):
    matches = [
        value for name, value in values.items() if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise KeyError(
            f"{context}: expected one key ending in {suffix!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def load_train_tail_sequences(dump_dir: Path) -> list[TrainTailSequence]:
    dump_files = sorted(dump_dir.glob("rank*/actor_Pass*.pt"))
    if not dump_files:
        raise FileNotFoundError(f"No Megatron tail dumps found under {dump_dir}")

    sequences = []
    for dump_file in dump_files:
        for record in _load_records(dump_file):
            tokens = record["input_ids"].reshape(-1).to(torch.int64)
            cu_seqlens = record["cu_seqlens"].reshape(-1).tolist()
            final_norm = _find_unique_suffix(
                record.get("modules", {}),
                "decoder.final_layernorm",
                str(dump_file),
            )
            final_norm_rows = _token_rows(
                final_norm, len(tokens), f"{dump_file}:final_norm"
            )
            for sequence_index, (start, end) in enumerate(
                zip(cu_seqlens, cu_seqlens[1:], strict=False)
            ):
                sequence_tokens = tokens[start:end]
                if (
                    sequence_tokens.numel() == 0
                    or torch.count_nonzero(sequence_tokens) == 0
                ):
                    continue
                sequences.append(
                    TrainTailSequence(
                        tokens=sequence_tokens,
                        final_norm=final_norm_rows[start:end],
                        source=f"{dump_file}:sequence{sequence_index}",
                    )
                )
    if not sequences:
        raise RuntimeError("Megatron dumps contained no non-padding sequences")
    return sequences


def _sglang_final_norm(record, context: str):
    matches = [
        value
        for name, value in record.items()
        if name == "model.norm" or name.endswith(".model.norm")
    ]
    if len(matches) != 1:
        raise KeyError(
            f"{context}: expected one SGLang model.norm output, "
            f"found {len(matches)}"
        )
    return matches[0]


def _local_sglang_rows(
    value,
    num_tokens: int,
    local_start: int,
    local_end: int,
    context: str,
) -> torch.Tensor:
    tensor = _as_tensor(value)
    if tensor.ndim < 2:
        raise ValueError(
            f"{context} must have a hidden dimension, got {tensor.shape}"
        )
    rows = tensor.reshape(-1, tensor.shape[-1])
    local_num_tokens = local_end - local_start
    if rows.shape[0] == num_tokens:
        return rows[local_start:local_end]
    if rows.shape[0] == local_num_tokens:
        return rows
    raise ValueError(
        f"{context} has {rows.shape[0]} token rows; expected "
        f"{num_tokens} full or {local_num_tokens} attention-TP-local rows"
    )


def compare_final_norm(
    dump_files: list[Path],
    train_sequences: list[TrainTailSequence],
    request_mapping: dict[str, int],
    attn_tp_size: int = 1,
):
    if attn_tp_size <= 0:
        raise ValueError(
            f"SGLang attention TP size must be positive, got {attn_tp_size}"
        )
    stats = {
        "max_abs": 0.0,
        "sum_abs": 0.0,
        "numel": 0,
        "tokens": 0,
        "mismatched_elements": 0,
    }
    compared: set[tuple[str, int]] = set()

    for dump_file in dump_files:
        for record in _load_records(dump_file):
            input_ids, positions, segments = _sglang_segments(record)
            if not segments:
                continue
            if input_ids.numel() % attn_tp_size:
                raise ValueError(
                    f"{dump_file} has {input_ids.numel()} input rows, which "
                    f"cannot be evenly scattered over attention TP size "
                    f"{attn_tp_size}"
                )
            local_num_tokens = input_ids.numel() // attn_tp_size
            if attn_tp_size == 1:
                attn_tp_rank = 0
            else:
                match = _TP_RANK_RE.search(dump_file.parent.name)
                if match is None:
                    raise ValueError(
                        f"Cannot determine SGLang TP rank from "
                        f"{dump_file.parent.name!r}"
                    )
                attn_tp_rank = int(match.group(1)) % attn_tp_size
            local_start = attn_tp_rank * local_num_tokens
            local_end = local_start + local_num_tokens
            rollout_rows = _local_sglang_rows(
                _sglang_final_norm(record, str(dump_file)),
                input_ids.numel(),
                local_start,
                local_end,
                f"{dump_file}:final_norm",
            )

            for rid, token_slice in segments:
                segment_start = max(token_slice.start, local_start)
                segment_end = min(token_slice.stop, local_end)
                if segment_start >= segment_end:
                    continue
                train_sequence = train_sequences[request_mapping[rid]]
                segment_positions = positions[segment_start:segment_end]
                keep = torch.tensor(
                    [
                        int(position) < train_sequence.tokens.numel() - 1
                        and (rid, int(position)) not in compared
                        for position in segment_positions.tolist()
                    ],
                    dtype=torch.bool,
                )
                if not torch.any(keep):
                    continue
                kept_positions = segment_positions[keep]
                if torch.any(kept_positions < 0) or torch.any(
                    kept_positions >= train_sequence.tokens.numel()
                ):
                    raise IndexError(
                        f"SGLang request {rid} contains positions outside "
                        "its Megatron sequence"
                    )
                rollout_value = rollout_rows[
                    segment_start - local_start : segment_end - local_start
                ][keep]
                train_value = train_sequence.final_norm[kept_positions]
                difference = (
                    rollout_value.float() - train_value.float()
                ).abs()
                stats["max_abs"] = max(
                    stats["max_abs"], float(difference.max().item())
                )
                stats["sum_abs"] += float(difference.sum().item())
                stats["numel"] += difference.numel()
                stats["tokens"] += kept_positions.numel()
                stats["mismatched_elements"] += int(
                    torch.count_nonzero(difference).item()
                )
                for position in kept_positions.tolist():
                    compared.add((rid, int(position)))

    stats["mean_abs"] = (
        stats.pop("sum_abs") / stats["numel"]
        if stats["numel"]
        else float("nan")
    )
    stats["exact"] = (
        stats["mismatched_elements"] == 0 and stats["tokens"] > 0
    )
    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--sglang-dir", type=Path, required=True)
    parser.add_argument("--sglang-attn-tp-size", type=int, default=1)
    parser.add_argument("--max-final-norm-diff", type=float, default=0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_threads <= 0:
        raise ValueError(
            f"--num-threads must be positive, got {args.num_threads}"
        )
    torch.set_num_threads(args.num_threads)
    train_sequences = load_train_tail_sequences(args.megatron_dir)
    dump_files = _sglang_dump_files(args.sglang_dir)
    request_mapping = map_requests_to_train_sequences(
        dump_files, train_sequences
    )
    stats = compare_final_norm(
        dump_files,
        train_sequences,
        request_mapping,
        args.sglang_attn_tp_size,
    )
    print(
        f"final_norm tokens={stats['tokens']} "
        f"max_abs={stats['max_abs']:.12g} "
        f"mean_abs={stats['mean_abs']:.12g} "
        f"mismatched_elements={stats['mismatched_elements']}"
    )
    result = {
        "max_final_norm_diff": args.max_final_norm_diff,
        "sglang_attn_tp_size": args.sglang_attn_tp_size,
        "request_mapping": request_mapping,
        "final_norm": stats,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if not stats["tokens"] or stats["max_abs"] > args.max_final_norm_diff:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
