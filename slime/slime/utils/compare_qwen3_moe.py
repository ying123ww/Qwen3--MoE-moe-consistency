#!/usr/bin/env python3
"""Compare matching Qwen3 Megatron and SGLang MoE block inputs/outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from slime.utils.compare_glm52_layerwise import (
    _TP_RANK_RE,
    _find_suffix,
    _load_records,
    _sglang_dump_files,
    _sglang_segments,
    map_requests_to_train_sequences,
)
from slime.utils.compare_qwen3_router import _router_rows


@dataclass
class TrainMoeSequence:
    tokens: torch.Tensor
    layers: dict[int, dict[str, torch.Tensor]]
    source: str


def load_train_moe_sequences(
    megatron_dir: Path,
    selected_layers: set[int],
) -> list[TrainMoeSequence]:
    dump_files = sorted(megatron_dir.glob("rank*/actor_Pass*.pt"))
    if not dump_files:
        raise FileNotFoundError(f"No Megatron layer dumps found under {megatron_dir}")

    sequences = []
    for dump_file in dump_files:
        for record_index, record in enumerate(_load_records(dump_file)):
            tokens = record["input_ids"].reshape(-1).to(torch.int64)
            layer_rows = {}
            for layer_id in selected_layers:
                suffix = f"decoder.layers.{layer_id}.mlp"
                layer_rows[layer_id] = {
                    "input": _router_rows(
                        _find_suffix(record["module_inputs"], suffix),
                        tokens.numel(),
                        f"{dump_file}:layer{layer_id}:mlp_input",
                    ),
                    "output": _router_rows(
                        _find_suffix(record["modules"], suffix),
                        tokens.numel(),
                        f"{dump_file}:layer{layer_id}:mlp_output",
                    ),
                }

            cu_seqlens = record["cu_seqlens"].reshape(-1).tolist()
            for sequence_index, (start, end) in enumerate(
                zip(cu_seqlens, cu_seqlens[1:], strict=False)
            ):
                sequence_tokens = tokens[start:end]
                if sequence_tokens.numel() == 0 or torch.count_nonzero(sequence_tokens) == 0:
                    continue
                sequences.append(
                    TrainMoeSequence(
                        tokens=sequence_tokens,
                        layers={
                            layer_id: {
                                field: value[start:end]
                                for field, value in values.items()
                            }
                            for layer_id, values in layer_rows.items()
                        },
                        source=(
                            f"{dump_file}:record{record_index}:"
                            f"sequence{sequence_index}"
                        ),
                    )
                )
    if not sequences:
        raise RuntimeError("Megatron dumps contained no non-padding MoE sequences")
    return sequences


def compare_moe_outputs(
    dump_files: list[Path],
    train_sequences: list[TrainMoeSequence],
    request_mapping: dict[str, int],
    selected_layers: set[int],
    sglang_attn_tp_size: int,
):
    if sglang_attn_tp_size <= 0:
        raise ValueError(
            f"SGLang attention TP size must be positive, got {sglang_attn_tp_size}"
        )
    stats = {
        layer_id: {
            "tokens": 0,
            "input": {"max_abs": 0.0, "sum_abs": 0.0, "numel": 0},
            "output": {"max_abs": 0.0, "sum_abs": 0.0, "numel": 0},
        }
        for layer_id in selected_layers
    }
    compared: set[tuple[str, int, int]] = set()

    for dump_file in dump_files:
        for record in _load_records(dump_file):
            input_ids, positions, segments = _sglang_segments(record)
            if not segments:
                continue
            if input_ids.numel() % sglang_attn_tp_size:
                raise ValueError(
                    f"{dump_file} has {input_ids.numel()} rows, not divisible by "
                    f"attention TP size {sglang_attn_tp_size}"
                )
            local_num_tokens = input_ids.numel() // sglang_attn_tp_size
            match = _TP_RANK_RE.search(dump_file.parent.name)
            if match is None:
                raise ValueError(
                    f"Cannot determine SGLang TP rank from {dump_file.parent.name!r}"
                )
            attn_tp_rank = int(match.group(1)) % sglang_attn_tp_size
            local_start = attn_tp_rank * local_num_tokens
            local_end = local_start + local_num_tokens

            rollout_layers = {}
            for layer_id in selected_layers:
                suffix = f".layers.{layer_id}.mlp"
                rollout_layers[layer_id] = {
                    "input": _router_rows(
                        _find_suffix(record, suffix + ".input"),
                        local_num_tokens,
                        f"{dump_file}:layer{layer_id}:mlp_input",
                    ),
                    "output": _router_rows(
                        _find_suffix(record, suffix),
                        local_num_tokens,
                        f"{dump_file}:layer{layer_id}:mlp_output",
                    ),
                }

            for rid, token_slice in segments:
                segment_start = max(token_slice.start, local_start)
                segment_end = min(token_slice.stop, local_end)
                if segment_start >= segment_end:
                    continue
                train_sequence = train_sequences[request_mapping[rid]]
                segment_positions = positions[segment_start:segment_end]
                for layer_id in selected_layers:
                    keep = torch.tensor(
                        [
                            int(position) < train_sequence.tokens.numel() - 1
                            and (rid, int(position), layer_id) not in compared
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
                    layer_stats = stats[layer_id]
                    layer_stats["tokens"] += kept_positions.numel()
                    local_slice = slice(
                        segment_start - local_start,
                        segment_end - local_start,
                    )
                    for field in ("input", "output"):
                        rollout_value = rollout_layers[layer_id][field][local_slice][keep]
                        train_value = train_sequence.layers[layer_id][field][
                            kept_positions
                        ]
                        difference = (
                            rollout_value.float() - train_value.float()
                        ).abs()
                        field_stats = layer_stats[field]
                        field_stats["max_abs"] = max(
                            field_stats["max_abs"],
                            float(difference.max().item()),
                        )
                        field_stats["sum_abs"] += float(difference.sum().item())
                        field_stats["numel"] += difference.numel()
                    for position in kept_positions.tolist():
                        compared.add((rid, int(position), layer_id))

    for layer_stats in stats.values():
        for field in ("input", "output"):
            field_stats = layer_stats[field]
            field_stats["mean_abs"] = (
                field_stats.pop("sum_abs") / field_stats["numel"]
                if field_stats["numel"]
                else float("nan")
            )
    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--sglang-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--sglang-attn-tp-size", type=int, default=1)
    parser.add_argument("--max-input-diff", type=float, default=0.0)
    parser.add_argument("--max-output-diff", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_threads <= 0:
        raise ValueError(f"--num-threads must be positive, got {args.num_threads}")
    torch.set_num_threads(args.num_threads)
    selected_layers = set(args.layers)
    train_sequences = load_train_moe_sequences(args.megatron_dir, selected_layers)
    dump_files = _sglang_dump_files(args.sglang_dir)
    request_mapping = map_requests_to_train_sequences(dump_files, train_sequences)
    stats = compare_moe_outputs(
        dump_files,
        train_sequences,
        request_mapping,
        selected_layers,
        args.sglang_attn_tp_size,
    )

    failed = False
    for layer_id in sorted(stats):
        layer_stats = stats[layer_id]
        print(
            f"layer={layer_id} tokens={layer_stats['tokens']} "
            f"mlp_input_max_abs={layer_stats['input']['max_abs']:.12g} "
            f"mlp_output_max_abs={layer_stats['output']['max_abs']:.12g}"
        )
        if (
            not layer_stats["tokens"]
            or layer_stats["input"]["max_abs"] > args.max_input_diff
            or layer_stats["output"]["max_abs"] > args.max_output_diff
        ):
            failed = True

    result = {
        "max_input_diff": args.max_input_diff,
        "max_output_diff": args.max_output_diff,
        "sglang_attn_tp_size": args.sglang_attn_tp_size,
        "request_mapping": request_mapping,
        "layers": stats,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
