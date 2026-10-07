#!/usr/bin/env python3
"""Compare matching Qwen3 Megatron and SGLang router values."""

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

_UPSTREAM_FIELDS = (
    "layer_input",
    "post_attention_residual",
    "post_attention_norm",
)
_ROUTER_ROW_FIELDS = (*_UPSTREAM_FIELDS, "input", "logits", "topk_ids", "topk_weights")
_LOCAL_ROUTER_ROW_FIELDS = _ROUTER_ROW_FIELDS
_SGLANG_FIELD_SUFFIXES = {
    "layer_input": ".layers.{layer}.input",
    "post_attention_residual": ".layers.{layer}.post_attention_layernorm.output.1",
    "post_attention_norm": ".layers.{layer}.post_attention_layernorm",
    "input": ".layers.{layer}.mlp.router_input",
    "weight": ".layers.{layer}.mlp.router_weight",
    "logits": ".layers.{layer}.mlp.router_logits",
    "topk_ids": ".layers.{layer}.mlp.topk_ids",
    "topk_weights": ".layers.{layer}.mlp.topk_weights",
}


@dataclass
class TrainRouterSequence:
    tokens: torch.Tensor
    routers: dict[int, dict[str, torch.Tensor]]
    source: str


def _router_rows(value, num_tokens: int, context: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{context} must be a tensor")
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0, :]
    if value.ndim != 2:
        raise TypeError(f"{context} must have shape [tokens, features] or [tokens, 1, features]")
    if value.shape[0] != num_tokens:
        raise ValueError(f"{context} has {value.shape[0]} token rows, expected {num_tokens}")
    return value


def _router_weight(value, context: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise TypeError(f"{context} must be a two-dimensional tensor")
    return value


def load_train_router_sequences(
    megatron_dir: Path,
    router_dir: Path,
    selected_layers: set[int],
) -> list[TrainRouterSequence]:
    dump_files = sorted(megatron_dir.glob("rank*/actor_Pass*.pt"))
    if not dump_files:
        raise FileNotFoundError(f"No Megatron layer dumps found under {megatron_dir}")

    sequences = []
    for dump_file in dump_files:
        router_name = dump_file.name.replace("_Pass", "_RouterPass")
        router_file = router_dir / dump_file.parent.name / router_name
        if not router_file.is_file():
            raise FileNotFoundError(f"Missing Megatron router dump {router_file}")
        layer_records = list(_load_records(dump_file))
        router_records = list(_load_records(router_file))
        if len(layer_records) != len(router_records):
            raise ValueError(
                f"Megatron dump record mismatch: {dump_file} has {len(layer_records)}, "
                f"{router_file} has {len(router_records)}"
            )

        for record_index, (layer_record, router_record) in enumerate(
            zip(layer_records, router_records, strict=True)
        ):
            tokens = layer_record["input_ids"].reshape(-1).to(torch.int64)
            router_tokens = router_record["input_ids"].reshape(-1).to(torch.int64)
            if not torch.equal(tokens, router_tokens):
                raise ValueError(f"Megatron input IDs differ between {dump_file} and {router_file}")
            router_layers = router_record["routers"]
            missing = selected_layers - set(router_layers)
            if missing:
                raise KeyError(f"{router_file} is missing router layers {sorted(missing)}")
            rows = {}
            for layer_id in selected_layers:
                layer_values = dict(router_layers[layer_id])
                layer_values.update(
                    {
                        "layer_input": layer_record["layer_inputs"][layer_id],
                        "post_attention_residual": _find_suffix(
                            layer_record["module_inputs"],
                            f"decoder.layers.{layer_id}.pre_mlp_layernorm",
                        ),
                        "post_attention_norm": _find_suffix(
                            layer_record["modules"],
                            f"decoder.layers.{layer_id}.pre_mlp_layernorm",
                        ),
                    }
                )
                rows[layer_id] = {
                    **{
                        field: _router_rows(
                            layer_values[field],
                            tokens.numel(),
                            f"{dump_file}:layer{layer_id}:{field}",
                        )
                        for field in _ROUTER_ROW_FIELDS
                    },
                    "weight": _router_weight(
                        layer_values["weight"],
                        f"{router_file}:layer{layer_id}:weight",
                    ),
                }

            cu_seqlens = layer_record["cu_seqlens"].reshape(-1).tolist()
            for sequence_index, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:], strict=False)):
                sequence_tokens = tokens[start:end]
                if sequence_tokens.numel() == 0 or torch.count_nonzero(sequence_tokens) == 0:
                    continue
                sequences.append(
                    TrainRouterSequence(
                        tokens=sequence_tokens,
                        routers={
                            layer_id: {
                                field: value if field == "weight" else value[start:end]
                                for field, value in layer_rows.items()
                            }
                            for layer_id, layer_rows in rows.items()
                        },
                        source=f"{dump_file}:record{record_index}:sequence{sequence_index}",
                    )
                )
    if not sequences:
        raise RuntimeError("Megatron dumps contained no non-padding router sequences")
    return sequences


def compare_router_outputs(
    dump_files: list[Path],
    train_sequences: list[TrainRouterSequence],
    request_mapping: dict[str, int],
    selected_layers: set[int],
    sglang_attn_tp_size: int = 1,
):
    if sglang_attn_tp_size <= 0:
        raise ValueError(f"SGLang attention TP size must be positive, got {sglang_attn_tp_size}")
    stats = {
        layer_id: {
            "tokens": 0,
            **{
                field: {"max_abs": 0.0, "sum_abs": 0.0, "numel": 0}
                for field in _UPSTREAM_FIELDS
            },
            "input": {"max_abs": 0.0, "sum_abs": 0.0, "numel": 0},
            "logits": {"max_abs": 0.0, "sum_abs": 0.0, "numel": 0},
            "weight": {
                "exact": True,
                "comparisons": 0,
                "dtype_mismatches": 0,
                "mismatched_elements": 0,
                "max_abs": 0.0,
                "numel": 0,
            },
            "topk_weights": {
                "max_abs": 0.0,
                "sum_abs": 0.0,
                "numel": 0,
                "matched_routes": 0,
                "unmatched_routes": 0,
            },
            "topk_ids": {
                "mismatched_elements": 0,
                "mismatched_tokens": 0,
                "set_mismatched_tokens": 0,
                "numel": 0,
            },
        }
        for layer_id in selected_layers
    }
    compared: set[tuple[str, int, int]] = set()
    compared_weights: set[tuple[str, int]] = set()

    for dump_file in dump_files:
        for record in _load_records(dump_file):
            input_ids, positions, segments = _sglang_segments(record)
            if not segments:
                continue
            if input_ids.numel() % sglang_attn_tp_size:
                raise ValueError(
                    f"{dump_file} has {input_ids.numel()} input rows, which cannot be evenly "
                    f"scattered over attention TP size {sglang_attn_tp_size}"
                )
            local_num_tokens = input_ids.numel() // sglang_attn_tp_size
            rollout_routers = {}
            for layer_id in selected_layers:
                values = {
                    **{
                        field: _router_rows(
                            _find_suffix(
                                record,
                                _SGLANG_FIELD_SUFFIXES[field].format(layer=layer_id),
                            ),
                            local_num_tokens,
                            f"{dump_file}:layer{layer_id}:{field}",
                        )
                        for field in _LOCAL_ROUTER_ROW_FIELDS
                    },
                    "weight": _router_weight(
                        _find_suffix(
                            record,
                            _SGLANG_FIELD_SUFFIXES["weight"].format(layer=layer_id),
                        ),
                        f"{dump_file}:layer{layer_id}:weight",
                    ),
                }
                if values["logits"].dtype != torch.float32 or values["topk_weights"].dtype != torch.float32:
                    raise TypeError(f"{dump_file}:layer{layer_id} router logits/weights must be FP32")
                rollout_routers[layer_id] = values

            if sglang_attn_tp_size == 1:
                attn_tp_rank = 0
            else:
                match = _TP_RANK_RE.search(dump_file.parent.name)
                if match is None:
                    raise ValueError(f"Cannot determine SGLang TP rank from {dump_file.parent.name!r}")
                attn_tp_rank = int(match.group(1)) % sglang_attn_tp_size
            local_start = attn_tp_rank * local_num_tokens
            local_end = local_start + local_num_tokens

            for rid, token_slice in segments:
                segment_start = max(token_slice.start, local_start)
                segment_end = min(token_slice.stop, local_end)
                if segment_start >= segment_end:
                    continue
                train_sequence = train_sequences[request_mapping[rid]]
                segment_positions = positions[segment_start:segment_end]
                for layer_id in selected_layers:
                    weight_key = (dump_file.parent.name, layer_id)
                    if weight_key not in compared_weights:
                        rollout_weight = rollout_routers[layer_id]["weight"]
                        train_weight = train_sequence.routers[layer_id]["weight"]
                        if rollout_weight.shape != train_weight.shape:
                            raise ValueError(
                                f"{dump_file}:layer{layer_id} router weight shape differs: "
                                f"{tuple(rollout_weight.shape)} != {tuple(train_weight.shape)}"
                            )
                        weight_stats = stats[layer_id]["weight"]
                        dtype_matches = rollout_weight.dtype == train_weight.dtype
                        exact = dtype_matches and torch.equal(rollout_weight, train_weight)
                        difference = (rollout_weight.float() - train_weight.float()).abs()
                        weight_stats["exact"] = weight_stats["exact"] and exact
                        weight_stats["comparisons"] += 1
                        weight_stats["dtype_mismatches"] += int(not dtype_matches)
                        weight_stats["mismatched_elements"] += int(
                            torch.count_nonzero(rollout_weight != train_weight).item()
                        )
                        weight_stats["max_abs"] = max(
                            weight_stats["max_abs"], float(difference.max().item())
                        )
                        weight_stats["numel"] += difference.numel()
                        compared_weights.add(weight_key)

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
                    if torch.any(kept_positions < 0) or torch.any(kept_positions >= train_sequence.tokens.numel()):
                        raise IndexError(f"SGLang request {rid} contains positions outside its Megatron sequence")
                    layer_stats = stats[layer_id]
                    layer_stats["tokens"] += kept_positions.numel()
                    for field in (*_UPSTREAM_FIELDS, "input", "logits"):
                        rollout_value = rollout_routers[layer_id][field][
                            segment_start - local_start : segment_end - local_start
                        ][keep]
                        train_value = train_sequence.routers[layer_id][field][kept_positions]
                        difference = (rollout_value.float() - train_value.float()).abs()
                        field_stats = layer_stats[field]
                        field_stats["max_abs"] = max(field_stats["max_abs"], float(difference.max().item()))
                        field_stats["sum_abs"] += float(difference.sum().item())
                        field_stats["numel"] += difference.numel()

                    rollout_ids = rollout_routers[layer_id]["topk_ids"][
                        segment_start - local_start : segment_end - local_start
                    ][keep].to(torch.int64)
                    train_ids = train_sequence.routers[layer_id]["topk_ids"][kept_positions].to(torch.int64)
                    mismatch = rollout_ids != train_ids
                    set_mismatch = (
                        torch.sort(rollout_ids, dim=-1).values
                        != torch.sort(train_ids, dim=-1).values
                    )
                    id_stats = layer_stats["topk_ids"]
                    id_stats["mismatched_elements"] += int(mismatch.sum().item())
                    id_stats["mismatched_tokens"] += int(mismatch.any(dim=-1).sum().item())
                    id_stats["set_mismatched_tokens"] += int(
                        set_mismatch.any(dim=-1).sum().item()
                    )
                    id_stats["numel"] += mismatch.numel()

                    rollout_weights = rollout_routers[layer_id]["topk_weights"][
                        segment_start - local_start : segment_end - local_start
                    ][keep]
                    train_weights = train_sequence.routers[layer_id]["topk_weights"][
                        kept_positions
                    ]
                    matching_experts = rollout_ids.unsqueeze(2) == train_ids.unsqueeze(1)
                    if torch.any(matching_experts.sum(dim=2) > 1):
                        raise ValueError("SGLang top-k IDs contain duplicates")
                    matched = matching_experts.nonzero(as_tuple=False)
                    weight_stats = layer_stats["topk_weights"]
                    weight_stats["unmatched_routes"] += int(
                        rollout_ids.numel() - matched.shape[0]
                    )
                    if matched.numel():
                        rollout_aligned = rollout_weights[matched[:, 0], matched[:, 1]]
                        train_aligned = train_weights[matched[:, 0], matched[:, 2]]
                        difference = (rollout_aligned.float() - train_aligned.float()).abs()
                        weight_stats["max_abs"] = max(
                            weight_stats["max_abs"], float(difference.max().item())
                        )
                        weight_stats["sum_abs"] += float(difference.sum().item())
                        weight_stats["numel"] += difference.numel()
                        weight_stats["matched_routes"] += difference.numel()
                    for position in kept_positions.tolist():
                        compared.add((rid, int(position), layer_id))

    for layer_stats in stats.values():
        for field in (*_UPSTREAM_FIELDS, "input", "logits", "topk_weights"):
            field_stats = layer_stats[field]
            field_stats["mean_abs"] = (
                field_stats.pop("sum_abs") / field_stats["numel"] if field_stats["numel"] else float("nan")
            )
    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--megatron-router-dir", type=Path, required=True)
    parser.add_argument("--sglang-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--max-stage-diff", type=float, default=0.0)
    parser.add_argument("--max-input-diff", type=float, default=0.0)
    parser.add_argument("--max-logit-diff", type=float, default=0.0)
    parser.add_argument("--max-weight-diff", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--sglang-attn-tp-size", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_threads <= 0:
        raise ValueError(f"--num-threads must be positive, got {args.num_threads}")
    torch.set_num_threads(args.num_threads)
    selected_layers = set(args.layers)
    train_sequences = load_train_router_sequences(
        args.megatron_dir,
        args.megatron_router_dir,
        selected_layers,
    )
    dump_files = _sglang_dump_files(args.sglang_dir)
    request_mapping = map_requests_to_train_sequences(dump_files, train_sequences)
    stats = compare_router_outputs(
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
            f"layer_input_max_abs={layer_stats['layer_input']['max_abs']:.12g} "
            f"post_attention_residual_max_abs="
            f"{layer_stats['post_attention_residual']['max_abs']:.12g} "
            f"post_attention_norm_max_abs="
            f"{layer_stats['post_attention_norm']['max_abs']:.12g} "
            f"input_max_abs={layer_stats['input']['max_abs']:.12g} "
            f"logits_max_abs={layer_stats['logits']['max_abs']:.12g} "
            f"router_weight_exact={layer_stats['weight']['exact']} "
            f"aligned_weights_max_abs={layer_stats['topk_weights']['max_abs']:.12g} "
            f"id_order_mismatched_tokens={layer_stats['topk_ids']['mismatched_tokens']} "
            f"id_set_mismatched_tokens={layer_stats['topk_ids']['set_mismatched_tokens']}"
        )
        if (
            not layer_stats["tokens"]
            or any(
                layer_stats[field]["max_abs"] > args.max_stage_diff
                for field in _UPSTREAM_FIELDS
            )
            or not layer_stats["weight"]["comparisons"]
            or not layer_stats["weight"]["exact"]
            or layer_stats["input"]["max_abs"] > args.max_input_diff
            or layer_stats["logits"]["max_abs"] > args.max_logit_diff
            or layer_stats["topk_weights"]["max_abs"] > args.max_weight_diff
            or layer_stats["topk_ids"]["mismatched_elements"]
        ):
            failed = True

    result = {
        "max_stage_diff": args.max_stage_diff,
        "max_input_diff": args.max_input_diff,
        "max_logit_diff": args.max_logit_diff,
        "max_weight_diff": args.max_weight_diff,
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
