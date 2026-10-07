#!/usr/bin/env python3
"""Compare Qwen3 Megatron expert rows with SGLang Triton route outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import torch

from slime.utils.compare_glm52_layerwise import _find_suffix, _load_records


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.contiguous().view(torch.uint8).numpy().tobytes()


def _difference_stats() -> dict[str, float | int]:
    return {"max_abs": 0.0, "sum_abs": 0.0, "numel": 0}


def _update_difference(stats: dict, actual: torch.Tensor, expected: torch.Tensor) -> None:
    difference = (actual.float() - expected.float()).abs()
    stats["max_abs"] = max(stats["max_abs"], float(difference.max().item()))
    stats["sum_abs"] += float(difference.sum().item())
    stats["numel"] += difference.numel()


def _finish_difference(stats: dict) -> None:
    stats["mean_abs"] = (
        stats.pop("sum_abs") / stats["numel"] if stats["numel"] else float("nan")
    )


def _load_megatron_routes(dump_file: Path, layer_id: int):
    records = list(_load_records(dump_file))
    if len(records) != 1:
        raise ValueError(f"Expected one Megatron record in {dump_file}, got {len(records)}")
    record = records[0]
    suffix = f"decoder.layers.{layer_id}.mlp.experts"
    hidden_states = _find_suffix(record["module_inputs"], suffix + ".input.0")
    tokens_per_expert = _find_suffix(record["module_inputs"], suffix + ".input.1")
    route_weights = _find_suffix(record["module_inputs"], suffix + ".input.2")
    route_outputs = _find_suffix(record["modules"], suffix)
    if hidden_states.shape != route_outputs.shape:
        raise ValueError(f"Megatron expert input/output shapes differ in {dump_file}")
    if hidden_states.shape[0] != route_weights.numel():
        raise ValueError(f"Megatron expert rows/weights differ in {dump_file}")
    if int(tokens_per_expert.sum().item()) != hidden_states.shape[0]:
        raise ValueError(f"Megatron expert counts do not cover all rows in {dump_file}")

    routes = defaultdict(deque)
    offset = 0
    for local_expert_id, count in enumerate(tokens_per_expert.tolist()):
        for row in range(offset, offset + count):
            routes[(local_expert_id, _tensor_bytes(hidden_states[row]))].append(
                (route_weights[row], route_outputs[row])
            )
        offset += count
    return routes, hidden_states.shape[0]


def _matching_route(candidates: deque, sglang_weight: torch.Tensor):
    for index, (weight, output) in enumerate(candidates):
        if torch.equal(weight, sglang_weight):
            del candidates[index]
            return weight, output
    if candidates:
        return candidates.popleft()
    return None


def compare_rank(
    megatron_file: Path,
    sglang_process_dir: Path,
    layer_id: int,
    megatron_route_output_weighted: bool = False,
) -> dict:
    megatron_routes, megatron_route_count = _load_megatron_routes(
        megatron_file, layer_id
    )
    prefix = f"model.layers.{layer_id}.mlp"
    stats = {
        "megatron_routes": megatron_route_count,
        "sglang_routes": 0,
        "matched_routes": 0,
        "matched_received_rows": 0,
        "route_weight": _difference_stats(),
        "expert_route_output": _difference_stats(),
        "route_weight_application": _difference_stats(),
        "weighted_route_output": _difference_stats(),
    }

    for dump_file in sorted(sglang_process_dir.glob("Pass*.pt")):
        for record in _load_records(dump_file):
            hidden_key = prefix + ".dispatch.hidden_states"
            if hidden_key not in record:
                continue
            hidden_states = record[hidden_key]
            topk_ids = record[prefix + ".expert_routes.route_topk_ids"]
            topk_weights = record[prefix + ".expert_routes.route_topk_weights"]
            route_outputs = record[prefix + ".expert_routes.route_outputs"]
            expert_route_outputs = record[
                prefix + ".expert_routes.unweighted_route_outputs"
            ]
            if topk_ids.shape != topk_weights.shape:
                raise ValueError(f"SGLang route IDs/weights differ in {dump_file}")
            expected_shape = (*topk_ids.shape, hidden_states.shape[-1])
            if (
                tuple(route_outputs.shape) != expected_shape
                or tuple(expert_route_outputs.shape) != expected_shape
            ):
                raise ValueError(
                    "SGLang weighted/unweighted route output shapes "
                    f"{tuple(route_outputs.shape)}/{tuple(expert_route_outputs.shape)} "
                    f"!= {expected_shape} in {dump_file}"
                )

            matched_rows = set()
            for token_row in range(hidden_states.shape[0]):
                hidden_key_bytes = _tensor_bytes(hidden_states[token_row])
                for topk_slot, local_expert_id in enumerate(topk_ids[token_row].tolist()):
                    if local_expert_id < 0:
                        continue
                    stats["sglang_routes"] += 1
                    candidates = megatron_routes.get(
                        (local_expert_id, hidden_key_bytes)
                    )
                    if not candidates:
                        continue
                    sglang_weight = topk_weights[token_row, topk_slot]
                    match = _matching_route(candidates, sglang_weight)
                    if match is None:
                        continue
                    megatron_weight, megatron_output = match
                    stats["matched_routes"] += 1
                    matched_rows.add(token_row)
                    _update_difference(
                        stats["route_weight"], sglang_weight, megatron_weight
                    )
                    if not megatron_route_output_weighted:
                        _update_difference(
                            stats["expert_route_output"],
                            expert_route_outputs[token_row, topk_slot],
                            megatron_output,
                        )
                    expected_from_unweighted = (
                        expert_route_outputs[token_row, topk_slot].float()
                        * sglang_weight.float()
                    ).to(route_outputs.dtype)
                    _update_difference(
                        stats["route_weight_application"],
                        route_outputs[token_row, topk_slot],
                        expected_from_unweighted,
                    )
                    expected_output = (
                        megatron_output
                        if megatron_route_output_weighted
                        else (megatron_output.float() * megatron_weight.float()).to(
                            route_outputs.dtype
                        )
                    )
                    _update_difference(
                        stats["weighted_route_output"],
                        route_outputs[token_row, topk_slot],
                        expected_output,
                    )
            stats["matched_received_rows"] += len(matched_rows)

    stats["unmatched_megatron_routes"] = sum(
        len(candidates) for candidates in megatron_routes.values()
    )
    for field in (
        "route_weight",
        "expert_route_output",
        "route_weight_application",
        "weighted_route_output",
    ):
        _finish_difference(stats[field])
    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--sglang-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--min-matched-routes", type=int, default=1)
    parser.add_argument("--max-weight-diff", type=float, default=0.0)
    parser.add_argument("--max-route-output-diff", type=float, default=0.0)
    parser.add_argument("--max-weighted-route-output-diff", type=float, default=0.0)
    parser.add_argument("--megatron-route-output-weighted", action="store_true")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.world_size <= 0 or args.num_threads <= 0:
        raise ValueError("world size and thread count must be positive")
    torch.set_num_threads(args.num_threads)
    ranks = {}
    for rank in range(args.world_size):
        megatron_files = sorted(
            (args.megatron_dir / f"rank{rank:05d}").glob("actor_Pass*.pt")
        )
        if len(megatron_files) != 1:
            raise FileNotFoundError(
                f"Expected one Megatron dump for rank {rank}, got {megatron_files}"
            )
        sglang_dirs = sorted(args.sglang_dir.glob(f"TP{rank}_PP*_Rank*_pid*"))
        if len(sglang_dirs) != 1:
            raise FileNotFoundError(
                f"Expected one SGLang process dir for rank {rank}, got {sglang_dirs}"
            )
        rank_stats = compare_rank(
            megatron_files[0],
            sglang_dirs[0],
            args.layer,
            megatron_route_output_weighted=args.megatron_route_output_weighted,
        )
        ranks[str(rank)] = rank_stats
        print(
            f"rank={rank} matched_routes={rank_stats['matched_routes']} "
            f"weight_max_abs={rank_stats['route_weight']['max_abs']:.12g} "
            f"expert_route_output_max_abs={rank_stats['expert_route_output']['max_abs']:.12g} "
            f"weight_application_max_abs={rank_stats['route_weight_application']['max_abs']:.12g} "
            "weighted_route_output_max_abs="
            f"{rank_stats['weighted_route_output']['max_abs']:.12g}"
        )

    matched_routes = sum(rank["matched_routes"] for rank in ranks.values())
    max_weight_diff = max(rank["route_weight"]["max_abs"] for rank in ranks.values())
    max_route_output_diff = max(
        rank["expert_route_output"]["max_abs"] for rank in ranks.values()
    )
    max_weight_application_diff = max(
        rank["route_weight_application"]["max_abs"] for rank in ranks.values()
    )
    max_weighted_route_output_diff = max(
        rank["weighted_route_output"]["max_abs"] for rank in ranks.values()
    )
    passed = (
        matched_routes >= args.min_matched_routes
        and max_weight_diff <= args.max_weight_diff
        and (
            args.megatron_route_output_weighted
            or max_route_output_diff <= args.max_route_output_diff
        )
        and max_weighted_route_output_diff <= args.max_weighted_route_output_diff
    )
    result = {
        "passed": passed,
        "layer": args.layer,
        "world_size": args.world_size,
        "megatron_route_output_weighted": args.megatron_route_output_weighted,
        "thresholds": {
            "min_matched_routes": args.min_matched_routes,
            "max_weight_diff": args.max_weight_diff,
            "max_route_output_diff": args.max_route_output_diff,
            "max_weighted_route_output_diff": args.max_weighted_route_output_diff,
        },
        "summary": {
            "matched_routes": matched_routes,
            "max_weight_diff": max_weight_diff,
            "max_expert_route_output_diff": max_route_output_diff,
            "max_weight_application_diff": max_weight_application_diff,
            "max_weighted_route_output_diff": max_weighted_route_output_diff,
        },
        "ranks": ranks,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
