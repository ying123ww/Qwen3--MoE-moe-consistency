#!/usr/bin/env python3
"""Compare Qwen3 attention stages between Megatron and SGLang dumps."""

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

_PARTITIONED_ATTENTION_FIELDS = (
    "layer_input",
    "input_norm",
    "qkv",
    "q_norm_input",
    "k_norm_input",
    "q_norm",
    "k_norm",
    "q_rope",
    "k_rope",
    "value",
    "core_attention",
)
_LOCAL_ROW_FIELDS = (
    "output_projection",
    "post_attention_residual",
    "pre_mlp_norm",
)
_TOKEN_LAYOUT_FIELDS = (
    "layer_input",
    "input_norm",
)
_DERIVED_LOCAL_ROW_FIELDS = ("pre_mlp_norm_input_fp32",)
_ATTENTION_FIELDS = (
    *_PARTITIONED_ATTENTION_FIELDS,
    *_LOCAL_ROW_FIELDS,
    *_DERIVED_LOCAL_ROW_FIELDS,
)
_PARAMETER_FIELDS = (
    "input_norm_weight",
    "qkv_weight",
    "q_norm_weight",
    "k_norm_weight",
    "output_projection_weight",
    "pre_mlp_norm_weight",
)
_SGLANG_SUFFIXES = {
    "input_norm": ".layers.{layer}.input_layernorm",
    "qkv": ".layers.{layer}.self_attn.qkv_proj",
    "q_norm": ".layers.{layer}.self_attn.q_norm",
    "k_norm": ".layers.{layer}.self_attn.k_norm",
    "q_rope": ".layers.{layer}.self_attn.attn.input",
    "core_attention": ".layers.{layer}.self_attn.attn",
}


@dataclass
class TrainAttentionSequence:
    tokens: torch.Tensor
    attention: dict[int, dict[str, torch.Tensor]]
    source: str


def _rows(
    value,
    num_tokens: int | tuple[int, ...],
    context: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{context} must be a tensor")
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0, :]
    expected_rows = (num_tokens,) if isinstance(num_tokens, int) else tuple(dict.fromkeys(num_tokens))
    if value.ndim != 2 or value.shape[0] not in expected_rows:
        raise ValueError(f"{context} must have rows in {expected_rows} and two dimensions after removing a singleton batch dimension, got {tuple(value.shape)}")
    return value


def _heads(
    value,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    context: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{context} must be a tensor")
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim == 3 and value.shape[1] == 1 and value.shape[2] == num_heads * head_dim:
        value = value[:, 0]
    if value.ndim == 2:
        if value.shape == (num_tokens, num_heads * head_dim):
            value = value.reshape(num_tokens, num_heads, head_dim)
        elif value.shape == (num_tokens * num_heads, head_dim):
            value = value.reshape(num_tokens, num_heads, head_dim)
    if value.shape != (num_tokens, num_heads, head_dim):
        raise ValueError(f"{context} must resolve to {(num_tokens, num_heads, head_dim)}, got {tuple(value.shape)}")
    return value


def _head_partition(
    value: torch.Tensor,
    attn_tp_rank: int,
    attn_tp_size: int,
) -> torch.Tensor:
    if value.ndim != 3 or value.shape[1] % attn_tp_size:
        raise ValueError(f"Cannot partition head tensor {tuple(value.shape)} over attention TP {attn_tp_size}")
    heads_per_rank = value.shape[1] // attn_tp_size
    start = attn_tp_rank * heads_per_rank
    return value[:, start : start + heads_per_rank]


def _megatron_qkv_partition(
    value: torch.Tensor,
    attn_tp_rank: int,
    attn_tp_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    num_tokens = value.shape[0]
    query_heads_per_group = num_query_heads // num_kv_heads
    expected_width = (num_query_heads + 2 * num_kv_heads) * head_dim
    if value.ndim != 2 or value.shape[1] != expected_width:
        raise ValueError(f"Megatron QKV must have shape ({num_tokens}, {expected_width}), got {tuple(value.shape)}")
    grouped = value.reshape(
        num_tokens,
        num_kv_heads,
        (query_heads_per_group + 2) * head_dim,
    )
    query_width = query_heads_per_group * head_dim
    query = grouped[:, :, :query_width].reshape(num_tokens, num_query_heads, head_dim)
    key = grouped[:, :, query_width : query_width + head_dim]
    value_heads = grouped[:, :, query_width + head_dim :]
    query = _head_partition(query, attn_tp_rank, attn_tp_size)
    key = _head_partition(key, attn_tp_rank, attn_tp_size)
    value_heads = _head_partition(value_heads, attn_tp_rank, attn_tp_size)
    return torch.cat(
        (
            query.reshape(num_tokens, -1),
            key.reshape(num_tokens, -1),
            value_heads.reshape(num_tokens, -1),
        ),
        dim=-1,
    )


def _tensor_comparison(rollout: torch.Tensor, train: torch.Tensor) -> dict:
    if rollout.shape != train.shape:
        raise ValueError(f"Tensor shape mismatch: rollout={tuple(rollout.shape)} train={tuple(train.shape)}")
    difference = (rollout.float() - train.float()).abs()
    mismatched_elements = int(torch.count_nonzero(rollout != train).item())
    dtype_exact = rollout.dtype == train.dtype
    return {
        "rollout_shape": list(rollout.shape),
        "train_shape": list(train.shape),
        "rollout_dtype": str(rollout.dtype),
        "train_dtype": str(train.dtype),
        "dtype_exact": dtype_exact,
        "mismatched_elements": mismatched_elements,
        "numel": difference.numel(),
        "max_abs": float(difference.max().item()),
        "sum_abs": float(difference.sum().item()),
        "exact": dtype_exact and mismatched_elements == 0,
    }


def load_train_attention_parameters(
    megatron_dir: Path,
    selected_layers: set[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], str]:
    for dump_file in sorted(megatron_dir.glob("rank*/actor_Pass*.pt")):
        for record in _load_records(dump_file):
            parameters = record.get("module_params")
            if not parameters:
                continue
            layers = {}
            for layer_id in selected_layers:
                prefix = f"decoder.layers.{layer_id}.self_attention.linear_qkv.parameter."
                layers[layer_id] = {
                    "input_norm_weight": _find_suffix(parameters, prefix + "layer_norm_weight"),
                    "qkv_weight": _find_suffix(parameters, prefix + "weight"),
                    "q_norm_weight": _find_suffix(
                        parameters,
                        f"decoder.layers.{layer_id}.self_attention.q_layernorm.parameter.weight",
                    ),
                    "k_norm_weight": _find_suffix(
                        parameters,
                        f"decoder.layers.{layer_id}.self_attention.k_layernorm.parameter.weight",
                    ),
                    "output_projection_weight": _find_suffix(
                        parameters,
                        f"decoder.layers.{layer_id}.self_attention.linear_proj.parameter.weight",
                    ),
                    "pre_mlp_norm_weight": _find_suffix(
                        parameters,
                        f"decoder.layers.{layer_id}.pre_mlp_layernorm.parameter.weight",
                    ),
                }
            return layers, str(dump_file)
    raise RuntimeError("Megatron dumps contained no captured attention parameters")


def load_train_attention_sequences(
    megatron_dir: Path,
    selected_layers: set[int],
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> list[TrainAttentionSequence]:
    dump_files = sorted(megatron_dir.glob("rank*/actor_Pass*.pt"))
    if not dump_files:
        raise FileNotFoundError(f"No Megatron layer dumps found under {megatron_dir}")

    sequences = []
    for dump_file in dump_files:
        for record_index, record in enumerate(_load_records(dump_file)):
            tokens = record["input_ids"].reshape(-1).to(torch.int64)
            modules = record["modules"]
            module_inputs = record["module_inputs"]
            layers = {}
            for layer_id in selected_layers:
                prefix = f"decoder.layers.{layer_id}.self_attention"
                layers[layer_id] = {
                    "layer_input": _rows(
                        _find_suffix(module_inputs, prefix + ".linear_qkv"),
                        tokens.numel(),
                        f"{dump_file}:layer{layer_id}:layer_input",
                    ),
                    "input_norm": _rows(
                        _find_suffix(modules, prefix + ".linear_qkv.layernorm_output"),
                        tokens.numel(),
                        f"{dump_file}:layer{layer_id}:input_norm",
                    ),
                    "qkv": _rows(
                        _find_suffix(modules, prefix + ".linear_qkv"),
                        tokens.numel(),
                        f"{dump_file}:layer{layer_id}:qkv",
                    ),
                    "q_norm_input": _heads(
                        _find_suffix(module_inputs, prefix + ".q_layernorm"),
                        tokens.numel(),
                        num_query_heads,
                        head_dim,
                        f"{dump_file}:layer{layer_id}:q_norm_input",
                    ),
                    "k_norm_input": _heads(
                        _find_suffix(module_inputs, prefix + ".k_layernorm"),
                        tokens.numel(),
                        num_kv_heads,
                        head_dim,
                        f"{dump_file}:layer{layer_id}:k_norm_input",
                    ),
                    "q_norm": _heads(
                        _find_suffix(modules, prefix + ".q_layernorm"),
                        tokens.numel(),
                        num_query_heads,
                        head_dim,
                        f"{dump_file}:layer{layer_id}:q_norm",
                    ),
                    "k_norm": _heads(
                        _find_suffix(modules, prefix + ".k_layernorm"),
                        tokens.numel(),
                        num_kv_heads,
                        head_dim,
                        f"{dump_file}:layer{layer_id}:k_norm",
                    ),
                    "q_rope": _heads(
                        _find_suffix(module_inputs, prefix + ".core_attention"),
                        tokens.numel(),
                        num_query_heads,
                        head_dim,
                        f"{dump_file}:layer{layer_id}:q_rope",
                    ),
                    "k_rope": _heads(
                        _find_suffix(
                            module_inputs,
                            prefix + ".core_attention.input.1",
                        ),
                        tokens.numel(),
                        num_kv_heads,
                        head_dim,
                        f"{dump_file}:layer{layer_id}:k_rope",
                    ),
                    "value": _heads(
                        _find_suffix(
                            module_inputs,
                            prefix + ".core_attention.input.2",
                        ),
                        tokens.numel(),
                        num_kv_heads,
                        head_dim,
                        f"{dump_file}:layer{layer_id}:value",
                    ),
                    "core_attention": _heads(
                        _find_suffix(modules, prefix + ".core_attention"),
                        tokens.numel(),
                        num_query_heads,
                        head_dim,
                        f"{dump_file}:layer{layer_id}:core_attention",
                    ),
                    "output_projection": _rows(
                        _find_suffix(modules, prefix + ".linear_proj"),
                        tokens.numel(),
                        f"{dump_file}:layer{layer_id}:output_projection",
                    ),
                    "post_attention_residual": _rows(
                        _find_suffix(
                            module_inputs,
                            f"decoder.layers.{layer_id}.pre_mlp_layernorm",
                        ),
                        tokens.numel(),
                        f"{dump_file}:layer{layer_id}:post_attention_residual",
                    ),
                    "pre_mlp_norm": _rows(
                        _find_suffix(
                            modules,
                            f"decoder.layers.{layer_id}.pre_mlp_layernorm",
                        ),
                        tokens.numel(),
                        f"{dump_file}:layer{layer_id}:pre_mlp_norm",
                    ),
                }

            cu_seqlens = record["cu_seqlens"].reshape(-1).tolist()
            for sequence_index, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:], strict=False)):
                sequence_tokens = tokens[start:end]
                if sequence_tokens.numel() == 0 or torch.count_nonzero(sequence_tokens) == 0:
                    continue
                sequences.append(
                    TrainAttentionSequence(
                        tokens=sequence_tokens,
                        attention={layer_id: {field: value[start:end] for field, value in values.items()} for layer_id, values in layers.items()},
                        source=(f"{dump_file}:record{record_index}:sequence{sequence_index}"),
                    )
                )
    if not sequences:
        raise RuntimeError("Megatron dumps contained no non-padding attention sequences")
    return sequences


def _load_rollout_attention(
    record,
    dump_file: Path,
    selected_layers: set[int],
    num_tokens: int,
    local_num_tokens: int,
    local_query_heads: int,
    local_kv_heads: int,
    head_dim: int,
):
    rollout = {}
    for layer_id in selected_layers:
        rollout[layer_id] = {
            "layer_input": _rows(
                _find_suffix(
                    record,
                    _SGLANG_SUFFIXES["input_norm"].format(layer=layer_id) + ".input",
                ),
                (num_tokens, local_num_tokens),
                f"{dump_file}:layer{layer_id}:layer_input",
            ),
            "input_norm": _rows(
                _find_suffix(
                    record,
                    _SGLANG_SUFFIXES["input_norm"].format(layer=layer_id),
                ),
                (num_tokens, local_num_tokens),
                f"{dump_file}:layer{layer_id}:input_norm",
            ),
            "qkv": _rows(
                _find_suffix(record, _SGLANG_SUFFIXES["qkv"].format(layer=layer_id)),
                num_tokens,
                f"{dump_file}:layer{layer_id}:qkv",
            ),
            "q_norm_input": _heads(
                _find_suffix(
                    record,
                    _SGLANG_SUFFIXES["q_norm"].format(layer=layer_id) + ".input",
                ),
                num_tokens,
                local_query_heads,
                head_dim,
                f"{dump_file}:layer{layer_id}:q_norm_input",
            ),
            "k_norm_input": _heads(
                _find_suffix(
                    record,
                    _SGLANG_SUFFIXES["k_norm"].format(layer=layer_id) + ".input",
                ),
                num_tokens,
                local_kv_heads,
                head_dim,
                f"{dump_file}:layer{layer_id}:k_norm_input",
            ),
            "q_norm": _heads(
                _find_suffix(record, _SGLANG_SUFFIXES["q_norm"].format(layer=layer_id)),
                num_tokens,
                local_query_heads,
                head_dim,
                f"{dump_file}:layer{layer_id}:q_norm",
            ),
            "k_norm": _heads(
                _find_suffix(record, _SGLANG_SUFFIXES["k_norm"].format(layer=layer_id)),
                num_tokens,
                local_kv_heads,
                head_dim,
                f"{dump_file}:layer{layer_id}:k_norm",
            ),
            "q_rope": _heads(
                _find_suffix(record, _SGLANG_SUFFIXES["q_rope"].format(layer=layer_id)),
                num_tokens,
                local_query_heads,
                head_dim,
                f"{dump_file}:layer{layer_id}:q_rope",
            ),
            "k_rope": _heads(
                _find_suffix(
                    record,
                    _SGLANG_SUFFIXES["core_attention"].format(layer=layer_id) + ".input.1",
                ),
                num_tokens,
                local_kv_heads,
                head_dim,
                f"{dump_file}:layer{layer_id}:k_rope",
            ),
            "value": _heads(
                _find_suffix(
                    record,
                    _SGLANG_SUFFIXES["core_attention"].format(layer=layer_id) + ".input.2",
                ),
                num_tokens,
                local_kv_heads,
                head_dim,
                f"{dump_file}:layer{layer_id}:value",
            ),
            "core_attention": _heads(
                _find_suffix(
                    record,
                    _SGLANG_SUFFIXES["core_attention"].format(layer=layer_id),
                ),
                num_tokens,
                local_query_heads,
                head_dim,
                f"{dump_file}:layer{layer_id}:core_attention",
            ),
            "output_projection": _rows(
                _find_suffix(
                    record,
                    f".layers.{layer_id}.post_attention_layernorm.input",
                ),
                local_num_tokens,
                f"{dump_file}:layer{layer_id}:output_projection",
            ),
            "post_attention_residual": _rows(
                _find_suffix(
                    record,
                    f".layers.{layer_id}.post_attention_layernorm.output.1",
                ),
                local_num_tokens,
                f"{dump_file}:layer{layer_id}:post_attention_residual",
            ),
            "pre_mlp_norm": _rows(
                _find_suffix(
                    record,
                    f".layers.{layer_id}.post_attention_layernorm",
                ),
                local_num_tokens,
                f"{dump_file}:layer{layer_id}:pre_mlp_norm",
            ),
        }
    return rollout


def compare_attention_parameters(
    dump_files: list[Path],
    train_parameters: dict[int, dict[str, torch.Tensor]],
    selected_layers: set[int],
    attn_tp_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict:
    expected_ranks = {int(match.group(1)) for dump_file in dump_files if (match := _TP_RANK_RE.search(dump_file.parent.name)) is not None}
    per_layer = {layer_id: {"per_tp_rank": {}, "attn_tp_rank_counts": {}} for layer_id in selected_layers}
    seen_ranks = set()
    for dump_file in dump_files:
        match = _TP_RANK_RE.search(dump_file.parent.name)
        if match is None:
            continue
        tp_rank = int(match.group(1))
        if tp_rank in seen_ranks:
            continue
        parameter_record = None
        for record in _load_records(dump_file):
            if any(".parameter." in key for key in record):
                parameter_record = record
                break
        if parameter_record is None:
            continue
        attn_tp_rank = tp_rank % attn_tp_size
        for layer_id in selected_layers:
            input_norm = _find_suffix(
                parameter_record,
                f".layers.{layer_id}.input_layernorm.parameter.weight",
            )
            qkv = _find_suffix(
                parameter_record,
                f".layers.{layer_id}.self_attn.qkv_proj.parameter.weight",
            )
            q_norm = _find_suffix(
                parameter_record,
                f".layers.{layer_id}.self_attn.q_norm.parameter.weight",
            )
            k_norm = _find_suffix(
                parameter_record,
                f".layers.{layer_id}.self_attn.k_norm.parameter.weight",
            )
            output_projection = _find_suffix(
                parameter_record,
                f".layers.{layer_id}.self_attn.o_proj.parameter.weight",
            )
            pre_mlp_norm = _find_suffix(
                parameter_record,
                f".layers.{layer_id}.post_attention_layernorm.parameter.weight",
            )
            train_input_norm = train_parameters[layer_id]["input_norm_weight"]
            train_qkv = _megatron_qkv_partition(
                train_parameters[layer_id]["qkv_weight"].T,
                attn_tp_rank,
                attn_tp_size,
                num_query_heads,
                num_kv_heads,
                head_dim,
            ).T.contiguous()
            train_output_projection = train_parameters[layer_id]["output_projection_weight"]
            local_width = train_output_projection.shape[1] // attn_tp_size
            start = attn_tp_rank * local_width
            train_output_projection = train_output_projection[:, start : start + local_width].contiguous()
            layer_result = per_layer[layer_id]
            layer_result["per_tp_rank"][tp_rank] = {
                "attn_tp_rank": attn_tp_rank,
                "source": str(dump_file),
                "input_norm_weight": _tensor_comparison(input_norm, train_input_norm),
                "qkv_weight": _tensor_comparison(qkv, train_qkv),
                "q_norm_weight": _tensor_comparison(q_norm, train_parameters[layer_id]["q_norm_weight"]),
                "k_norm_weight": _tensor_comparison(k_norm, train_parameters[layer_id]["k_norm_weight"]),
                "output_projection_weight": _tensor_comparison(output_projection, train_output_projection),
                "pre_mlp_norm_weight": _tensor_comparison(
                    pre_mlp_norm,
                    train_parameters[layer_id]["pre_mlp_norm_weight"],
                ),
            }
            counts = layer_result["attn_tp_rank_counts"]
            counts[attn_tp_rank] = counts.get(attn_tp_rank, 0) + 1
        seen_ranks.add(tp_rank)

    if seen_ranks != expected_ranks:
        raise RuntimeError(f"Captured SGLang attention parameters for TP ranks {sorted(seen_ranks)}, expected {sorted(expected_ranks)}")

    for layer_result in per_layer.values():
        layer_result["replicas"] = len(layer_result["per_tp_rank"])
        for field in _PARAMETER_FIELDS:
            comparisons = [rank_result[field] for rank_result in layer_result["per_tp_rank"].values()]
            numel = sum(item["numel"] for item in comparisons)
            sum_abs = sum(item["sum_abs"] for item in comparisons)
            layer_result[field] = {
                "comparisons": len(comparisons),
                "max_abs": max(item["max_abs"] for item in comparisons),
                "mean_abs": sum_abs / numel,
                "mismatched_elements": sum(item["mismatched_elements"] for item in comparisons),
                "numel": numel,
                "dtype_exact": all(item["dtype_exact"] for item in comparisons),
                "exact": all(item["exact"] for item in comparisons),
            }
        for rank_result in layer_result["per_tp_rank"].values():
            for field in _PARAMETER_FIELDS:
                rank_result[field].pop("sum_abs")
    return per_layer


def compare_attention_outputs(
    dump_files: list[Path],
    train_sequences: list[TrainAttentionSequence],
    request_mapping: dict[str, int],
    selected_layers: set[int],
    attn_tp_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    if attn_tp_size <= 0:
        raise ValueError(f"Attention TP size must be positive, got {attn_tp_size}")
    if num_query_heads % attn_tp_size or num_kv_heads % attn_tp_size:
        raise ValueError("Query and KV heads must be divisible by attention TP size")
    local_query_heads = num_query_heads // attn_tp_size
    local_kv_heads = num_kv_heads // attn_tp_size
    stats = {
        layer_id: {
            "token_head_partitions": 0,
            "local_tokens": 0,
            "rank_tokens": {rank: 0 for rank in range(attn_tp_size)},
            **{
                field: {
                    "max_abs": 0.0,
                    "sum_abs": 0.0,
                    "numel": 0,
                    "mismatched_elements": 0,
                    "dtype_exact": True,
                }
                for field in _ATTENTION_FIELDS
            },
        }
        for layer_id in selected_layers
    }
    compared: set[tuple[str, int, int, int]] = set()
    compared_local: set[tuple[str, int, int]] = set()

    for dump_file in dump_files:
        match = _TP_RANK_RE.search(dump_file.parent.name)
        if match is None:
            raise ValueError(f"Cannot determine SGLang TP rank from {dump_file.parent.name!r}")
        attn_tp_rank = int(match.group(1)) % attn_tp_size
        for record in _load_records(dump_file):
            input_ids, positions, segments = _sglang_segments(record)
            if not segments:
                continue
            if input_ids.numel() % attn_tp_size:
                raise ValueError(f"{dump_file} has {input_ids.numel()} input rows, which cannot be evenly scattered over attention TP {attn_tp_size}")
            local_num_tokens = input_ids.numel() // attn_tp_size
            local_start = attn_tp_rank * local_num_tokens
            local_end = local_start + local_num_tokens
            rollout = _load_rollout_attention(
                record,
                dump_file,
                selected_layers,
                input_ids.numel(),
                local_num_tokens,
                local_query_heads,
                local_kv_heads,
                head_dim,
            )

            for rid, token_slice in segments:
                train_sequence = train_sequences[request_mapping[rid]]
                segment_positions = positions[token_slice]
                for layer_id in selected_layers:
                    keep = torch.tensor(
                        [
                            int(position) < train_sequence.tokens.numel() - 1
                            and (
                                rid,
                                int(position),
                                layer_id,
                                attn_tp_rank,
                            )
                            not in compared
                            for position in segment_positions.tolist()
                        ],
                        dtype=torch.bool,
                    )
                    if not torch.any(keep):
                        continue
                    kept_positions = segment_positions[keep]
                    if torch.any(kept_positions < 0) or torch.any(kept_positions >= train_sequence.tokens.numel()):
                        raise IndexError(f"SGLang request {rid} contains positions outside Megatron")
                    layer_stats = stats[layer_id]
                    layer_stats["token_head_partitions"] += kept_positions.numel()
                    layer_stats["rank_tokens"][attn_tp_rank] += kept_positions.numel()
                    rollout_values = rollout[layer_id]
                    train_values = train_sequence.attention[layer_id]
                    for field in _PARTITIONED_ATTENTION_FIELDS:
                        if field in _TOKEN_LAYOUT_FIELDS and rollout_values[field].shape[0] == local_num_tokens:
                            continue
                        rollout_value = rollout_values[field][token_slice][keep]
                        train_value = train_values[field][kept_positions]
                        if field in _TOKEN_LAYOUT_FIELDS:
                            pass
                        elif field == "qkv":
                            train_value = _megatron_qkv_partition(
                                train_value,
                                attn_tp_rank,
                                attn_tp_size,
                                num_query_heads,
                                num_kv_heads,
                                head_dim,
                            )
                        else:
                            train_value = _head_partition(
                                train_value,
                                attn_tp_rank,
                                attn_tp_size,
                            )
                        difference = (rollout_value.float() - train_value.float()).abs()
                        field_stats = layer_stats[field]
                        field_stats["max_abs"] = max(
                            field_stats["max_abs"],
                            float(difference.max().item()),
                        )
                        field_stats["sum_abs"] += float(difference.sum().item())
                        field_stats["numel"] += difference.numel()
                        field_stats["mismatched_elements"] += int(torch.count_nonzero(rollout_value != train_value).item())
                        field_stats["dtype_exact"] = field_stats["dtype_exact"] and rollout_value.dtype == train_value.dtype
                    for position in kept_positions.tolist():
                        compared.add(
                            (
                                rid,
                                int(position),
                                layer_id,
                                attn_tp_rank,
                            )
                        )

                segment_start = max(token_slice.start, local_start)
                segment_end = min(token_slice.stop, local_end)
                if segment_start >= segment_end:
                    continue
                local_positions = positions[segment_start:segment_end]
                for layer_id in selected_layers:
                    local_keep = torch.tensor(
                        [int(position) < train_sequence.tokens.numel() - 1 and (rid, int(position), layer_id) not in compared_local for position in local_positions.tolist()],
                        dtype=torch.bool,
                    )
                    if not torch.any(local_keep):
                        continue
                    kept_positions = local_positions[local_keep]
                    layer_stats = stats[layer_id]
                    layer_stats["local_tokens"] += kept_positions.numel()
                    rollout_values = rollout[layer_id]
                    train_values = train_sequence.attention[layer_id]
                    local_fields = list(_LOCAL_ROW_FIELDS)
                    local_fields.extend(field for field in _TOKEN_LAYOUT_FIELDS if rollout_values[field].shape[0] == local_num_tokens)
                    local_slice = slice(
                        segment_start - local_start,
                        segment_end - local_start,
                    )
                    for field in local_fields:
                        rollout_value = rollout_values[field][local_slice][local_keep]
                        train_value = train_values[field][kept_positions]
                        difference = (rollout_value.float() - train_value.float()).abs()
                        field_stats = layer_stats[field]
                        field_stats["max_abs"] = max(
                            field_stats["max_abs"],
                            float(difference.max().item()),
                        )
                        field_stats["sum_abs"] += float(difference.sum().item())
                        field_stats["numel"] += difference.numel()
                        field_stats["mismatched_elements"] += int(torch.count_nonzero(rollout_value != train_value).item())
                        field_stats["dtype_exact"] = field_stats["dtype_exact"] and rollout_value.dtype == train_value.dtype
                    rollout_layer_input = rollout_values["layer_input"]
                    if rollout_layer_input.shape[0] == local_num_tokens:
                        rollout_layer_input = rollout_layer_input[local_slice][local_keep]
                    else:
                        rollout_layer_input = rollout_layer_input[segment_start:segment_end][local_keep]
                    rollout_fp32_sum = rollout_values["output_projection"][local_slice][local_keep].float() + rollout_layer_input.float()
                    train_fp32_sum = train_values["output_projection"][kept_positions].float() + train_values["layer_input"][kept_positions].float()
                    difference = (rollout_fp32_sum - train_fp32_sum).abs()
                    field_stats = layer_stats["pre_mlp_norm_input_fp32"]
                    field_stats["max_abs"] = max(
                        field_stats["max_abs"],
                        float(difference.max().item()),
                    )
                    field_stats["sum_abs"] += float(difference.sum().item())
                    field_stats["numel"] += difference.numel()
                    field_stats["mismatched_elements"] += int(torch.count_nonzero(rollout_fp32_sum != train_fp32_sum).item())
                    field_stats["dtype_exact"] = field_stats["dtype_exact"] and rollout_fp32_sum.dtype == train_fp32_sum.dtype
                    for position in kept_positions.tolist():
                        compared_local.add((rid, int(position), layer_id))

    for layer_stats in stats.values():
        for field in _ATTENTION_FIELDS:
            field_stats = layer_stats[field]
            field_stats["mean_abs"] = field_stats.pop("sum_abs") / field_stats["numel"] if field_stats["numel"] else float("nan")
            field_stats["exact"] = field_stats["dtype_exact"] and field_stats["mismatched_elements"] == 0
    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--sglang-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--sglang-attn-tp-size", type=int, default=1)
    parser.add_argument("--num-query-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--max-stage-diff", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_threads <= 0:
        raise ValueError(f"--num-threads must be positive, got {args.num_threads}")
    torch.set_num_threads(args.num_threads)
    selected_layers = set(args.layers)
    train_sequences = load_train_attention_sequences(
        args.megatron_dir,
        selected_layers,
        args.num_query_heads,
        args.num_kv_heads,
        args.head_dim,
    )
    train_parameters, train_parameter_source = load_train_attention_parameters(args.megatron_dir, selected_layers)
    dump_files = _sglang_dump_files(args.sglang_dir)
    parameter_stats = compare_attention_parameters(
        dump_files,
        train_parameters,
        selected_layers,
        args.sglang_attn_tp_size,
        args.num_query_heads,
        args.num_kv_heads,
        args.head_dim,
    )
    request_mapping = map_requests_to_train_sequences(dump_files, train_sequences)
    stats = compare_attention_outputs(
        dump_files,
        train_sequences,
        request_mapping,
        selected_layers,
        args.sglang_attn_tp_size,
        args.num_query_heads,
        args.num_kv_heads,
        args.head_dim,
    )

    failed = False
    for layer_id in sorted(stats):
        layer_stats = stats[layer_id]
        print(f"layer={layer_id} token_head_partitions={layer_stats['token_head_partitions']} " + " ".join(f"{field}_max_abs={layer_stats[field]['max_abs']:.12g}" for field in _ATTENTION_FIELDS))
        if not layer_stats["token_head_partitions"] or not layer_stats["local_tokens"] or any(not count for count in layer_stats["rank_tokens"].values()) or any(layer_stats[field]["max_abs"] > args.max_stage_diff for field in _ATTENTION_FIELDS):
            failed = True
        layer_parameters = parameter_stats[layer_id]
        print(
            f"layer={layer_id} "
            f"input_norm_weight_exact="
            f"{layer_parameters['input_norm_weight']['exact']} "
            f"qkv_weight_exact={layer_parameters['qkv_weight']['exact']} "
            f"q_norm_weight_exact={layer_parameters['q_norm_weight']['exact']} "
            f"k_norm_weight_exact={layer_parameters['k_norm_weight']['exact']} "
            f"output_projection_weight_exact="
            f"{layer_parameters['output_projection_weight']['exact']} "
            f"pre_mlp_norm_weight_exact="
            f"{layer_parameters['pre_mlp_norm_weight']['exact']}"
        )
        if not all(layer_parameters[field]["exact"] for field in _PARAMETER_FIELDS):
            failed = True

    result = {
        "max_stage_diff": args.max_stage_diff,
        "sglang_attn_tp_size": args.sglang_attn_tp_size,
        "num_query_heads": args.num_query_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "request_mapping": request_mapping,
        "train_parameter_source": train_parameter_source,
        "parameters": parameter_stats,
        "layers": stats,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
