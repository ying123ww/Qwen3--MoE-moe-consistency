from pathlib import Path

import pytest
import torch

from slime.utils.compare_qwen3_attention import (
    TrainAttentionSequence,
    compare_attention_outputs,
    compare_attention_parameters,
)


def _pack_megatron_qkv(query, key, value):
    groups = []
    query_heads_per_group = query.shape[1] // key.shape[1]
    for group in range(key.shape[1]):
        start = group * query_heads_per_group
        groups.extend(
            (
                query[:, start : start + query_heads_per_group].reshape(query.shape[0], -1),
                key[:, group].reshape(query.shape[0], -1),
                value[:, group].reshape(query.shape[0], -1),
            )
        )
    return torch.cat(groups, dim=-1)


def _write_sglang_dump(
    path: Path,
    rank: int,
    query,
    key,
    value,
    q_rope,
    core_attention,
    layer_input,
    output_projection,
    post_attention_residual,
    pre_mlp_norm,
    local_token_inputs=False,
):
    query_slice = query[:4, rank * 2 : (rank + 1) * 2]
    key_slice = key[:4, rank : rank + 1]
    value_slice = value[:4, rank : rank + 1]
    local_token_slice = slice(rank * 2, (rank + 1) * 2)
    layer_input_dump = layer_input[local_token_slice] if local_token_inputs else layer_input[:4]
    input_norm_dump = query[local_token_slice].reshape(2, -1) if local_token_inputs else query[:4].reshape(4, -1)
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "model.forward_batch_info.input_ids": torch.tensor([10, 11, 12, 13]),
            "model.forward_batch_info.positions": torch.tensor([0, 1, 2, 3]),
            "model.forward_batch_info.rids": ["rollout-0"],
            "model.forward_batch_info.extend_seq_lens": torch.tensor([4]),
            "model.layers.0.input_layernorm.input": layer_input_dump,
            "model.layers.0.input_layernorm": input_norm_dump,
            "model.layers.0.self_attn.qkv_proj": torch.cat(
                (
                    query_slice.reshape(4, -1),
                    key_slice.reshape(4, -1),
                    value_slice.reshape(4, -1),
                ),
                dim=-1,
            ),
            "model.layers.0.self_attn.q_norm.input": query_slice.reshape(-1, 2),
            "model.layers.0.self_attn.k_norm.input": key_slice.reshape(-1, 2),
            "model.layers.0.self_attn.q_norm": query_slice.reshape(-1, 2),
            "model.layers.0.self_attn.k_norm": key_slice.reshape(-1, 2),
            "model.layers.0.self_attn.attn.input": q_rope[:4, rank * 2 : (rank + 1) * 2].reshape(4, -1),
            "model.layers.0.self_attn.attn.input.0": q_rope[:4, rank * 2 : (rank + 1) * 2].reshape(4, -1),
            "model.layers.0.self_attn.attn.input.1": key_slice.reshape(4, -1),
            "model.layers.0.self_attn.attn.input.2": value_slice.reshape(4, -1),
            "model.layers.0.self_attn.attn": core_attention[:4, rank * 2 : (rank + 1) * 2].reshape(4, -1),
            "model.layers.0.post_attention_layernorm.input": (output_projection[rank * 2 : (rank + 1) * 2]),
            "model.layers.0.post_attention_layernorm.output.1": (post_attention_residual[rank * 2 : (rank + 1) * 2]),
            "model.layers.0.post_attention_layernorm": (pre_mlp_norm[rank * 2 : (rank + 1) * 2]),
        },
        path,
    )


def test_attention_parameter_comparison_maps_interleaved_qkv(tmp_path):
    hidden_size = 3
    query = torch.arange(hidden_size * 4 * 2, dtype=torch.bfloat16).reshape(hidden_size, 4, 2)
    key = (100 + torch.arange(hidden_size * 2 * 2)).to(torch.bfloat16).reshape(hidden_size, 2, 2)
    value = (200 + torch.arange(hidden_size * 2 * 2)).to(torch.bfloat16).reshape(hidden_size, 2, 2)
    input_norm_weight = torch.arange(8, dtype=torch.bfloat16)
    q_norm_weight = torch.arange(2, dtype=torch.bfloat16)
    k_norm_weight = torch.arange(2, dtype=torch.bfloat16) + 2
    output_projection_weight = torch.arange(40, dtype=torch.bfloat16).reshape(5, 8)
    pre_mlp_norm_weight = torch.arange(5, dtype=torch.bfloat16)
    dump_files = []
    for rank in range(2):
        query_slice = query[:, rank * 2 : (rank + 1) * 2]
        key_slice = key[:, rank : rank + 1]
        value_slice = value[:, rank : rank + 1]
        local_qkv_weight = torch.cat(
            (
                query_slice.reshape(hidden_size, -1),
                key_slice.reshape(hidden_size, -1),
                value_slice.reshape(hidden_size, -1),
            ),
            dim=-1,
        ).T.contiguous()
        dump_file = tmp_path / f"TP{rank}_PP0_Rank{rank}_pid1" / "Pass00000.pt"
        dump_file.parent.mkdir(parents=True)
        torch.save(
            {
                "model.layers.0.input_layernorm.parameter.weight": (input_norm_weight),
                "model.layers.0.self_attn.qkv_proj.parameter.weight": (local_qkv_weight),
                "model.layers.0.self_attn.q_norm.parameter.weight": (q_norm_weight),
                "model.layers.0.self_attn.k_norm.parameter.weight": (k_norm_weight),
                "model.layers.0.self_attn.o_proj.parameter.weight": (output_projection_weight[:, rank * 4 : (rank + 1) * 4]),
                "model.layers.0.post_attention_layernorm.parameter.weight": (pre_mlp_norm_weight),
            },
            dump_file,
        )
        dump_files.append(dump_file)

    stats = compare_attention_parameters(
        dump_files,
        {
            0: {
                "input_norm_weight": input_norm_weight,
                "qkv_weight": _pack_megatron_qkv(query, key, value).T,
                "q_norm_weight": q_norm_weight,
                "k_norm_weight": k_norm_weight,
                "output_projection_weight": output_projection_weight,
                "pre_mlp_norm_weight": pre_mlp_norm_weight,
            }
        },
        {0},
        attn_tp_size=2,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
    )

    assert stats[0]["replicas"] == 2
    assert stats[0]["attn_tp_rank_counts"] == {0: 1, 1: 1}
    assert stats[0]["input_norm_weight"]["exact"]
    assert stats[0]["qkv_weight"]["exact"]
    assert stats[0]["q_norm_weight"]["exact"]
    assert stats[0]["k_norm_weight"]["exact"]
    assert stats[0]["output_projection_weight"]["exact"]
    assert stats[0]["pre_mlp_norm_weight"]["exact"]


@pytest.mark.parametrize("local_token_inputs", [False, True])
def test_attention_comparison_maps_token_full_head_partitioned_tp(tmp_path, local_token_inputs):
    query = torch.arange(5 * 4 * 2, dtype=torch.float32).reshape(5, 4, 2)
    key = 100 + torch.arange(5 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2)
    value = 200 + torch.arange(5 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2)
    q_rope = query + 0.25
    core_attention = query + 0.5
    layer_input = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 4
    output_projection = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    post_attention_residual = output_projection + 10
    pre_mlp_norm = output_projection + 20
    train_sequence = TrainAttentionSequence(
        tokens=torch.tensor([10, 11, 12, 13, 14]),
        attention={
            0: {
                "layer_input": layer_input,
                "input_norm": query.reshape(5, -1),
                "qkv": _pack_megatron_qkv(query, key, value),
                "q_norm_input": query,
                "k_norm_input": key,
                "q_norm": query,
                "k_norm": key,
                "q_rope": q_rope,
                "k_rope": key,
                "value": value,
                "core_attention": core_attention,
                "output_projection": output_projection,
                "post_attention_residual": post_attention_residual,
                "pre_mlp_norm": pre_mlp_norm,
            }
        },
        source="train",
    )

    dump_files = []
    for rank in range(2):
        dump_file = tmp_path / f"TP{rank}_PP0_Rank{rank}_pid1" / "Pass00000.pt"
        _write_sglang_dump(
            dump_file,
            rank,
            query,
            key,
            value,
            q_rope,
            core_attention,
            layer_input,
            output_projection,
            post_attention_residual,
            pre_mlp_norm,
            local_token_inputs=local_token_inputs,
        )
        dump_files.append(dump_file)

    stats = compare_attention_outputs(
        dump_files,
        [train_sequence],
        {"rollout-0": 0},
        {0},
        attn_tp_size=2,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
    )

    assert stats[0]["token_head_partitions"] == 8
    assert stats[0]["rank_tokens"] == {0: 4, 1: 4}
    for field in (
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
        "output_projection",
        "post_attention_residual",
        "pre_mlp_norm",
        "pre_mlp_norm_input_fp32",
    ):
        assert stats[0][field]["max_abs"] == 0.0
        assert stats[0][field]["exact"]
        assert stats[0][field]["mismatched_elements"] == 0
