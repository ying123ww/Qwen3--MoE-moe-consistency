import torch

from slime.utils.compare_qwen3_router import (
    TrainRouterSequence,
    compare_router_outputs,
    load_train_router_sequences,
)


def test_qwen3_router_comparison_uses_matching_nonterminal_rows(tmp_path):
    megatron_file = tmp_path / "megatron" / "rank00000" / "actor_Pass00000.pt"
    megatron_file.parent.mkdir(parents=True)
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0]])
    torch.save(
        {
            "input_ids": torch.tensor([7, 8, 9, 0]),
            "cu_seqlens": torch.tensor([0, 3, 4]),
            "layers": {},
            "layer_inputs": {0: logits.unsqueeze(1)},
            "modules": {"decoder.layers.0.pre_mlp_layernorm": logits.unsqueeze(1)},
            "module_inputs": {"decoder.layers.0.pre_mlp_layernorm": logits.unsqueeze(1)},
        },
        megatron_file,
    )
    router_file = tmp_path / "router" / "rank00000" / "actor_RouterPass00000.pt"
    router_file.parent.mkdir(parents=True)
    topk_ids = torch.tensor([[1], [1], [1], [0]], dtype=torch.int32)
    topk_weights = torch.ones(4, 1)
    router_weight = torch.eye(2)
    torch.save(
        {
            "input_ids": torch.tensor([7, 8, 9, 0]),
            "routers": {
                0: {
                    "input": logits,
                    "weight": router_weight,
                    "logits": logits,
                    "topk_ids": topk_ids,
                    "topk_weights": topk_weights,
                }
            },
        },
        router_file,
    )
    sglang_file = tmp_path / "sglang" / "TP0_PP0_Rank0_pid1" / "Pass00000.pt"
    sglang_file.parent.mkdir(parents=True)
    torch.save(
        {
            "model.forward_batch_info.input_ids": torch.tensor([7, 8, 9]),
            "model.forward_batch_info.positions": torch.tensor([0, 1, 2]),
            "model.forward_batch_info.rids": ["rollout-0"],
            "model.forward_batch_info.extend_seq_lens": torch.tensor([3]),
            "model.layers.0.input": logits[:3],
            "model.layers.0.post_attention_layernorm.output.1": logits[:3],
            "model.layers.0.post_attention_layernorm": logits[:3],
            "model.layers.0.mlp.router_input": logits[:3],
            "model.layers.0.mlp.router_weight": router_weight,
            "model.layers.0.mlp.router_logits": logits[:3],
            "model.layers.0.mlp.topk_ids": topk_ids[:3],
            "model.layers.0.mlp.topk_weights": topk_weights[:3],
        },
        sglang_file,
    )

    train_sequences = load_train_router_sequences(tmp_path / "megatron", tmp_path / "router", {0})
    stats = compare_router_outputs([sglang_file], train_sequences, {"rollout-0": 0}, {0})

    assert stats[0]["tokens"] == 2
    assert stats[0]["layer_input"]["max_abs"] == 0.0
    assert stats[0]["post_attention_residual"]["max_abs"] == 0.0
    assert stats[0]["post_attention_norm"]["max_abs"] == 0.0
    assert stats[0]["input"]["max_abs"] == 0.0
    assert stats[0]["weight"]["exact"]
    assert stats[0]["logits"]["max_abs"] == 0.0
    assert stats[0]["topk_weights"]["max_abs"] == 0.0
    assert stats[0]["topk_ids"]["mismatched_elements"] == 0


def test_qwen3_router_comparison_maps_attention_tp_and_aligns_weights_by_expert(tmp_path):
    router_weight = torch.eye(3)
    inputs = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    logits = inputs / 10
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [0, 2], [1, 2], [0, 1]], dtype=torch.int32
    )
    topk_weights = torch.tensor(
        [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]]
    )
    train_sequence = TrainRouterSequence(
        tokens=torch.tensor([10, 11, 12, 13, 14]),
        routers={
            0: {
                "layer_input": inputs,
                "post_attention_residual": inputs,
                "post_attention_norm": inputs,
                "input": inputs,
                "weight": router_weight,
                "logits": logits,
                "topk_ids": topk_ids,
                "topk_weights": topk_weights,
            }
        },
        source="train",
    )

    sglang_file = tmp_path / "TP1_PP0_Rank1_pid1" / "Pass00000.pt"
    sglang_file.parent.mkdir(parents=True)
    torch.save(
        {
            "model.forward_batch_info.input_ids": torch.tensor([10, 11, 12, 13]),
            "model.forward_batch_info.positions": torch.tensor([0, 1, 2, 3]),
            "model.forward_batch_info.rids": ["rollout-0"],
            "model.forward_batch_info.extend_seq_lens": torch.tensor([4]),
            "model.layers.0.input": inputs[2:4],
            "model.layers.0.post_attention_layernorm.output.1": inputs[2:4],
            "model.layers.0.post_attention_layernorm": inputs[2:4],
            "model.layers.0.mlp.router_input": inputs[2:4],
            "model.layers.0.mlp.router_weight": router_weight,
            "model.layers.0.mlp.router_logits": logits[2:4],
            "model.layers.0.mlp.topk_ids": torch.tensor(
                [[2, 0], [1, 2]], dtype=torch.int32
            ),
            "model.layers.0.mlp.topk_weights": torch.tensor(
                [[0.3, 0.7], [0.6, 0.4]]
            ),
        },
        sglang_file,
    )

    stats = compare_router_outputs(
        [sglang_file],
        [train_sequence],
        {"rollout-0": 0},
        {0},
        sglang_attn_tp_size=2,
    )

    assert stats[0]["tokens"] == 2
    assert stats[0]["layer_input"]["max_abs"] == 0.0
    assert stats[0]["weight"]["exact"]
    assert stats[0]["topk_ids"]["mismatched_tokens"] == 1
    assert stats[0]["topk_ids"]["set_mismatched_tokens"] == 0
    assert stats[0]["topk_weights"]["matched_routes"] == 4
    assert stats[0]["topk_weights"]["unmatched_routes"] == 0
    assert stats[0]["topk_weights"]["max_abs"] == 0.0
