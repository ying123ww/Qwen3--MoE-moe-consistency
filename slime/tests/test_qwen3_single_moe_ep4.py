"""EP4 exact-output gate for one Qwen3 BF16 MoE layer.

This is deliberately a route-level test before a full model test:

1. normal DeepEP produces Megatron's received-token layout;
2. the #2262 metadata bridge verifies and expands it into expert-major rows;
3. the training wrapper's SGLang Triton path and SGLang's normal adapter run
   independently on that layout; and
4. both route results are returned through separate metadata handles and
   gathered with the same FP32 top-k weights.

Run:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
        tests/test_qwen3_single_moe_ep4.py
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def _install_standalone_sglang_runtime() -> None:
    """Supply the two read-only runtime flags used by SGLang Triton kernels."""

    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

    exec_config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
        moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=False),
    )
    fused_moe.get_exec = lambda: exec_config
    fused_moe_triton_config.get_exec = lambda: exec_config


def _normal_dispatch(expert_inputs, tokens_per_expert, route_weights):
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalDispatchOutput

    counts = tokens_per_expert.detach().to(device="cpu", dtype=torch.long).tolist()
    local_ids = torch.arange(
        len(counts), device=expert_inputs.device, dtype=torch.int64
    ).repeat_interleave(torch.tensor(counts, device=expert_inputs.device))
    return DeepEPNormalDispatchOutput(
        hidden_states=expert_inputs,
        hidden_states_scale=None,
        topk_ids=local_ids.unsqueeze(1),
        topk_weights=route_weights.unsqueeze(1),
        num_recv_tokens_per_expert=counts,
    )


def _run_rollout_normal_moe(
    expert_inputs, tokens_per_expert, route_weights, fc1, fc2, layout
):
    """Call SGLang's normal-DeepEP Triton adapter without the train wrapper."""

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import (
        TritonMoeQuantInfo,
        TritonRunnerCore,
        pre_permute_deepep_normal_to_triton,
    )

    config = MoeRunnerConfig(
        num_experts=layout.num_local_experts,
        num_local_experts=layout.num_local_experts,
        hidden_size=layout.hidden_size,
        intermediate_size_per_partition=layout.ffn_hidden_size,
        top_k=1,
        params_dtype=torch.bfloat16,
        inplace=True,
    )
    quant_info = TritonMoeQuantInfo(w13_weight=fc1, w2_weight=fc2)
    state = {}
    runner_input = pre_permute_deepep_normal_to_triton(
        _normal_dispatch(expert_inputs, tokens_per_expert, route_weights),
        quant_info,
        config,
        state,
    )
    return TritonRunnerCore(config).run(runner_input, quant_info, state).hidden_states


def _dispatch_metadata(manager, hidden_states, topk_ids, topk_weights):
    from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import (
        _dispatch_route_preserving_deepep_metadata,
    )

    return _dispatch_route_preserving_deepep_metadata(
        manager,
        hidden_states,
        topk_ids,
        topk_weights,
        assume_all_routes_valid=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    _install_standalone_sglang_runtime()

    from megatron.core.transformer.moe.fused_a2a import fused_combine, fused_dispatch
    from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import (
        _SGLangEPGatherWithBF16Backward,
        _deepep_route_handle_received_rows,
        _scatter_deepep_routes_with_padding,
        _validate_and_order_route_preserving_outputs,
    )
    from slime.backends.megatron_utils.alignment.qwen3 import _run_sglang_triton_moe

    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size == 4, f"This exact-output gate requires EP4, got EP{world_size}"
    num_tokens, hidden_size, intermediate_size, num_experts, topk = 4, 128, 128, 8, 2
    local_experts = num_experts // world_size
    layout = SimpleNamespace(
        num_local_experts=local_experts,
        hidden_size=hidden_size,
        ffn_hidden_size=intermediate_size,
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260818 + rank)
    hidden_states = torch.randn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    token_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    topk_ids = torch.stack(
        ((token_ids + rank) % num_experts, (token_ids + rank + 4) % num_experts), dim=1
    )
    topk_weights = torch.tensor(
        [[0.125, 0.875], [0.25, 0.75], [0.375, 0.625], [0.5, 0.5]],
        device="cuda",
        dtype=torch.float32,
    )
    fc1 = torch.randn(
        (local_experts, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ) / 32
    fc2 = torch.randn(
        (local_experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ) / 32

    recv_hidden, recv_ids, recv_weights, tokens_per_expert, primary_handle = fused_dispatch(
        hidden_states, topk_ids, topk_weights, num_experts, dist.group.WORLD
    )
    manager = SimpleNamespace(group=dist.group.WORLD, num_experts=num_experts)
    train_metadata = _dispatch_metadata(manager, hidden_states, topk_ids, topk_weights)
    rollout_metadata = _dispatch_metadata(manager, hidden_states, topk_ids, topk_weights)
    (
        train_handle,
        train_fingerprints,
        train_route_ids,
        train_route_weights,
        source_output_index,
        _,
    ) = train_metadata
    (
        rollout_handle,
        rollout_fingerprints,
        rollout_route_ids,
        rollout_route_weights,
        _,
        _,
    ) = rollout_metadata
    expected_routes = _deepep_route_handle_received_rows(train_handle)
    assert expected_routes == _deepep_route_handle_received_rows(rollout_handle)
    (
        expert_inputs,
        expert_probs,
        received_output_index,
        sanitized_ids,
        _,
        _,
        route_positions,
    ) = _scatter_deepep_routes_with_padding(
        recv_hidden,
        recv_ids,
        recv_weights,
        tokens_per_expert,
        return_route_positions=True,
        expected_route_count=expected_routes,
    )

    # This is the actual function installed by ``_wrap_sglang_triton_moe``.
    train_route_outputs = _run_sglang_triton_moe(
        expert_inputs.clone(), tokens_per_expert, expert_probs, fc1, fc2, layout
    )
    # This follows SGLang rollout's deepep_normal -> triton adapter directly.
    rollout_route_outputs = _run_rollout_normal_moe(
        expert_inputs.clone(), tokens_per_expert, expert_probs, fc1, fc2, layout
    )
    route_output_max_abs = (
        train_route_outputs.float() - rollout_route_outputs.float()
    ).abs().max()

    ordered_train = _validate_and_order_route_preserving_outputs(
        train_route_outputs,
        recv_hidden,
        sanitized_ids,
        recv_weights,
        received_output_index,
        train_fingerprints,
        train_route_ids,
        train_route_weights,
        route_positions=route_positions,
    )
    ordered_rollout = _validate_and_order_route_preserving_outputs(
        rollout_route_outputs,
        recv_hidden,
        sanitized_ids,
        recv_weights,
        received_output_index,
        rollout_fingerprints,
        rollout_route_ids,
        rollout_route_weights,
        route_positions=route_positions,
    )
    train_routes, _ = fused_combine(ordered_train, dist.group.WORLD, train_handle)
    rollout_routes, _ = fused_combine(ordered_rollout, dist.group.WORLD, rollout_handle)
    gather_weights = torch.ones_like(topk_weights)
    train_output = _SGLangEPGatherWithBF16Backward.apply(
        train_routes, topk_ids, gather_weights, source_output_index, False, True
    )
    rollout_output = _SGLangEPGatherWithBF16Backward.apply(
        rollout_routes, topk_ids, gather_weights, source_output_index, False, True
    )
    torch.cuda.synchronize()
    combined_output_max_abs = (
        train_output.float() - rollout_output.float()
    ).abs().max()

    stats = torch.tensor(
        [expected_routes, train_output.shape[0], train_output.shape[1]],
        device="cuda",
        dtype=torch.int64,
    )
    diffs = torch.stack((route_output_max_abs, combined_output_max_abs))
    gathered_stats = [torch.empty_like(stats) for _ in range(world_size)]
    gathered_diffs = [torch.empty_like(diffs) for _ in range(world_size)]
    dist.all_gather(gathered_stats, stats)
    dist.all_gather(gathered_diffs, diffs)

    if rank == 0:
        rank_stats = [
            {
                "rank": rank_id,
                "received_routes": int(item[0].item()),
                "output_shape": [int(item[1].item()), int(item[2].item())],
                "expert_route_output_max_abs": float(gathered_diffs[rank_id][0].item()),
                "combined_output_max_abs": float(gathered_diffs[rank_id][1].item()),
            }
            for rank_id, item in enumerate(gathered_stats)
        ]
        passed = all(
            item["expert_route_output_max_abs"] == 0.0
            and item["combined_output_max_abs"] == 0.0
            for item in rank_stats
        )
        payload = {
            "gate": "qwen3_single_moe_ep4",
            "world_size": world_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "topk": topk,
            "passed": passed,
            "ranks": rank_stats,
        }
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"QWEN3_EP4_SINGLE_MOE_EXACT_PASS routes_per_rank={expected_routes} "
            f"shape={tuple(train_output.shape)}",
            flush=True,
        )
        if not passed:
            raise AssertionError(f"Single-MoE exact gate mismatch: {rank_stats}")
    del primary_handle
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
