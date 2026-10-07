"""EP4 normal-DeepEP route-fingerprint gate for one Qwen3 MoE layer.

Run:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
        tests/test_qwen3_deepep_metadata_ep4.py
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    from megatron.core.transformer.moe.fused_a2a import fused_combine, fused_dispatch
    from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import (
        _SGLangEPGatherWithBF16Backward,
        _deepep_route_handle_received_rows,
        _dispatch_route_preserving_deepep_metadata,
        _scatter_deepep_routes_with_padding,
        _validate_and_order_route_preserving_outputs,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4, f"This receive-layout gate requires EP4, got EP{world_size}"
    num_tokens, hidden_size, num_experts, topk = 4, 128, 8, 2
    token_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    hidden_states = (
        torch.arange(num_tokens * hidden_size, device="cuda", dtype=torch.float32)
        .reshape(num_tokens, hidden_size)
        .add_(rank * 1024)
        .to(torch.bfloat16)
    )
    topk_indices = torch.stack(
        ((token_ids + rank) % num_experts, (token_ids + rank + 4) % num_experts), dim=1
    ).to(torch.int64)
    topk_weights = torch.tensor(
        [[0.125, 0.875], [0.25, 0.75], [0.375, 0.625], [0.5, 0.5]],
        device="cuda",
        dtype=torch.float32,
    )

    recv_hidden, recv_indices, recv_weights, tokens_per_expert, primary_handle = fused_dispatch(
        hidden_states,
        topk_indices,
        topk_weights,
        num_experts,
        dist.group.WORLD,
    )
    manager = SimpleNamespace(group=dist.group.WORLD, num_experts=num_experts)
    (
        metadata_handle,
        recv_fingerprints,
        recv_route_indices,
        recv_route_weights,
        source_output_index,
        source_all_routes_valid,
    ) = _dispatch_route_preserving_deepep_metadata(
        manager,
        hidden_states,
        topk_indices,
        topk_weights,
        assume_all_routes_valid=True,
    )
    expected_routes = _deepep_route_handle_received_rows(metadata_handle)
    (
        expert_inputs,
        _,
        output_index,
        sanitized_indices,
        _,
        _,
        route_positions,
    ) = _scatter_deepep_routes_with_padding(
        recv_hidden,
        recv_indices,
        recv_weights,
        tokens_per_expert,
        return_route_positions=True,
        expected_route_count=expected_routes,
    )
    ordered_routes = _validate_and_order_route_preserving_outputs(
        expert_inputs,
        recv_hidden,
        sanitized_indices,
        recv_weights,
        output_index,
        recv_fingerprints,
        recv_route_indices,
        recv_route_weights,
        route_positions=route_positions,
    )
    combined_routes, _ = fused_combine(
        ordered_routes,
        dist.group.WORLD,
        metadata_handle,
    )
    restored = _SGLangEPGatherWithBF16Backward.apply(
        combined_routes,
        topk_indices,
        topk_weights,
        source_output_index,
        False,
        source_all_routes_valid,
    )
    torch.cuda.synchronize()
    combine_max_abs = (restored.float() - hidden_states.float()).abs().max()

    stats = torch.tensor(
        [expected_routes, expert_inputs.shape[0]],
        device="cuda",
        dtype=torch.int64,
    )
    gathered = [torch.empty_like(stats) for _ in range(world_size)]
    dist.all_gather(gathered, stats)
    gathered_diffs = [torch.empty_like(combine_max_abs) for _ in range(world_size)]
    dist.all_gather(gathered_diffs, combine_max_abs)
    if rank == 0:
        rank_stats = [
            {
                "rank": rank_id,
                "received_routes": int(item[0].item()),
                "expert_rows": int(item[1].item()),
                "identity_combine_max_abs": float(gathered_diffs[rank_id].item()),
            }
            for rank_id, item in enumerate(gathered)
        ]
        passed = all(item["identity_combine_max_abs"] == 0.0 for item in rank_stats)
        payload = {
            "gate": "qwen3_deepep_metadata_ep4",
            "world_size": world_size,
            "source_tokens_per_rank": num_tokens,
            "topk": topk,
            "passed": passed,
            "ranks": rank_stats,
        }
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            "QWEN3_EP4_METADATA_BRIDGE_PASS "
            + " ".join(f"rank{i}:routes={item[0].item()}" for i, item in enumerate(gathered)),
            flush=True,
        )
        if not passed:
            raise AssertionError(f"DeepEP identity combine mismatch: {rank_stats}")

    del primary_handle
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
