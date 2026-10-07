"""Experiment A: DeepEP LL -> SGLang Triton BF16 MoE -> DeepEP LL.

Run with, for example:

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc-per-node=4 \
        tests/test_deepep_ll_triton_a100_e2e.py

The DeepEP build and SGLang checkout are selected through PYTHONPATH by the
caller. This test intentionally uses identical weights for every expert so an
independent dense PyTorch SwiGLU computation is the end-to-end reference.
"""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _enable_a100_p2p_only_mode() -> None:
    """Configure DeepEP low-latency for one NVLink-connected A100 node."""

    os.environ["DEEPEP_LL_P2P_ONLY"] = "1"
    os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "0"
    os.environ["NVSHMEM_REMOTE_TRANSPORT"] = "none"
    os.environ["NVSHMEM_DISABLE_P2P"] = "0"


def _make_weights(num_local_experts: int, hidden: int, intermediate: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260815)
    gate = torch.randn(
        (intermediate, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
    ) / 32
    up = torch.randn(
        (intermediate, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
    ) / 32
    down = torch.randn(
        (hidden, intermediate), device="cuda", dtype=torch.bfloat16, generator=generator
    ) / 32
    w13 = torch.cat((gate, up), dim=0).unsqueeze(0).expand(
        num_local_experts, -1, -1
    ).contiguous()
    w2 = down.unsqueeze(0).expand(num_local_experts, -1, -1).contiguous()
    return gate, up, down, w13, w2


def main() -> None:
    _enable_a100_p2p_only_mode()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    import deep_ep
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import (
        TritonMoeQuantInfo,
        TritonRunnerCore,
        post_permute_triton_to_deepep_ll,
        pre_permute_deepep_ll_to_triton,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput

    # A standalone kernel test does not publish the server-wide runtime
    # context. Supply only the two read-only flags consulted by fused_moe.
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe_triton_config,
    )

    exec_config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
        moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=False),
    )
    fused_moe.get_exec = lambda: exec_config
    fused_moe_triton_config.get_exec = lambda: exec_config

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    assert local_world_size == world_size, "A100 P2P-only mode does not support multiple nodes"
    assert world_size <= 8, "A100 P2P-only mode is restricted to one NVLink node"
    num_tokens = 4
    hidden = 2048
    intermediate = 256
    num_topk = 2
    num_experts = 8
    assert num_experts % world_size == 0
    num_local_experts = num_experts // world_size

    generator = torch.Generator(device="cuda")
    generator.manual_seed(1000 + rank)
    x = torch.randn(
        (num_tokens, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    token_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    topk_ids = torch.stack(
        ((token_ids + rank) % num_experts, (token_ids + rank + 3) % num_experts),
        dim=1,
    )
    topk_weights = torch.tensor(
        [[0.25, 0.75]], device="cuda", dtype=torch.float32
    ).expand(num_tokens, num_topk).contiguous()

    rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, world_size, num_experts
    )
    buffer = deep_ep.Buffer(
        dist.group.WORLD,
        num_rdma_bytes=rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_local_experts,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
    )

    recv_x, recv_count, handle, _, _ = buffer.low_latency_dispatch(
        x,
        topk_ids,
        num_tokens,
        num_experts,
        use_fp8=False,
        async_finish=False,
    )
    dispatch_output = DeepEPLLDispatchOutput(
        hidden_states=recv_x,
        hidden_states_scale=None,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        masked_m=recv_count,
        expected_m=(num_tokens * world_size * num_topk + num_experts) // num_experts,
    )

    gate, up, down, w13, w2 = _make_weights(
        num_local_experts, hidden, intermediate
    )
    quant_info = TritonMoeQuantInfo(w13_weight=w13, w2_weight=w2)
    runner_config = MoeRunnerConfig(
        num_experts=num_local_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=1,
        params_dtype=torch.bfloat16,
        inplace=True,
    )
    running_state = {}
    runner_input = pre_permute_deepep_ll_to_triton(
        dispatch_output, quant_info, runner_config, running_state
    )
    runner_output = TritonRunnerCore(runner_config).run(
        runner_input, quant_info, running_state
    )
    combine_input = post_permute_triton_to_deepep_ll(
        runner_output, quant_info, runner_config, running_state
    )
    combined, _, _ = buffer.low_latency_combine(
        combine_input.hidden_states,
        combine_input.topk_ids,
        combine_input.topk_weights,
        handle,
        async_finish=False,
    )
    torch.cuda.synchronize()

    reference = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
    reference = reference * topk_weights.sum(dim=1, keepdim=True)
    abs_diff = (combined.float() - reference.float()).abs()
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    finite = torch.isfinite(combined).all()
    stats = torch.stack((max_diff, mean_diff, finite.float()))
    all_stats = [torch.empty_like(stats) for _ in range(world_size)]
    dist.all_gather(all_stats, stats)

    print(
        f"rank={rank} recv_count={recv_count.cpu().tolist()} "
        f"compact_rows={runner_input.hidden_states.shape[0]} "
        f"max_abs_diff={max_diff.item():.8g} mean_abs_diff={mean_diff.item():.8g}",
        flush=True,
    )
    assert finite.item()
    assert max_diff.item() <= 0.0625
    if rank == 0:
        print(
            "EXPERIMENT_A_PASS "
            + " ".join(
                f"rank{i}:max={item[0].item():.8g},mean={item[1].item():.8g}"
                for i, item in enumerate(all_stats)
            ),
            flush=True,
        )

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
