import os

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"local_rank={local_rank} stage=before_nccl", flush=True)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    print(f"local_rank={local_rank} stage=after_nccl", flush=True)

    import deep_ep

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.group.WORLD
    num_tokens = 4
    hidden = 2048
    num_topk = 2
    num_experts = 8
    num_local_experts = num_experts // world_size

    x = torch.full(
        (num_tokens, hidden), rank + 1, dtype=torch.bfloat16, device="cuda"
    )
    x[:, -1] = torch.arange(num_tokens, device="cuda", dtype=torch.bfloat16)
    token_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    topk_idx = torch.stack(
        ((token_ids + rank) % num_experts, (token_ids + rank + 3) % num_experts),
        dim=1,
    )
    topk_weights = torch.tensor(
        [[0.25, 0.75]], dtype=torch.float32, device="cuda"
    ).expand(num_tokens, num_topk).contiguous()

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, world_size, num_experts
    )
    print(
        f"rank={rank} stage=before_buffer bytes={num_rdma_bytes}", flush=True
    )
    buffer = deep_ep.Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_local_experts,
        allow_nvlink_for_low_latency_mode=(
            os.environ.get("DEEPEP_LL_ALLOW_NVLINK", "1") == "1"
        ),
        explicitly_destroy=True,
    )
    print(f"rank={rank} stage=after_buffer", flush=True)
    recv_x, recv_count, handle, _, _ = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_tokens,
        num_experts,
        use_fp8=False,
        async_finish=False,
    )
    torch.cuda.synchronize()
    combined_x, _, _ = buffer.low_latency_combine(
        recv_x,
        topk_idx,
        topk_weights,
        handle,
        async_finish=False,
    )
    torch.cuda.synchronize()

    expected = x * topk_weights.sum(dim=1, keepdim=True)
    max_diff = (combined_x - expected).abs().max()
    print(
        f"rank={rank} counts={recv_count.cpu().tolist()} "
        f"max_diff={max_diff.item()}",
        flush=True,
    )
    assert max_diff.item() == 0.0

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
