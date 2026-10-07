import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(local_rank: int) -> None:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29631",
        world_size=4,
        rank=local_rank,
        device_id=torch.device("cuda", local_rank),
    )
    import deep_ep

    group = dist.new_group(list(range(4)))
    buffer = deep_ep.Buffer(
        group,
        62915456,
        0,
        num_qps_per_rank=20,
        explicitly_destroy=True,
    )
    config = deep_ep.Config(20, 9, 256, 6, 128)
    # Mirror DP-attention's sparse source layout: only EP rank 0 has a local
    # token; every route targets EP ranks 1--3, leaving EP rank 0 with no
    # received expert activation.  Empty top-k tensors remain slices of an
    # allocated CUDA tensor, as they do in SGLang.
    num_tokens = int(local_rank == 0)
    x = torch.full((num_tokens, 2048), float(local_rank), device="cuda", dtype=torch.bfloat16)
    all_topk_ids = torch.tensor(
        [[32, 33, 34, 64, 65, 66, 96, 97]], device="cuda", dtype=torch.int64
    )
    all_topk_weights = torch.full((1, 8), 0.125, device="cuda", dtype=torch.float32)
    topk_ids = all_topk_ids[:num_tokens]
    topk_weights = all_topk_weights[:num_tokens]
    previous = buffer.capture()
    counts, _, expert_counts, in_rank, _ = buffer.get_dispatch_layout(
        topk_ids,
        128,
        previous_event=previous,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    recv_x, _, recv_weights, _, handle, event = buffer.dispatch(
        x,
        num_tokens_per_rank=counts,
        is_token_in_rank=in_rank,
        num_tokens_per_expert=expert_counts,
        topk_idx=topk_ids,
        topk_weights=topk_weights,
        config=config,
        previous_event=previous,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    event.current_stream_wait()
    combined, combined_weights, event = buffer.combine(
        recv_x,
        handle,
        topk_weights=recv_weights,
        config=config,
        previous_event=buffer.capture(),
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    event.current_stream_wait()
    torch.cuda.synchronize()
    assert torch.isfinite(combined).all()
    assert torch.isfinite(combined_weights).all()
    if local_rank == 0:
        print("DEEPEP_NORMAL_EP4_SPARSE_SOURCE_PASS", flush=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, nprocs=4)
