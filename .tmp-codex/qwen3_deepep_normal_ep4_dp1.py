import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29632",
        world_size=8,
        rank=rank,
        device_id=torch.device("cuda", rank),
    )
    import deep_ep

    group0 = dist.new_group(list(range(4)))
    group1 = dist.new_group(list(range(4, 8)))
    if rank >= 4:
        buffer = deep_ep.Buffer(
            group1,
            62915456,
            0,
            num_qps_per_rank=20,
            explicitly_destroy=True,
        )
        config = deep_ep.Config(20, 9, 256, 6, 128)
        x = torch.empty((0, 2048), device="cuda", dtype=torch.bfloat16)
        ids = torch.empty((0, 8), device="cuda", dtype=torch.int64)
        weights = torch.empty((0, 8), device="cuda", dtype=torch.float32)
        previous = buffer.capture()
        counts, _, expert_counts, in_rank, _ = buffer.get_dispatch_layout(
            ids, 128, previous_event=previous, async_finish=True,
            allocate_on_comm_stream=True,
        )
        recv_x, _, recv_weights, _, handle, event = buffer.dispatch(
            x, num_tokens_per_rank=counts, is_token_in_rank=in_rank,
            num_tokens_per_expert=expert_counts, topk_idx=ids, topk_weights=weights,
            config=config, previous_event=previous, async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        _, _, event = buffer.combine(
            recv_x, handle, topk_weights=recv_weights, config=config,
            previous_event=buffer.capture(), async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        torch.cuda.synchronize()
        buffer.destroy()
    dist.barrier()
    if rank == 0:
        print("DEEPEP_NORMAL_EP4_DP1_EMPTY_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, nprocs=8)
