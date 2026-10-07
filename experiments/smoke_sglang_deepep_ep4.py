"""GPU4-7 smoke gate for SGLang's DeepEP normal dispatcher.

Run from the SGLang checkout with four visible GPUs. This intentionally tests
the communication integration only: the current unquantized BF16 normal-mode
expert runner is disabled in SGLang.
"""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    from sglang.srt.layers.dp_attention import set_is_extend_in_batch
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import DeepEPMode
    from sglang.srt.runtime_context import get_context

    get_context().set_server_args(
        SimpleNamespace(deepep_dispatcher_output_dtype="bf16")
    )
    set_is_extend_in_batch(True)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4

    hidden_size = 2048
    num_experts = 128
    topk = 8
    num_tokens = 32
    dispatcher = DeepEPDispatcher(
        group=dist.group.WORLD,
        router_topk=topk,
        num_experts=num_experts,
        num_local_experts=num_experts // world_size,
        hidden_size=hidden_size,
        params_dtype=torch.bfloat16,
        deepep_mode=DeepEPMode.NORMAL,
    )

    values = torch.arange(num_tokens * hidden_size, device="cuda")
    hidden_states = ((values % 17) - 8 + rank).reshape(num_tokens, hidden_size)
    hidden_states = hidden_states.to(torch.bfloat16)
    topk_ids = torch.tensor(
        [0, 1, 32, 33, 64, 65, 96, 97], device="cuda", dtype=torch.int64
    ).expand(num_tokens, -1).contiguous()
    topk_weights = torch.full(
        (num_tokens, topk), 1.0 / topk, device="cuda", dtype=torch.float32
    )
    topk_output = StandardTopKOutput(
        topk_weights, topk_ids, torch.empty(0, device="cuda")
    )

    dispatched = dispatcher.dispatch(hidden_states, topk_output)
    local_weights = torch.where(
        dispatched.topk_ids >= 0,
        dispatched.topk_weights,
        torch.zeros_like(dispatched.topk_weights),
    ).sum(dim=-1)
    identity_expert_output = dispatched.hidden_states * local_weights.to(
        dispatched.hidden_states.dtype
    ).unsqueeze(-1)

    # SGLang's public combine wrapper rejects unquantized BF16 normal mode.
    # Call the same DeepEP combine primitive directly for this communication gate.
    combined, _ = dispatcher._normal_dispatcher._combine_core(
        identity_expert_output.contiguous(), previous_event=None
    )
    torch.cuda.synchronize()

    max_diff = (combined - hidden_states).abs().max().float()
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    recv_tokens = torch.tensor(dispatched.hidden_states.shape[0], device="cuda")
    dist.all_reduce(recv_tokens, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            "SGLANG_DEEPEP_NORMAL_EP4_OK "
            f"recv_tokens_total={recv_tokens.item()} max_diff={max_diff.item()}"
        )
    assert max_diff.item() == 0.0
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
