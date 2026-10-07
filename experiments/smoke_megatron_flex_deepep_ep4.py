"""Qwen3-30B-A3B-shaped Megatron Flex+DeepEP EP4 communication gate.

The identity expert calculation verifies both the forward dispatch/combine and
their autograd backward path without allocating or running the full 30B model.
"""

import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def main() -> None:
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
    )
    _set_random_seed(seed_=123, data_parallel_random_init=False)

    cfg = TransformerConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        num_layers=1,
        hidden_size=2048,
        num_attention_heads=16,
        ffn_hidden_size=6144,
        num_moe_experts=128,
        moe_ffn_hidden_size=768,
        moe_router_topk=8,
        moe_router_load_balancing_type="none",
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="deepep",
        moe_router_dtype="fp32",
        moe_permute_fusion=False,
        use_cpu_initialization=False,
        add_bias_linear=False,
    )
    spec = get_gpt_layer_local_spec(num_experts=128, moe_grouped_gemm=False)
    layer = MoELayer(cfg, spec.submodules.mlp.submodules).cuda().to(torch.bfloat16)
    layer.set_layer_number(0)

    hidden_states = torch.randn(
        (32, 1, cfg.hidden_size), device="cuda", dtype=torch.bfloat16
    )
    hidden_states.requires_grad_(True)
    probs, indices = apply_module(layer.router)(hidden_states)
    probs = torch.ones_like(probs) / layer.router.topk

    dispatcher = layer.token_dispatcher
    dispatched, probs = dispatcher.dispatch_preprocess(hidden_states, indices, probs)
    dispatched, probs = dispatcher.token_dispatch(dispatched, probs)
    dispatched, _, permuted_probs = dispatcher.dispatch_postprocess(dispatched, probs)
    dispatched = dispatched * permuted_probs.unsqueeze(-1).to(dispatched.dtype)
    combined = dispatcher.combine_preprocess(dispatched)
    combined = dispatcher.token_combine(combined)
    combined = dispatcher.combine_postprocess(combined)

    max_diff = (combined - hidden_states).abs().max().float()
    torch.autograd.backward(combined, hidden_states)
    grad_diff = (hidden_states.grad - hidden_states.detach()).abs().max().float()
    torch.distributed.all_reduce(max_diff, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(grad_diff, op=torch.distributed.ReduceOp.MAX)
    if torch.distributed.get_rank() == 0:
        print(
            "MEGATRON_FLEX_DEEPEP_EP4_FWD_BWD_OK "
            f"max_diff={max_diff.item()} grad_diff={grad_diff.item()}"
        )
    torch.testing.assert_close(combined, hidden_states)
    torch.testing.assert_close(hidden_states.grad, hidden_states.detach())

    del layer
    torch.distributed.barrier()
    Utils.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
