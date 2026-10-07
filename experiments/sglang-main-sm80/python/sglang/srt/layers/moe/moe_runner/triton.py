from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_cuda, is_gfx95_supported, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class TritonRunnerInput(RunnerInput):

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonRunnerOutput(RunnerOutput):

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    use_mxfp8: bool = False
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False
    per_channel_quant: bool = False
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w13_zp: Optional[torch.Tensor] = None
    w2_zp: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None


class TritonRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> TritonRunnerOutput:
        if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
            from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
                fused_experts_mxfp8,
            )

            out = fused_experts_mxfp8(
                runner_input.hidden_states,
                quant_info.w13_weight,
                quant_info.w2_weight,
                runner_input.topk_weights,
                runner_input.topk_ids,
                quant_info.w13_scale,
                quant_info.w2_scale,
                b1=quant_info.b13,
                b2=quant_info.b2,
                activation=self.config.activation,
                is_gated=self.config.is_gated,
                no_combine=self.config.no_combine,
                inplace=self.config.inplace,
                apply_router_weight_on_input=self.config.apply_router_weight_on_input,
                routed_scaling_factor=self.config.routed_scaling_factor,
                gemm1_alpha=self.config.gemm1_alpha,
                gemm1_limit=self.config.gemm1_clamp_limit,
                swiglu_limit=self.config.swiglu_limit,
                gate_up_interleaved=self.config.gate_up_interleaved,
            )
            return TritonRunnerOutput(hidden_states=out)

        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )

        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _fused_moe_kernel_sequence,
        )

        filter_expert = (
            self.config.num_experts is None
            or self.config.num_experts != self.config.num_local_experts
        )

        out = _fused_moe_kernel_sequence(
            runner_input.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            runner_input.topk_weights,
            runner_input.topk_ids,
            runner_input.sorted_token_ids,
            runner_input.expert_ids,
            runner_input.num_tokens_post_padded,
            running_state["config"],
            running_state.get("down_config"),
            running_state.get("down_moe_use_tma", False),
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=quant_info.a13_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
            activation=self.config.activation,
            is_gated=self.config.is_gated,
            no_combine=self.config.no_combine,
            inplace=self.config.inplace,
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            routed_scaling_factor=self.config.routed_scaling_factor,
            gemm1_alpha=self.config.gemm1_alpha,
            gemm1_limit=self.config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            hooks=hooks,
            swiglu_limit=self.config.swiglu_limit,
        )

        return TritonRunnerOutput(hidden_states=out)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@register_fused_func("none", "triton")
def fused_experts_none_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
        from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
            fused_experts_mxfp8,
        )

        topk_weights, topk_ids, _ = dispatch_output.topk_output
        output = fused_experts_mxfp8(
            hidden_states=dispatch_output.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            b1=quant_info.b13,
            b2=quant_info.b2,
            activation=runner_config.activation,
            is_gated=runner_config.is_gated,
            no_combine=runner_config.no_combine,
            inplace=runner_config.inplace,
            apply_router_weight_on_input=runner_config.apply_router_weight_on_input,
            routed_scaling_factor=runner_config.routed_scaling_factor,
            gemm1_alpha=runner_config.gemm1_alpha,
            gemm1_limit=runner_config.gemm1_clamp_limit,
            swiglu_limit=runner_config.swiglu_limit,
            gate_up_interleaved=runner_config.gate_up_interleaved,
        )
    else:
        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )

        # SGLANG_OPT_MOE_QUANT_ONCE: use the caller's pre-quantized activation
        # (per-token-group-128 fp8 q + scales) instead of re-quantizing inside
        # invoke_fused_moe_kernel.
        pre_quant = dispatch_output.hidden_states_pre_quant
        if pre_quant is not None:
            a1_q, a1_scale = pre_quant
        else:
            a1_q, a1_scale = None, quant_info.a13_scale

        output = fused_experts(
            hidden_states=dispatch_output.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_output=dispatch_output.topk_output,
            moe_runner_config=runner_config,
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=a1_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
            a1_q=a1_q,
        )

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("standard", "triton")
def pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )

    assert TopKOutputChecker.format_is_standard(topk_output)

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_output.topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )

    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "standard")
def post_permute_triton_to_standard(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )


@register_pre_permute("deepep_normal", "triton")
def pre_permute_deepep_normal_to_triton(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )

    hidden_states, hidden_states_scale, topk_ids, topk_weights, _ = dispatch_output
    if hidden_states_scale is not None:
        raise ValueError(
            "Triton MoE with DeepEP normal requires BF16 dispatch output; "
            "set --deepep-dispatcher-output-dtype bf16."
        )
    empty_payload = hidden_states if hidden_states.numel() == 0 else None
    if empty_payload is None:
        runner_hidden_states = hidden_states
        runner_topk_ids = topk_ids.to(torch.int32)
        runner_topk_weights = topk_weights
    else:
        # A zero-row Triton launch on SM80 leaves an asynchronous invalid-grid
        # error that surfaces at the following DeepEP combine.  Run one ignored
        # route instead; post-permute returns the original empty payload.
        runner_hidden_states = hidden_states.new_zeros((1, hidden_states.shape[-1]))
        runner_topk_ids = torch.zeros(
            (1, topk_ids.shape[-1]), dtype=torch.int32, device=hidden_states.device
        )
        runner_topk_weights = torch.zeros(
            (1, topk_weights.shape[-1]), dtype=topk_weights.dtype, device=hidden_states.device
        )

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        runner_hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        runner_topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )

    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    # DeepEP normal combine needs a produced payload on every EP rank.  With
    # no locally received expert rows, Triton returns a fresh zero-row tensor
    # with no producer.  Preserve DeepEP's dispatch buffer instead: it has the
    # same empty shape but retains the dispatch stream/storage contract.
    running_state["deepep_normal_empty_payload"] = empty_payload

    return TritonRunnerInput(
        hidden_states=runner_hidden_states,
        topk_weights=runner_topk_weights,
        topk_ids=runner_topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "deepep_normal")
def post_permute_triton_to_deepep_normal(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPNormalCombineInput,
    )

    empty_payload = running_state["deepep_normal_empty_payload"]
    return DeepEPNormalCombineInput(
        hidden_states=(
            empty_payload.clone()
            if empty_payload is not None
            else runner_output.hidden_states
        ),
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("deepep_ll", "triton")
def pre_permute_deepep_ll_to_triton(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )

    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, _ = (
        dispatch_output
    )
    if hidden_states_scale is not None or hidden_states.dtype != torch.bfloat16:
        raise ValueError(
            "Triton MoE with DeepEP low latency requires BF16 dispatch output; "
            "set --deepep-dispatcher-output-dtype bf16."
        )
    if hidden_states.ndim != 3 or masked_m.ndim != 1:
        raise ValueError(
            "DeepEP low-latency dispatch must provide hidden_states [E, M, H] "
            "and masked_m [E]."
        )
    num_local_experts, capacity, hidden_size = hidden_states.shape
    if masked_m.shape[0] != num_local_experts:
        raise ValueError("masked_m must contain one count per local expert.")
    if runner_config.apply_router_weight_on_input:
        raise ValueError(
            "DeepEP low-latency Triton adapter requires output-side router weights."
        )

    row_ids = torch.arange(capacity, device=hidden_states.device)
    valid_mask = row_ids.unsqueeze(0) < masked_m.unsqueeze(1)
    valid_indices = valid_mask.flatten().nonzero(as_tuple=False).flatten()
    flat_hidden_states = hidden_states.reshape(-1, hidden_size)
    compact_hidden_states = flat_hidden_states.index_select(0, valid_indices)
    compact_topk_ids = (
        torch.arange(
            num_local_experts, device=hidden_states.device, dtype=torch.int32
        )
        .unsqueeze(1)
        .expand(num_local_experts, capacity)
        .reshape(-1)
        .index_select(0, valid_indices)
        .unsqueeze(1)
    )

    routed_scale = runner_config.routed_scaling_factor or 1.0
    compact_topk_weights = torch.full(
        (valid_indices.shape[0], 1),
        1.0 / routed_scale,
        dtype=torch.float32,
        device=hidden_states.device,
    )

    # Triton does not accept an empty token batch. Keep one ignored route so a
    # rank that receives no tokens can still participate in DeepEP combine.
    num_valid_rows = valid_indices.shape[0]
    if num_valid_rows == 0:
        compact_hidden_states = hidden_states.new_zeros((1, hidden_size))
        compact_topk_ids = torch.zeros(
            (1, 1), dtype=torch.int32, device=hidden_states.device
        )
        compact_topk_weights = torch.zeros(
            (1, 1), dtype=torch.float32, device=hidden_states.device
        )

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        compact_hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        compact_topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )

    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["ll_hidden_states_shape"] = hidden_states.shape
    running_state["ll_valid_indices"] = valid_indices
    running_state["ll_num_valid_rows"] = num_valid_rows

    return TritonRunnerInput(
        hidden_states=compact_hidden_states,
        topk_weights=compact_topk_weights,
        topk_ids=compact_topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "deepep_ll")
def post_permute_triton_to_deepep_ll(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    slab_shape = running_state["ll_hidden_states_shape"]
    valid_indices = running_state["ll_valid_indices"]
    num_valid_rows = running_state["ll_num_valid_rows"]
    output_slab = runner_output.hidden_states.new_zeros(slab_shape)
    output_slab.view(-1, slab_shape[-1]).index_copy_(
        0, valid_indices, runner_output.hidden_states[:num_valid_rows]
    )
    return DeepEPLLCombineInput(
        hidden_states=output_slab,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )
