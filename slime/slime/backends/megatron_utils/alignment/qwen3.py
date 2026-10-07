"""Qwen3 MoE alignment gates for Megatron training.

The first gate deliberately aligns only the router.  It reuses SGLang's
batch-invariant FP32 router GEMM while retaining Megatron's analytic backward.
The normal DeepEP metadata bridge is installed only after this gate is enabled.
"""

from __future__ import annotations

import json
import logging
import os
import re
import types
from copy import copy
from pathlib import Path

import torch

from slime.backends.megatron_utils.alignment.deepgemm_forward import (
    _SGLangRMSNormWithAnalyticBackward,
    _SGLangRouterGEMMWithMegatronBackward,
    _sum_to_parameter_dtype,
    enable_sglang_final_rmsnorm,
)
from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import (
    _get_expert_weights,
    _normalize_model_chunks,
    _validate_te_grouped_mlp,
    enable_sglang_deepep_moe_alignment,
)
from slime.utils.routing_replay import register_ordered_topk_capture

logger = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"^(?:.*\.)?decoder\.layers\.\d+$")


def _selected_layers(args) -> set[int]:
    layers = getattr(args, "qwen3_alignment_layers", None)
    if layers is None:
        layers = getattr(args, "megatron_deepgemm_moe_forward_layers", None)
    return {int(layer) for layer in (layers or ())}


def _validate_fixed_normal_deepep_config(args) -> None:
    """Reject per-rank auto-config for exact normal-DeepEP Qwen3 runs."""

    config_value = getattr(args, "sglang_deepep_config", None) or os.getenv(
        "SLIME_QWEN3_FIXED_DEEPEP_CONFIG"
    )
    if not config_value:
        raise RuntimeError(
            "Qwen3 exact normal-DeepEP alignment requires --sglang-deepep-config "
            "or SLIME_QWEN3_FIXED_DEEPEP_CONFIG with fixed normal_dispatch and normal_combine settings."
        )
    try:
        config = json.loads(config_value)
    except json.JSONDecodeError:
        try:
            config = json.loads(Path(config_value).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("--sglang-deepep-config must contain normal_dispatch and normal_combine Config values.") from error
    try:
        dispatch = config["normal_dispatch"]
        combine = config["normal_combine"]
        dispatch_sms = int(dispatch["num_sms"])
        combine_sms = int(combine["num_sms"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("--sglang-deepep-config must contain normal_dispatch and normal_combine Config values.") from error
    if dispatch_sms <= 0 or dispatch_sms != combine_sms:
        raise RuntimeError("Qwen3 exact normal-DeepEP alignment requires the same positive num_sms for dispatch and combine.")


def _ensure_sglang_runtime_context() -> None:
    """Publish the deterministic kernel config when running outside a server."""

    from sglang.srt.runtime_context import get_exec, publish

    try:
        get_exec()
        return
    except ValueError as error:
        if str(error) != "config namespace 'exec' not published":
            raise

    from sglang.srt.server_args import ServerArgs

    publish(
        ServerArgs(model_path="dummy", enable_deterministic_inference=True),
        role="launcher",
    )


class _SGLangBatchInvariantLinear(torch.autograd.Function):
    """Use SGLang's BF16 persistent GEMM with an ordinary BF16 backward."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if value.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError(f"Qwen3 aligned QKV requires BF16 input/weight, got {value.dtype}/{weight.dtype}")
        from sglang.srt.batch_invariant_ops import matmul_persistent

        input_shape = value.shape
        value_2d = value.contiguous().view(-1, input_shape[-1])
        output = matmul_persistent(value_2d, weight.T)
        ctx.input_shape = input_shape
        ctx.save_for_backward(value, weight)
        return output.view(*input_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        value, weight = ctx.saved_tensors
        grad_2d = grad_output.contiguous().view(-1, grad_output.shape[-1]).to(value.dtype)
        value_2d = value.contiguous().view(-1, value.shape[-1])
        grad_input = torch.mm(grad_2d, weight).view(ctx.input_shape) if ctx.needs_input_grad[0] else None
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = _sum_to_parameter_dtype(torch.mm(grad_2d.T, value_2d), weight)
        return grad_input, grad_weight


def _wrap_sglang_rms_norm(module: torch.nn.Module, module_name: str) -> bool:
    if getattr(module, "_slime_qwen3_rms_norm_wrapped", False):
        return False
    if not hasattr(module, "weight") or not hasattr(module, "eps"):
        raise TypeError(f"{module_name} must expose RMSNorm weight and eps")
    config = getattr(module, "config", None)
    if config is not None and (getattr(config, "normalization", None) != "RMSNorm" or bool(getattr(config, "layernorm_zero_centered_gamma", False))):
        raise ValueError(f"{module_name} requires ordinary RMSNorm")
    bias = getattr(module, "bias", None)
    if isinstance(bias, torch.Tensor) and bias.numel():
        raise ValueError(f"{module_name} requires bias-free RMSNorm")

    original_forward = module.forward

    def forward(patched_module, value: torch.Tensor):
        return _SGLangRMSNormWithAnalyticBackward.apply(
            value.detach(),
            value,
            patched_module.weight,
            float(patched_module.eps),
            False,
            value.dtype,
        )

    module.forward = types.MethodType(forward, module)
    module._slime_qwen3_rms_norm_wrapped = True
    module._slime_qwen3_rms_norm_original_forward = original_forward
    return True


def _wrap_sglang_pre_mlp_norm(layer: torch.nn.Module, module_name: str) -> bool:
    """Match SGLang's FP32 residual-add/native-RMSNorm boundary."""

    if getattr(layer, "_slime_qwen3_pre_mlp_norm_wrapped", False):
        return False
    module = getattr(layer, "pre_mlp_layernorm", None)
    if module is None or not hasattr(module, "weight") or not hasattr(module, "eps"):
        raise TypeError(f"{module_name} must expose RMSNorm weight and eps")
    config = getattr(module, "config", None)
    if config is not None and (getattr(config, "normalization", None) != "RMSNorm" or bool(getattr(config, "layernorm_zero_centered_gamma", False))):
        raise ValueError(f"{module_name} requires ordinary RMSNorm")
    bias = getattr(module, "bias", None)
    if isinstance(bias, torch.Tensor) and bias.numel():
        raise ValueError(f"{module_name} requires bias-free RMSNorm")
    if float(getattr(layer, "hidden_dropout", 0.0)) != 0.0:
        raise ValueError(f"{module_name} requires zero hidden dropout")
    if not callable(getattr(layer, "self_attn_bda", None)):
        raise TypeError(f"{module_name} requires a callable self_attn_bda factory")

    original_bda_factory = layer.self_attn_bda
    original_norm_forward = module.forward

    def self_attn_bda(patched_layer, training: bool, fused: bool):
        del patched_layer, training, fused

        def add_residual(attention_output_with_bias, residual, dropout):
            if float(dropout) != 0.0:
                raise ValueError(f"{module_name} requires zero residual dropout")
            if not isinstance(attention_output_with_bias, tuple) or len(attention_output_with_bias) != 2:
                raise TypeError(f"{module_name} received an unsupported attention output")
            attention_delta, attention_bias = attention_output_with_bias
            if not isinstance(attention_delta, torch.Tensor) or not isinstance(residual, torch.Tensor):
                raise TypeError(f"{module_name} residual inputs must be tensors")
            if attention_bias is not None:
                raise ValueError(f"{module_name} requires bias-free attention")
            exact_sum = attention_delta.float() + residual.float()
            output = exact_sum.to(attention_delta.dtype)
            output._slime_qwen3_pre_mlp_sum_fp32 = exact_sum
            return output

        return add_residual

    def forward(patched_module, value: torch.Tensor):
        exact_sum = getattr(value, "_slime_qwen3_pre_mlp_sum_fp32", None)
        if not isinstance(exact_sum, torch.Tensor):
            raise RuntimeError(f"{module_name} is missing its exact FP32 residual sum")
        return _SGLangRMSNormWithAnalyticBackward.apply(
            exact_sum.detach(),
            value,
            patched_module.weight,
            float(patched_module.eps),
            True,
            value.dtype,
        )

    layer.self_attn_bda = types.MethodType(self_attn_bda, layer)
    module.forward = types.MethodType(forward, module)
    layer._slime_qwen3_pre_mlp_norm_wrapped = True
    layer._slime_qwen3_pre_mlp_norm_original_bda_factory = original_bda_factory
    module._slime_qwen3_pre_mlp_norm_original_forward = original_norm_forward
    return True


def _wrap_sglang_layer_output_residual(
    layer: torch.nn.Module,
    module_name: str,
) -> bool:
    """Retain the FP32 MLP-delta/residual sum used by the next SGLang layer."""

    if getattr(layer, "_slime_qwen3_layer_output_residual_wrapped", False):
        return False
    if float(getattr(layer, "hidden_dropout", 0.0)) != 0.0:
        raise ValueError(f"{module_name} requires zero hidden dropout")
    if not callable(getattr(layer, "mlp_bda", None)):
        raise TypeError(f"{module_name} requires a callable mlp_bda factory")

    original_bda_factory = layer.mlp_bda

    def mlp_bda(patched_layer, training: bool, fused: bool):
        del training, fused

        def add_residual(mlp_output_with_bias, residual, dropout):
            if float(dropout) != 0.0:
                raise ValueError(f"{module_name} requires zero residual dropout")
            if not isinstance(mlp_output_with_bias, tuple) or len(mlp_output_with_bias) != 2:
                raise TypeError(f"{module_name} received an unsupported MLP output")
            mlp_delta, mlp_bias = mlp_output_with_bias
            if not isinstance(mlp_delta, torch.Tensor) or not isinstance(residual, torch.Tensor):
                raise TypeError(f"{module_name} residual inputs must be tensors")
            if mlp_bias is not None:
                raise ValueError(f"{module_name} requires bias-free MLP output")
            exact_sum = mlp_delta.float() + residual.float()
            patched_layer._slime_qwen3_layer_output_sum_fp32 = exact_sum
            return exact_sum.to(mlp_delta.dtype)

        return add_residual

    layer.mlp_bda = types.MethodType(mlp_bda, layer)
    layer._slime_qwen3_layer_output_residual_wrapped = True
    layer._slime_qwen3_layer_output_residual_original_bda_factory = original_bda_factory
    return True


def _wrap_sglang_input_norm_qkv(
    module: torch.nn.Module,
    module_name: str,
    *,
    residual_source: torch.nn.Module | None = None,
) -> bool:
    if getattr(module, "_slime_qwen3_input_norm_qkv_wrapped", False):
        return False
    if type(module).__name__ != "TELayerNormColumnParallelLinear":
        raise TypeError(f"{module_name} must be TELayerNormColumnParallelLinear, got {type(module).__name__}")
    config = getattr(module, "config", None)
    if config is None or getattr(config, "normalization", None) != "RMSNorm" or bool(getattr(config, "layernorm_zero_centered_gamma", False)):
        raise ValueError(f"{module_name} requires ordinary RMSNorm")
    if getattr(module, "layer_norm_bias", None) is not None:
        raise ValueError(f"{module_name} requires bias-free RMSNorm")
    if bool(getattr(module, "use_bias", False)):
        raise ValueError(f"{module_name} requires a bias-free QKV projection")

    original_forward = module.forward

    def forward(patched_module, value: torch.Tensor):
        exact_sum = None
        if residual_source is not None:
            exact_sum = getattr(
                residual_source,
                "_slime_qwen3_layer_output_sum_fp32",
                None,
            )
            if not isinstance(exact_sum, torch.Tensor):
                raise RuntimeError(
                    f"{module_name} is missing the previous layer's exact FP32 residual sum"
                )
        visible_input = value if exact_sum is None else exact_sum
        normalized = _SGLangRMSNormWithAnalyticBackward.apply(
            visible_input.detach(),
            value,
            patched_module.layer_norm_weight,
            float(config.layernorm_epsilon),
            exact_sum is not None,
            value.dtype,
        )
        output = _SGLangBatchInvariantLinear.apply(normalized, patched_module.weight)
        if getattr(patched_module, "return_layernorm_output", False):
            return (output, normalized), None
        return output, None

    module.forward = types.MethodType(forward, module)
    module._slime_qwen3_input_norm_qkv_wrapped = True
    module._slime_qwen3_input_norm_qkv_original_forward = original_forward
    return True


def _wrap_sglang_output_projection(
    module: torch.nn.Module,
    module_name: str,
    attn_tp_size: int,
) -> bool:
    """Reproduce SGLang's attention-TP row-parallel output projection."""

    if getattr(module, "_slime_qwen3_output_projection_wrapped", False):
        return False
    if type(module).__name__ != "TERowParallelLinear":
        raise TypeError(f"{module_name} must be TERowParallelLinear, got {type(module).__name__}")
    if attn_tp_size <= 0:
        raise ValueError("Qwen3 aligned attention TP size must be positive")
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError(f"{module_name} must expose a two-dimensional weight")
    if weight.shape[1] % attn_tp_size:
        raise ValueError(f"{module_name} input width {weight.shape[1]} is not divisible by attention TP {attn_tp_size}")
    bias = getattr(module, "bias", None)
    if (isinstance(bias, torch.Tensor) and bias.numel()) or bool(getattr(module, "use_bias", False)):
        raise ValueError(f"{module_name} requires a bias-free output projection")

    original_forward = module.forward

    def forward(patched_module, value: torch.Tensor):
        if value.shape[-1] != patched_module.weight.shape[1]:
            raise ValueError(f"{module_name} expected input width {patched_module.weight.shape[1]}, got {value.shape[-1]}")
        value_parts = value.chunk(attn_tp_size, dim=-1)
        weight_parts = patched_module.weight.chunk(attn_tp_size, dim=-1)
        partial_outputs = [_SGLangBatchInvariantLinear.apply(value_part, weight_part) for value_part, weight_part in zip(value_parts, weight_parts, strict=True)]
        output = partial_outputs[0]
        for partial_output in partial_outputs[1:]:
            output = output + partial_output
        return output, None

    module.forward = types.MethodType(forward, module)
    module._slime_qwen3_output_projection_original_forward = original_forward
    module._slime_qwen3_output_projection_wrapped = True
    return True


class _SGLangForwardMegatronBackward(torch.autograd.Function):
    """Keep the native backward while making the visible forward exactly SGLang."""

    @staticmethod
    def forward(ctx, native_output: torch.Tensor, triton_output: torch.Tensor) -> torch.Tensor:
        del ctx, native_output
        return triton_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        del ctx
        return grad_output, None


def _run_sglang_unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
) -> torch.Tensor:
    """Run the deterministic SGLang causal attention kernel on packed THD input."""

    tensors = {"query": query, "key": key, "value": value}
    for name, tensor in tensors.items():
        if tensor.ndim != 3:
            raise ValueError(f"Qwen3 aligned attention {name} must be THD, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"Qwen3 aligned attention {name} must be BF16, got {tensor.dtype}")
        if tensor.device != query.device:
            raise ValueError("Qwen3 aligned Q/K/V must be on the same device")
    if query.shape[0] != key.shape[0] or key.shape != value.shape:
        raise ValueError("Qwen3 aligned attention requires equal Q/K/V token counts and K/V shapes")
    if query.shape[-1] != key.shape[-1] or query.shape[1] % key.shape[1] != 0:
        raise ValueError("Qwen3 aligned attention received an unsupported GQA layout")

    qo_indptr = cu_seqlens_q.to(device=query.device, dtype=torch.int32).contiguous()
    kv_indptr = cu_seqlens_kv.to(device=query.device, dtype=torch.int32).contiguous()
    if qo_indptr.ndim != 1 or qo_indptr.numel() < 2:
        raise ValueError("Qwen3 aligned attention requires non-empty cu_seqlens")
    if not torch.equal(qo_indptr, kv_indptr):
        raise ValueError("Qwen3 aligned self-attention requires identical Q/KV cu_seqlens")
    sequence_lengths = qo_indptr[1:] - qo_indptr[:-1]
    if int(qo_indptr[0].item()) != 0 or int(qo_indptr[-1].item()) != query.shape[0] or bool(torch.any(sequence_lengths <= 0).item()):
        raise ValueError("Qwen3 aligned attention received invalid cu_seqlens")

    from sglang.kernels.ops.attention.extend_attention import (
        extend_attention_fwd_unified,
    )

    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    kv_indices = torch.arange(key.shape[0], dtype=torch.int64, device=query.device)
    prefix_lens = torch.zeros(qo_indptr.numel() - 1, dtype=torch.int32, device=query.device)
    extend_attention_fwd_unified(
        query.contiguous(),
        output,
        key.contiguous(),
        value.contiguous(),
        1.0,
        1.0,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        max_len_extend=int(sequence_lengths.max().item()),
        sm_scale=query.shape[-1] ** -0.5,
        logit_cap=0.0,
        is_causal=True,
        sliding_window_size=-1,
        page_size=1,
    )
    return output


def _wrap_sglang_core_attention(module: torch.nn.Module, module_name: str) -> bool:
    if getattr(module, "_slime_qwen3_core_attention_wrapped", False):
        return False
    if type(module).__name__ != "TEDotProductAttention":
        raise TypeError(f"{module_name} must be TEDotProductAttention, got {type(module).__name__}")
    config = getattr(module, "config", None)
    if config is None:
        raise TypeError(f"{module_name} must expose its TransformerConfig")
    if float(getattr(config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError(f"{module_name} requires zero attention dropout")
    if int(getattr(config, "context_parallel_size", 1)) != 1:
        raise ValueError(f"{module_name} requires context parallel size 1")
    if bool(getattr(config, "apply_query_key_layer_scaling", False)):
        raise ValueError(f"{module_name} does not support QK layer scaling")
    if getattr(config, "softmax_type", "vanilla") != "vanilla":
        raise ValueError(f"{module_name} requires vanilla softmax")
    if getattr(config, "window_size", None) is not None:
        raise ValueError(f"{module_name} does not support sliding-window attention")
    if bool(getattr(config, "qk_clip", False)) or bool(getattr(config, "log_max_attention_logit", False)):
        raise ValueError(f"{module_name} does not support QK clipping")

    original_forward = module.forward

    def forward(
        patched_module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        attn_mask_type,
        attention_bias=None,
        packed_seq_params=None,
        num_splits=None,
    ):
        del patched_module
        if attention_mask is not None or attention_bias is not None:
            raise ValueError(f"{module_name} requires bias-free packed attention")
        if getattr(attn_mask_type, "name", None) != "causal":
            raise ValueError(f"{module_name} requires causal attention")
        if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
            raise ValueError(f"{module_name} requires packed THD input")
        cu_seqlens_q = getattr(packed_seq_params, "cu_seqlens_q", None)
        cu_seqlens_kv = getattr(packed_seq_params, "cu_seqlens_kv", None)
        if not isinstance(cu_seqlens_q, torch.Tensor) or not isinstance(cu_seqlens_kv, torch.Tensor):
            raise TypeError(f"{module_name} requires Q/KV cu_seqlens tensors")

        with torch.no_grad():
            sglang_output = _run_sglang_unified_attention(
                query.detach(),
                key.detach(),
                value.detach(),
                cu_seqlens_q,
                cu_seqlens_kv,
            ).reshape(query.shape[0], -1)
        if not torch.is_grad_enabled():
            return sglang_output

        native_output = original_forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias,
            packed_seq_params,
            num_splits,
        )
        if not isinstance(native_output, torch.Tensor) or (native_output.shape != sglang_output.shape):
            raise TypeError(f"{module_name} returned an unsupported native output shape")
        return _SGLangForwardMegatronBackward.apply(native_output, sglang_output)

    module.forward = types.MethodType(forward, module)
    module._slime_qwen3_core_attention_original_forward = original_forward
    module._slime_qwen3_core_attention_wrapped = True
    return True


def _patch_router(router: torch.nn.Module, global_layer: int) -> bool:
    if getattr(router, "_slime_qwen3_router_alignment", False):
        return False
    if getattr(router.config, "moe_router_dtype", None) != "fp32":
        raise RuntimeError(f"Qwen3 router layer {global_layer} requires --moe-router-dtype fp32")
    if getattr(router.config, "moe_router_score_function", None) != "softmax":
        raise RuntimeError(f"Qwen3 router layer {global_layer} requires softmax routing")
    if bool(getattr(router.config, "moe_router_pre_softmax", False)):
        raise RuntimeError(f"Qwen3 router layer {global_layer} requires pre-softmax routing to be disabled")
    if getattr(router.config, "moe_expert_capacity_factor", None) is not None:
        raise RuntimeError(f"Qwen3 router layer {global_layer} does not support capacity dropping in exact mode")
    if router.bias is not None:
        raise RuntimeError(f"Qwen3 router layer {global_layer} requires a bias-free gate")

    original_gating = router.gating

    def gating(patched_router, hidden_states: torch.Tensor) -> torch.Tensor:
        del patched_router
        logits = _SGLangRouterGEMMWithMegatronBackward.apply(
            hidden_states,
            router.weight,
        )
        router._slime_qwen3_router_logits = logits.detach()
        return logits

    router.gating = types.MethodType(gating, router)
    router._slime_qwen3_router_original_gating = original_gating
    router._slime_qwen3_router_alignment = True
    register_ordered_topk_capture(router)
    return True


def _wrap_sglang_triton_moe(mlp: torch.nn.Module, global_layer: int) -> bool:
    """Run one local Qwen3 expert group with SGLang's BF16 Triton forward.

    The #2262 bridge defers route probabilities to its ordered owner gather.
    Therefore this runner receives one local-expert ID per expert-major row and
    unit route weights.  In a grad-enabled pass, Megatron's original MoE is
    retained solely for backward until the matching Triton backward is added.
    """

    experts = getattr(mlp, "experts", None)
    if experts is None or getattr(experts, "_slime_qwen3_triton_moe_wrapped", False):
        return False
    layout = _validate_te_grouped_mlp(experts, f"decoder.layers.{global_layer}.mlp.experts")
    fc1_weights = _get_expert_weights(
        experts.linear_fc1,
        num_local_experts=layout.num_local_experts,
        expected_shape=layout.fc1_weight_shape,
        module_name=f"Qwen3 layer {global_layer} linear_fc1",
    )
    fc2_weights = _get_expert_weights(
        experts.linear_fc2,
        num_local_experts=layout.num_local_experts,
        expected_shape=layout.fc2_weight_shape,
        module_name=f"Qwen3 layer {global_layer} linear_fc2",
    )
    original_forward = experts.forward

    def triton_moe_forward(
        patched_experts,
        permuted_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        with torch.no_grad():
            triton_output = _run_sglang_triton_moe(
                permuted_hidden_states,
                tokens_per_expert,
                permuted_probs,
                torch.stack(fc1_weights).contiguous(),
                torch.stack(fc2_weights).contiguous(),
                layout,
            )
        if not torch.is_grad_enabled():
            return triton_output, None
        native_output, native_bias = original_forward(permuted_hidden_states, tokens_per_expert, permuted_probs)
        return _SGLangForwardMegatronBackward.apply(native_output, triton_output), native_bias

    experts.forward = types.MethodType(triton_moe_forward, experts)
    experts._slime_qwen3_triton_moe_original_forward = original_forward
    experts._slime_defer_router_probabilities = False
    experts._slime_qwen3_triton_moe_wrapped = True
    return True


def _run_sglang_triton_moe(
    expert_inputs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    route_weights: torch.Tensor,
    fc1_weights: torch.Tensor,
    fc2_weights: torch.Tensor,
    layout,
) -> torch.Tensor:
    """Run SGLang Triton on Megatron's local expert-major receive layout."""

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import (
        TritonMoeQuantInfo,
        TritonRunnerCore,
        pre_permute_deepep_normal_to_triton,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalDispatchOutput

    counts = tokens_per_expert.detach().to(device="cpu", dtype=torch.long).tolist()
    if sum(counts) != expert_inputs.shape[0]:
        raise RuntimeError("Qwen3 Triton MoE received inconsistent expert-major row counts")
    local_ids = torch.arange(
        layout.num_local_experts,
        device=expert_inputs.device,
        dtype=torch.int64,
    ).repeat_interleave(torch.tensor(counts, device=expert_inputs.device))
    route_ids = local_ids.unsqueeze(1)
    if route_weights.shape != local_ids.shape or route_weights.dtype != torch.float32:
        raise TypeError(
            "Qwen3 Triton MoE route weights must be FP32 [expert_rows], got "
            f"{route_weights.dtype} {tuple(route_weights.shape)}"
        )
    dispatch_output = DeepEPNormalDispatchOutput(
        hidden_states=expert_inputs,
        hidden_states_scale=None,
        topk_ids=route_ids,
        topk_weights=route_weights.unsqueeze(1),
        num_recv_tokens_per_expert=counts,
    )
    runner_config = MoeRunnerConfig(
        num_experts=layout.num_local_experts,
        num_local_experts=layout.num_local_experts,
        hidden_size=layout.hidden_size,
        intermediate_size_per_partition=layout.ffn_hidden_size,
        top_k=1,
        params_dtype=torch.bfloat16,
        inplace=True,
    )
    quant_info = TritonMoeQuantInfo(w13_weight=fc1_weights, w2_weight=fc2_weights)
    running_state = {}
    runner_input = pre_permute_deepep_normal_to_triton(dispatch_output, quant_info, runner_config, running_state)
    return TritonRunnerCore(runner_config).run(runner_input, quant_info, running_state).hidden_states


def enable_qwen3_moe_alignment(args, model, store_prefix: str) -> None:
    """Enable the Qwen3 router gate and normal-DeepEP metadata validation."""

    if not bool(getattr(args, "deterministic_mode", False)):
        raise RuntimeError("Qwen3 exact alignment requires --deterministic-mode")
    if not bool(getattr(args, "moe_enable_deepep", False)):
        raise RuntimeError("Qwen3 exact alignment requires --moe-enable-deepep")
    _validate_fixed_normal_deepep_config(args)
    _ensure_sglang_runtime_context()
    selected_layers = _selected_layers(args)
    if not selected_layers:
        raise RuntimeError("Select one MoE layer with --qwen3-alignment-layers (or --megatron-deepgemm-moe-forward-layers)")

    matched = []
    patched = []
    triton_moe_wrapped = []
    layer_output_residual_wrapped = []
    input_norm_qkv_wrapped = []
    qk_norm_wrapped = []
    pre_mlp_norm_wrapped = []
    core_attention_wrapped = []
    output_projection_wrapped = []
    align_input_norm_qkv = os.getenv("MEGATRON_USE_SGLANG_QWEN3_INPUT_NORM_QKV", "0") == "1"
    align_qk_norm = os.getenv("MEGATRON_USE_SGLANG_QWEN3_QK_NORM", "0") == "1"
    align_pre_mlp_norm = os.getenv("MEGATRON_USE_SGLANG_QWEN3_PRE_MLP_NORM", "0") == "1"
    align_core_attention = os.getenv("MEGATRON_USE_SGLANG_QWEN3_CORE_ATTENTION", "0") == "1"
    align_output_projection = os.getenv("MEGATRON_USE_SGLANG_QWEN3_OUTPUT_PROJECTION", "0") == "1"
    sglang_attn_tp_size = int(os.getenv("MEGATRON_QWEN3_SGLANG_ATTENTION_TP_SIZE", "1"))
    for model_chunk in _normalize_model_chunks(model):
        layers_by_global = {}
        for candidate_name, candidate in model_chunk.named_modules():
            if _LAYER_RE.match(candidate_name) is None:
                continue
            candidate_number = getattr(candidate, "layer_number", None)
            if candidate_number is not None:
                layers_by_global[int(candidate_number) - 1] = (
                    candidate_name,
                    candidate,
                )

        for name, layer in model_chunk.named_modules():
            if _LAYER_RE.match(name) is None:
                continue
            layer_number = getattr(layer, "layer_number", None)
            if layer_number is None:
                continue
            global_layer = int(layer_number) - 1
            if global_layer not in selected_layers:
                continue
            self_attention = getattr(layer, "self_attention", None)
            if align_input_norm_qkv:
                linear_qkv = getattr(self_attention, "linear_qkv", None)
                if linear_qkv is None:
                    raise RuntimeError(f"{name}.self_attention.linear_qkv is missing")
                residual_source = None
                previous = layers_by_global.get(global_layer - 1)
                if previous is not None and global_layer - 1 in selected_layers:
                    previous_name, residual_source = previous
                    if _wrap_sglang_layer_output_residual(
                        residual_source,
                        previous_name,
                    ):
                        layer_output_residual_wrapped.append(global_layer - 1)
                module_name = name + ".self_attention.linear_qkv"
                if _wrap_sglang_input_norm_qkv(
                    linear_qkv,
                    module_name,
                    residual_source=residual_source,
                ):
                    input_norm_qkv_wrapped.append(global_layer)
            if align_qk_norm:
                for norm_name in ("q_layernorm", "k_layernorm"):
                    norm = getattr(self_attention, norm_name, None)
                    if norm is None:
                        raise RuntimeError(f"{name}.self_attention.{norm_name} is missing")
                    module_name = name + ".self_attention." + norm_name
                    if _wrap_sglang_rms_norm(norm, module_name):
                        qk_norm_wrapped.append(f"{global_layer}:{norm_name[0]}")
            if align_core_attention:
                core_attention = getattr(self_attention, "core_attention", None)
                if core_attention is None:
                    raise RuntimeError(f"{name}.self_attention.core_attention is missing")
                module_name = name + ".self_attention.core_attention"
                if _wrap_sglang_core_attention(core_attention, module_name):
                    core_attention_wrapped.append(global_layer)
            if align_output_projection:
                linear_proj = getattr(self_attention, "linear_proj", None)
                if linear_proj is None:
                    raise RuntimeError(f"{name}.self_attention.linear_proj is missing")
                module_name = name + ".self_attention.linear_proj"
                if _wrap_sglang_output_projection(
                    linear_proj,
                    module_name,
                    sglang_attn_tp_size,
                ):
                    output_projection_wrapped.append(global_layer)
            if align_pre_mlp_norm:
                module_name = name + ".pre_mlp_layernorm"
                if _wrap_sglang_pre_mlp_norm(layer, module_name):
                    pre_mlp_norm_wrapped.append(global_layer)
            router = getattr(getattr(layer, "mlp", None), "router", None)
            if router is not None:
                matched.append(global_layer)
                if _patch_router(router, global_layer):
                    patched.append(global_layer)
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and _wrap_sglang_triton_moe(mlp, global_layer):
                triton_moe_wrapped.append(global_layer)

    if not matched:
        raise RuntimeError(f"No Qwen3 MoE routers found for layers {sorted(selected_layers)}")
    # This is the #2262 normal-DeepEP route-fingerprint bridge.  It validates
    # the EP receive layout before any MoE output is declared exact.
    bridge_args = args
    if getattr(args, "megatron_deepgemm_moe_forward_layers", None) is None:
        bridge_args = copy(args)
        bridge_args.megatron_deepgemm_moe_forward_layers = sorted(selected_layers)
    enable_sglang_deepep_moe_alignment(bridge_args, model, store_prefix)
    # The generic DeepEP bridge defaults to applying router probabilities in
    # its gather. Qwen's SGLang Triton forward already fuses them into FC2, so
    # restore the ownership flag after the bridge has installed its hooks.
    for model_chunk in _normalize_model_chunks(model):
        for module in model_chunk.modules():
            if getattr(module, "_slime_qwen3_triton_moe_wrapped", False):
                module._slime_defer_router_probabilities = False
    enable_sglang_final_rmsnorm(args, model, store_prefix)
    logger.info(
        "Enabled Qwen3 alignment for global MoE layers %s (SGLang layer-output residual=%s, input-norm/QKV=%s, Q/K norm=%s, core attention=%s, output projection=%s, pre-MLP norm=%s, Triton MoE forward=%s)",
        sorted(matched),
        sorted(layer_output_residual_wrapped),
        sorted(input_norm_qkv_wrapped),
        sorted(qk_norm_wrapped),
        sorted(core_attention_wrapped),
        sorted(output_projection_wrapped),
        sorted(pre_mlp_norm_wrapped),
        sorted(triton_moe_wrapped),
    )


def enable_qwen3_moe_alignment_before_train_step(args, rollout_id: int, step_id: int, model, optimizer, opt_param_scheduler) -> None:
    """Install the same hooks before the first train step."""

    del rollout_id, step_id, optimizer, opt_param_scheduler
    enable_qwen3_moe_alignment(args, model, store_prefix="")
