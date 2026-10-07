import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("megatron")

from slime.backends.megatron_utils.alignment import qwen3
from slime.utils import routing_replay


_FIXED_DEEPEP_CONFIG = json.dumps(
    {
        "normal_dispatch": {
            "num_sms": 20,
            "num_max_nvl_chunked_send_tokens": 9,
            "num_max_nvl_chunked_recv_tokens": 256,
            "num_max_rdma_chunked_send_tokens": 6,
            "num_max_rdma_chunked_recv_tokens": 128,
        },
        "normal_combine": {
            "num_sms": 20,
            "num_max_nvl_chunked_send_tokens": 9,
            "num_max_nvl_chunked_recv_tokens": 256,
            "num_max_rdma_chunked_send_tokens": 6,
            "num_max_rdma_chunked_recv_tokens": 128,
        },
    }
)


class _Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12, dtype=torch.bfloat16).reshape(3, 4) / 8)
        self.bias = None
        self.config = SimpleNamespace(
            moe_router_dtype="fp32",
            moe_router_score_function="softmax",
            moe_router_pre_softmax=False,
            moe_expert_capacity_factor=None,
        )

    def gating(self, hidden_states):
        return torch.mm(hidden_states.float(), self.weight.float().t())

    def routing(self, logits):
        compute_topk = routing_replay.get_routing_replay_compute_topk(lambda scores, topk, num_groups=None, group_topk=None: torch.topk(scores, topk, dim=-1, sorted=True))
        weights, ids = compute_topk(logits, 2)
        dense = torch.zeros_like(logits).scatter(1, ids, torch.softmax(weights, dim=-1, dtype=torch.float32))
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, ids, True)
        return dense, routing_map

    def forward(self, hidden_states):
        return self.routing(self.gating(hidden_states))


def test_qwen3_router_uses_sglang_gemm_and_unordered_topk(monkeypatch):
    monkeypatch.delenv("ENABLE_ROUTING_REPLAY", raising=False)
    monkeypatch.setattr(routing_replay, "ORDERED_TOPK_CAPTURE_ROUTER", None)
    monkeypatch.setattr(
        qwen3._SGLangRouterGEMMWithMegatronBackward,
        "apply",
        staticmethod(lambda value, weight: torch.mm(value.float(), weight.float().t())),
    )
    router = _Router()
    assert qwen3._patch_router(router, global_layer=0)

    hidden_states = torch.tensor([[0.5, -1.0, 0.25, 2.0]], dtype=torch.bfloat16, requires_grad=True)
    probs, _ = router(hidden_states)
    expected_logits = torch.mm(hidden_states.float(), router.weight.float().t())
    expected_ids = torch.topk(expected_logits, 2, dim=-1, sorted=False).indices
    expected_weights = torch.softmax(expected_logits.gather(1, expected_ids), dim=-1, dtype=torch.float32)

    torch.testing.assert_close(router._slime_qwen3_router_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(router._slime_ordered_topk_indices, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(probs.gather(1, expected_ids), expected_weights, rtol=0, atol=0)
    probs.sum().backward()
    assert hidden_states.grad is not None


def test_qwen3_input_norm_qkv_wrapper_preserves_megatron_interface(monkeypatch):
    class TELayerNormColumnParallelLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                normalization="RMSNorm",
                layernorm_zero_centered_gamma=False,
                layernorm_epsilon=1e-6,
            )
            self.layer_norm_weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
            self.layer_norm_bias = None
            self.weight = nn.Parameter(torch.arange(12, dtype=torch.bfloat16).reshape(3, 4))
            self.bias = torch.empty(0)
            self.use_bias = False
            self.return_layernorm_output = False

        def forward(self, value):
            return value, None

    monkeypatch.setattr(
        qwen3._SGLangRMSNormWithAnalyticBackward,
        "apply",
        staticmethod(lambda visible, backward, weight, eps, use_fp32_sum, dtype: backward * 2),
    )
    monkeypatch.setattr(
        qwen3._SGLangBatchInvariantLinear,
        "apply",
        staticmethod(lambda value, weight: torch.nn.functional.linear(value, weight)),
    )
    module = TELayerNormColumnParallelLinear()
    assert qwen3._wrap_sglang_input_norm_qkv(module, "linear_qkv")

    value = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)
    output, bias = module(value)
    expected = torch.nn.functional.linear(value * 2, module.weight)
    torch.testing.assert_close(output, expected)
    assert bias is None

    module.return_layernorm_output = True
    (output, normalized), bias = module(value)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(normalized, value * 2)
    assert bias is None


def test_qwen3_input_norm_qkv_uses_previous_layer_fp32_residual_sum(monkeypatch):
    class TELayerNormColumnParallelLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                normalization="RMSNorm",
                layernorm_zero_centered_gamma=False,
                layernorm_epsilon=1e-6,
            )
            self.layer_norm_weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.layer_norm_bias = None
            self.weight = nn.Parameter(torch.eye(2, dtype=torch.bfloat16))
            self.bias = torch.empty(0)
            self.use_bias = False
            self.return_layernorm_output = False

        def forward(self, value):
            return value, None

    previous_layer = nn.Module()
    calls = []
    monkeypatch.setattr(
        qwen3._SGLangRMSNormWithAnalyticBackward,
        "apply",
        staticmethod(lambda visible, backward, weight, eps, use_fp32_sum, dtype: calls.append((visible, backward, use_fp32_sum)) or visible.to(dtype)),
    )
    monkeypatch.setattr(
        qwen3._SGLangBatchInvariantLinear,
        "apply",
        staticmethod(lambda value, weight: torch.nn.functional.linear(value, weight)),
    )
    module = TELayerNormColumnParallelLinear()
    assert qwen3._wrap_sglang_input_norm_qkv(
        module,
        "linear_qkv",
        residual_source=previous_layer,
    )

    rounded = torch.tensor([[1.0, 0.5]], dtype=torch.bfloat16)
    exact_sum = torch.tensor([[1.00390625, 0.50390625]], dtype=torch.float32)
    previous_layer._slime_qwen3_layer_output_sum_fp32 = exact_sum
    output, _ = module(rounded)

    torch.testing.assert_close(output, exact_sum.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(calls[0][0], exact_sum, rtol=0, atol=0)
    torch.testing.assert_close(calls[0][1], rounded, rtol=0, atol=0)
    assert calls[0][2] is True


def test_qwen3_layer_output_retains_unrounded_fp32_residual_sum():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dropout = 0.0

        def mlp_bda(self, training, fused):
            del training, fused
            return lambda mlp_output_with_bias, residual, dropout: mlp_output_with_bias[0] + residual

    layer = Layer()
    assert qwen3._wrap_sglang_layer_output_residual(layer, "layer")
    assert not qwen3._wrap_sglang_layer_output_residual(layer, "layer")

    mlp_delta = torch.tensor([[1.0, 0.5]], dtype=torch.bfloat16)
    residual = torch.tensor([[0.00390625, 0.00390625]], dtype=torch.bfloat16)
    exact_sum = mlp_delta.float() + residual.float()
    output = layer.mlp_bda(True, True)((mlp_delta, None), residual, 0.0)

    torch.testing.assert_close(output, exact_sum.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(
        layer._slime_qwen3_layer_output_sum_fp32,
        exact_sum,
        rtol=0,
        atol=0,
    )


def test_qwen3_output_projection_reproduces_attention_tp(monkeypatch):
    class TERowParallelLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(24, dtype=torch.bfloat16).reshape(3, 8))
            self.bias = torch.empty(0)
            self.use_bias = False

        def forward(self, value):
            return torch.nn.functional.linear(value, self.weight), None

    calls = []

    def linear(value, weight):
        calls.append((value, weight))
        return torch.nn.functional.linear(value, weight)

    monkeypatch.setattr(
        qwen3._SGLangBatchInvariantLinear,
        "apply",
        staticmethod(linear),
    )
    module = TERowParallelLinear()
    assert qwen3._wrap_sglang_output_projection(
        module,
        "linear_proj",
        attn_tp_size=2,
    )
    assert not qwen3._wrap_sglang_output_projection(
        module,
        "linear_proj",
        attn_tp_size=2,
    )

    value = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    output, bias = module(value)
    expected = torch.nn.functional.linear(value[:, :4], module.weight[:, :4]) + torch.nn.functional.linear(value[:, 4:], module.weight[:, 4:])

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert bias is None
    assert [tuple(item[0].shape) for item in calls] == [(2, 4), (2, 4)]


def test_qwen3_rms_norm_wrapper_uses_batch_invariant_kernel(monkeypatch):
    class TENorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                normalization="RMSNorm",
                layernorm_zero_centered_gamma=False,
            )
            self.weight = nn.Parameter(torch.linspace(0.5, 1.5, 4, dtype=torch.bfloat16))
            self.bias = torch.empty(0)
            self.eps = 1e-6

        def forward(self, value):
            return value

    calls = []
    monkeypatch.setattr(
        qwen3._SGLangRMSNormWithAnalyticBackward,
        "apply",
        staticmethod(lambda visible, backward, weight, eps, use_fp32_sum, dtype: calls.append((visible, backward, weight, eps, use_fp32_sum, dtype)) or backward * 3),
    )
    module = TENorm()
    assert qwen3._wrap_sglang_rms_norm(module, "q_layernorm")
    assert not qwen3._wrap_sglang_rms_norm(module, "q_layernorm")
    value = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)

    output = module(value)

    torch.testing.assert_close(output, value * 3)
    assert len(calls) == 1
    assert calls[0][4] is False
    assert calls[0][5] == value.dtype


def test_qwen3_pre_mlp_norm_uses_unrounded_fp32_residual_sum(monkeypatch):
    class TENorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                normalization="RMSNorm",
                layernorm_zero_centered_gamma=False,
            )
            self.weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.bias = torch.empty(0)
            self.eps = 1e-6

        def forward(self, value):
            return value

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dropout = 0.0
            self.pre_mlp_layernorm = TENorm()

        def self_attn_bda(self, training, fused):
            del training, fused
            return lambda attention_output_with_bias, residual, dropout: attention_output_with_bias[0] + residual

    calls = []
    monkeypatch.setattr(
        qwen3._SGLangRMSNormWithAnalyticBackward,
        "apply",
        staticmethod(lambda visible, backward, weight, eps, use_fp32_sum, dtype: calls.append((visible, backward, weight, eps, use_fp32_sum, dtype)) or visible.to(dtype)),
    )
    layer = Layer()
    assert qwen3._wrap_sglang_pre_mlp_norm(layer, "pre_mlp_layernorm")
    assert not qwen3._wrap_sglang_pre_mlp_norm(layer, "pre_mlp_layernorm")

    attention_delta = torch.tensor([[1.0, 0.5]], dtype=torch.bfloat16)
    residual = torch.tensor([[0.00390625, 0.00390625]], dtype=torch.bfloat16)
    exact_sum = attention_delta.float() + residual.float()
    rounded_sum = layer.self_attn_bda(True, True)(
        (attention_delta, None),
        residual,
        0.0,
    )

    assert not torch.equal(rounded_sum.float(), exact_sum)
    output = layer.pre_mlp_layernorm(rounded_sum)

    torch.testing.assert_close(output, exact_sum.to(torch.bfloat16), rtol=0, atol=0)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], exact_sum, rtol=0, atol=0)
    torch.testing.assert_close(calls[0][1], rounded_sum, rtol=0, atol=0)
    assert calls[0][4] is True
    assert calls[0][5] == torch.bfloat16


def test_qwen3_core_attention_wrapper_uses_sglang_forward_and_native_backward(
    monkeypatch,
):
    native_calls = []

    class TEDotProductAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                attention_dropout=0.0,
                context_parallel_size=1,
                apply_query_key_layer_scaling=False,
                softmax_type="vanilla",
                window_size=None,
                qk_clip=False,
                log_max_attention_logit=False,
            )

        def forward(
            self,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias=None,
            packed_seq_params=None,
            num_splits=None,
        ):
            native_calls.append(
                (
                    key,
                    value,
                    attention_mask,
                    attn_mask_type,
                    attention_bias,
                    packed_seq_params,
                    num_splits,
                )
            )
            return (query * 2).reshape(query.shape[0], -1)

    kernel_calls = []

    def run_sglang(query, key, value, cu_seqlens_q, cu_seqlens_kv):
        kernel_calls.append((query, key, value, cu_seqlens_q, cu_seqlens_kv))
        return query + 3

    monkeypatch.setattr(qwen3, "_run_sglang_unified_attention", run_sglang)
    module = TEDotProductAttention()
    assert qwen3._wrap_sglang_core_attention(module, "core_attention")
    assert not qwen3._wrap_sglang_core_attention(module, "core_attention")

    query = torch.ones(4, 4, 2, dtype=torch.bfloat16, requires_grad=True)
    key = torch.ones(4, 2, 2, dtype=torch.bfloat16, requires_grad=True)
    value = torch.ones(4, 2, 2, dtype=torch.bfloat16, requires_grad=True)
    cu_seqlens = torch.tensor([0, 1, 4], dtype=torch.int32)
    packed = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
    )
    mask_type = SimpleNamespace(name="causal")

    output = module(
        query,
        key,
        value,
        None,
        mask_type,
        packed_seq_params=packed,
    )

    torch.testing.assert_close(output, (query.detach() + 3).reshape(query.shape[0], -1))
    assert len(kernel_calls) == 1
    assert len(native_calls) == 1
    output.float().sum().backward()
    torch.testing.assert_close(query.grad, torch.full_like(query, 2))
    assert key.grad is None
    assert value.grad is None

    with torch.no_grad():
        no_grad_output = module(
            query,
            key,
            value,
            None,
            mask_type,
            packed_seq_params=packed,
        )
    torch.testing.assert_close(no_grad_output, (query.detach() + 3).reshape(query.shape[0], -1))
    assert len(kernel_calls) == 2
    assert len(native_calls) == 1


def test_qwen3_alignment_accepts_inline_deepep_config():
    assert len(_FIXED_DEEPEP_CONFIG) > 255
    qwen3._validate_fixed_normal_deepep_config(SimpleNamespace(sglang_deepep_config=_FIXED_DEEPEP_CONFIG))


def test_qwen3_alignment_accepts_deepep_config_from_environment(monkeypatch):
    monkeypatch.setenv("SLIME_QWEN3_FIXED_DEEPEP_CONFIG", _FIXED_DEEPEP_CONFIG)

    qwen3._validate_fixed_normal_deepep_config(SimpleNamespace())


def test_qwen3_alignment_maps_qwen_layers_to_metadata_bridge(monkeypatch):
    bridge_calls = []
    monkeypatch.setattr(qwen3, "_ensure_sglang_runtime_context", lambda: None)

    def wrap_triton_moe(mlp, global_layer):
        del global_layer
        mlp.experts = nn.Module()
        mlp.experts._slime_qwen3_triton_moe_wrapped = True
        return True

    def install_bridge(args, model, store_prefix):
        bridge_calls.append((tuple(args.megatron_deepgemm_moe_forward_layers), model, store_prefix))
        model[0].decoder.layers[0].mlp.experts._slime_defer_router_probabilities = True

    monkeypatch.setattr(qwen3, "_wrap_sglang_triton_moe", wrap_triton_moe)
    monkeypatch.setattr(qwen3, "enable_sglang_deepep_moe_alignment", install_bridge)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 1
            self.mlp = nn.Module()
            self.mlp.router = _Router()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([Layer()])

    model = Model()
    args = SimpleNamespace(
        deterministic_mode=True,
        moe_enable_deepep=True,
        sglang_deepep_config=_FIXED_DEEPEP_CONFIG,
        qwen3_alignment_layers=[0],
        megatron_deepgemm_moe_forward_layers=None,
    )
    qwen3.enable_qwen3_moe_alignment(args, [model], "actor_")
    assert bridge_calls == [((0,), [model], "actor_")]
    assert model.decoder.layers[0].mlp.experts._slime_defer_router_probabilities is False
