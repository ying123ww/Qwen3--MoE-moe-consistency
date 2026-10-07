import unittest

import torch
from torch import nn

from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed.parallel_state import (
    get_default_distributed_backend,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import LinearBase
from sglang.srt.models.qwen2 import Qwen2MLP
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import add_prefix, get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=9,
    stage="base-b",
    runner_config="1-gpu-small",
    disabled="Test uses pytest-style function without TestCase class - see #17145",
)
register_amd_ci(
    est_time=15,
    suite="stage-b-test-1-gpu-small-amd",
    disabled="Test uses pytest-style function without TestCase class - see #17145",
)

TEST_HIDDEN_SIZE = 32


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = TEST_HIDDEN_SIZE
        self.rms_norm_eps = 1e-5
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.hidden_size,
            hidden_act="silu",
            quant_config=None,
            prefix=add_prefix("mlp", ""),
        )
        self.layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class MockCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleModel()

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model(hidden_states)


def init_weights(module):
    if isinstance(module, LinearBase):
        torch.nn.init.uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, RMSNorm):
        torch.nn.init.ones_(module.weight)


def test_model_forward_dump(tmp_path):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    device = get_device()
    backend = get_default_distributed_backend(device)
    init_distributed_environment(
        backend=backend,
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:2646",
    )
    initialize_model_parallel()
    model = MockCausalLM()
    model.apply(init_weights)
    model = model.to(device=device, dtype=torch.bfloat16)
    dumper = register_forward_hook_for_model(model, tmp_path / "sglang_dump", [0], 0, 0, 0)

    dir_path = dumper.get_dump_dir()
    inp = torch.randn(4, TEST_HIDDEN_SIZE, dtype=torch.bfloat16) * 0.01
    result = model(inp.to(device))
    data = torch.load(f"{dir_path}/Pass00000.pt")
    assert "model.layernorm" in data
    assert "model.mlp.down_proj" in data
    assert torch.allclose(data["model.mlp.down_proj"], result.cpu(), rtol=1e-5, atol=1e-5)


def test_selected_layer_outputs_only(monkeypatch, tmp_path):
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)

        def forward(self, positions, value):
            del positions
            return self.proj(value), value

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])

        def forward(self, value):
            positions = torch.arange(value.shape[0])
            for layer in self.layers:
                value, _ = layer(positions, value)
            return value

    class CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Model()

        def forward(self, value):
            return self.model(value)

    monkeypatch.setenv("SGLANG_TENSOR_DUMP_LAYER_OUTPUTS_ONLY", "1")
    monkeypatch.setenv("SGLANG_TENSOR_DUMP_MODULE_SUFFIXES", "proj")
    monkeypatch.setenv("SGLANG_TENSOR_DUMP_MODULE_PARAMS", "1")
    monkeypatch.setenv("SGLANG_TENSOR_DUMP_ALL_INPUT_SUFFIXES", "proj")
    model = CausalLM()
    dumper = register_forward_hook_for_model(model, tmp_path / "sglang_dump", [0], 1, 0, 0)

    model(torch.ones(1, 2))

    data = torch.load(f"{dumper.get_dump_dir()}/Pass00000.pt")
    assert set(data) == {
        "model.layers.0",
        "model.layers.0.input",
        "model.layers.0.proj",
        "model.layers.0.proj.input",
        "model.layers.0.proj.input.0",
        "model.layers.0.proj.parameter.weight",
    }
    torch.testing.assert_close(
        data["model.layers.0.proj.parameter.weight"],
        model.model.layers[0].proj.weight,
    )
    torch.testing.assert_close(data["model.layers.0.input"], torch.ones(1, 2))
    torch.testing.assert_close(data["model.layers.0.proj.input"], torch.ones(1, 2))
    torch.testing.assert_close(data["model.layers.0.proj.input.0"], torch.ones(1, 2))


def test_selected_module_can_dump_all_tensor_outputs(monkeypatch, tmp_path):
    class Pair(nn.Module):
        def forward(self, value):
            return value + 1, value + 2

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.pair = Pair()

        def forward(self, value):
            return self.pair(value)[0]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

        def forward(self, value):
            return self.layers[0](value)

    class CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Model()

        def forward(self, value):
            return self.model(value)

    monkeypatch.setenv("SGLANG_TENSOR_DUMP_LAYER_OUTPUTS_ONLY", "1")
    monkeypatch.setenv("SGLANG_TENSOR_DUMP_MODULE_SUFFIXES", "pair")
    monkeypatch.setenv("SGLANG_TENSOR_DUMP_ALL_OUTPUT_SUFFIXES", "pair")
    model = CausalLM()
    dumper = register_forward_hook_for_model(model, tmp_path / "sglang_dump", [0], 1, 0, 0)

    model(torch.ones(1, 2))

    data = torch.load(f"{dumper.get_dump_dir()}/Pass00000.pt")
    torch.testing.assert_close(data["model.layers.0.pair.output.0"], torch.full((1, 2), 2.0))
    torch.testing.assert_close(data["model.layers.0.pair.output.1"], torch.full((1, 2), 3.0))


def test_selected_layer_can_dump_router_outputs(monkeypatch, tmp_path):
    class SparseMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(2, 2, bias=False)

        def _router_logits(self, hidden_states):
            return hidden_states.float()

        def _select_experts(self, hidden_states, router_logits):
            del hidden_states
            topk_weights, topk_ids = torch.topk(router_logits, 1, dim=-1)
            return type(
                "TopKOutput",
                (),
                {"topk_weights": topk_weights, "topk_ids": topk_ids},
            )()

        def forward(self, hidden_states, forward_batch=None):
            del forward_batch
            return hidden_states

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = SparseMLP()

        def forward(self, value):
            return self.mlp(value), value

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

        def forward(self, value):
            return self.layers[0](value)[0]

    class CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Model()

        def forward(self, value):
            return self.model(value)

    monkeypatch.setenv("SGLANG_TENSOR_DUMP_LAYER_OUTPUTS_ONLY", "1")
    monkeypatch.setenv("SGLANG_TENSOR_DUMP_ROUTER_OUTPUTS", "1")
    model = CausalLM()
    dumper = register_forward_hook_for_model(model, tmp_path / "sglang_dump", [0], 1, 0, 0)

    model(torch.tensor([[1.0, 2.0]]))

    data = torch.load(f"{dumper.get_dump_dir()}/Pass00000.pt")
    assert set(data) == {
        "model.layers.0",
        "model.layers.0.mlp.router_input",
        "model.layers.0.mlp.router_weight",
        "model.layers.0.mlp.router_logits",
        "model.layers.0.mlp.topk_ids",
        "model.layers.0.mlp.topk_weights",
    }
    torch.testing.assert_close(data["model.layers.0.mlp.router_input"], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(data["model.layers.0.mlp.router_weight"], model.model.layers[0].mlp.gate.weight)
    torch.testing.assert_close(data["model.layers.0.mlp.router_logits"], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(data["model.layers.0.mlp.topk_ids"], torch.tensor([[1]]))
    torch.testing.assert_close(data["model.layers.0.mlp.topk_weights"], torch.tensor([[2.0]]))


if __name__ == "__main__":
    unittest.main()
