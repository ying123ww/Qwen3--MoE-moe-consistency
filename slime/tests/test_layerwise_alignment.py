from types import SimpleNamespace

import pytest
import torch
from torch import nn

from slime.backends.megatron_utils.alignment.layerwise_alignment import enable_megatron_layerwise_dump

NUM_GPUS = 0


class _Layer(nn.Module):
    def __init__(self, layer_number: int):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, value):
        return value + self.layer_number


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(1), _Layer(2)])

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _Decoder()

    def forward(self, *, input_ids, packed_seq_params):
        del packed_seq_params
        return self.decoder(input_ids.float())


def test_megatron_layerwise_dump(monkeypatch, tmp_path):
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR", str(tmp_path))
    args = SimpleNamespace(megatron_deepgemm_forward_layers=[0, 1])
    model = _Model()
    enable_megatron_layerwise_dump(args, [model], store_prefix="actor_")

    model(
        input_ids=torch.tensor([[7, 8]]),
        packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2])),
    )

    (dump_file,) = list(tmp_path.glob("rank*/actor_Pass*.pt"))
    values = torch.load(dump_file, weights_only=False)
    torch.testing.assert_close(values["input_ids"], torch.tensor([[7, 8]]))
    torch.testing.assert_close(values["layers"][0], torch.tensor([[8.0, 9.0]]))
    torch.testing.assert_close(values["layers"][1], torch.tensor([[10.0, 11.0]]))


def test_megatron_layerwise_dump_is_enabled_on_nonzero_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_MODULE_SUFFIXES", "decoder.layers.0")
    monkeypatch.setattr("slime.backends.megatron_utils.alignment.layerwise_alignment._global_rank", lambda: 3)
    args = SimpleNamespace(megatron_deepgemm_forward_layers=[0, 1])
    model = _Model()

    enable_megatron_layerwise_dump(args, [model], store_prefix="actor_")
    model(
        input_ids=torch.tensor([[7, 8]]),
        packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2])),
    )

    (dump_file,) = list(tmp_path.glob("rank00003/actor_Pass*.pt"))
    values = torch.load(dump_file, weights_only=False)
    assert "decoder.layers.0" in values["modules"]


def test_megatron_layerwise_dump_accepts_qwen3_selected_layers(monkeypatch, tmp_path):
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR", str(tmp_path))
    args = SimpleNamespace(qwen3_alignment_layers=[1])
    model = _Model()

    enable_megatron_layerwise_dump(args, [model], store_prefix="actor_")
    model(
        input_ids=torch.tensor([[7, 8]]),
        packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2])),
    )

    (dump_file,) = list(tmp_path.glob("rank*/actor_Pass*.pt"))
    values = torch.load(dump_file, weights_only=False)
    assert set(values["layers"]) == {1}


class _FakeFusedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.layer_norm_weight = nn.Parameter(torch.tensor(3.0))
        self.return_layernorm_output = False

    def forward(self, value):
        linear_output = value * self.weight
        if self.return_layernorm_output:
            return (linear_output, value + 100), None
        return linear_output, None


class _FusedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_number = 1
        self.self_attention = nn.Module()
        self.self_attention.linear_qkv = _FakeFusedLinear()

    def forward(self, value):
        value, _ = self.self_attention.linear_qkv(value)
        return value


def test_megatron_dump_captures_fused_layernorm_and_parameters(monkeypatch, tmp_path):
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv(
        "SLIME_LAYERWISE_ALIGNMENT_MODULE_SUFFIXES",
        "self_attention.linear_qkv",
    )
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_CAPTURE_FUSED_LAYERNORM", "1")
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_CAPTURE_MODULE_PARAMS", "1")
    model = _Model()
    model.decoder.layers = nn.ModuleList([_FusedLayer()])
    enable_megatron_layerwise_dump(
        SimpleNamespace(qwen3_alignment_layers=[0]), [model], store_prefix="actor"
    )

    model(
        input_ids=torch.tensor([[7, 8]]),
        packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2])),
    )

    (dump_file,) = list(tmp_path.glob("rank*/actor_Pass*.pt"))
    values = torch.load(dump_file, weights_only=False)
    prefix = "decoder.layers.0.self_attention.linear_qkv"
    torch.testing.assert_close(
        values["modules"][prefix + ".layernorm_output"],
        torch.tensor([[107.0, 108.0]]),
    )
    torch.testing.assert_close(
        values["modules"][prefix], torch.tensor([[14.0, 16.0]])
    )
    assert set(values["module_params"]) == {
        prefix + ".parameter.weight",
        prefix + ".parameter.layer_norm_weight",
    }
    output, bias = model.decoder.layers[0].self_attention.linear_qkv(
        torch.tensor([[1.0, 2.0]])
    )
    torch.testing.assert_close(output, torch.tensor([[2.0, 4.0]]))
    assert bias is None


class _ThreeInputModule(nn.Module):
    def forward(self, query, key, value):
        return query + key + value


class _ThreeInputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_number = 1
        self.self_attention = nn.Module()
        self.self_attention.core_attention = _ThreeInputModule()

    def forward(self, value):
        return self.self_attention.core_attention(value, value + 1, value + 2)


def test_megatron_dump_captures_all_selected_module_inputs(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv(
        "SLIME_LAYERWISE_ALIGNMENT_MODULE_SUFFIXES",
        "self_attention.core_attention",
    )
    monkeypatch.setenv(
        "SLIME_LAYERWISE_ALIGNMENT_ALL_INPUT_SUFFIXES",
        "self_attention.core_attention",
    )
    model = _Model()
    model.decoder.layers = nn.ModuleList([_ThreeInputLayer()])
    enable_megatron_layerwise_dump(
        SimpleNamespace(qwen3_alignment_layers=[0]),
        [model],
        store_prefix="actor",
    )

    model(
        input_ids=torch.tensor([[7, 8]]),
        packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2])),
    )

    (dump_file,) = list(tmp_path.glob("rank*/actor_Pass*.pt"))
    values = torch.load(dump_file, weights_only=False)
    prefix = "decoder.layers.0.self_attention.core_attention.input."
    torch.testing.assert_close(
        values["module_inputs"][prefix + "0"], torch.tensor([[7.0, 8.0]])
    )
    torch.testing.assert_close(
        values["module_inputs"][prefix + "1"], torch.tensor([[8.0, 9.0]])
    )
    torch.testing.assert_close(
        values["module_inputs"][prefix + "2"], torch.tensor([[9.0, 10.0]])
    )


def test_megatron_dump_layers_can_be_narrower_than_alignment_layers(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_LAYERS", "1")
    model = _Model()
    enable_megatron_layerwise_dump(
        SimpleNamespace(qwen3_alignment_layers=[0, 1]),
        [model],
        store_prefix="actor",
    )

    model(
        input_ids=torch.tensor([[7, 8]]),
        packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2])),
    )

    (dump_file,) = list(tmp_path.glob("rank*/actor_Pass*.pt"))
    values = torch.load(dump_file, weights_only=False)
    assert set(values["layers"]) == {1}
    assert set(values["layer_inputs"]) == {1}


def test_megatron_dump_captures_global_module_suffix(monkeypatch, tmp_path):
    class FinalNorm(nn.Module):
        def forward(self, value):
            return value + 1000

    class DecoderWithFinalNorm(_Decoder):
        def __init__(self):
            super().__init__()
            self.final_layernorm = FinalNorm()

        def forward(self, value):
            return self.final_layernorm(super().forward(value))

    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv(
        "SLIME_LAYERWISE_ALIGNMENT_GLOBAL_MODULE_SUFFIXES",
        "decoder.final_layernorm",
    )
    model = _Model()
    model.decoder = DecoderWithFinalNorm()
    enable_megatron_layerwise_dump(
        SimpleNamespace(qwen3_alignment_layers=[0, 1]),
        [model],
        store_prefix="actor",
    )

    model(
        input_ids=torch.tensor([[7, 8]]),
        packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2])),
    )

    (dump_file,) = list(tmp_path.glob("rank*/actor_Pass*.pt"))
    values = torch.load(dump_file, weights_only=False)
    torch.testing.assert_close(
        values["modules"]["decoder.final_layernorm"],
        torch.tensor([[1010.0, 1011.0]]),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
