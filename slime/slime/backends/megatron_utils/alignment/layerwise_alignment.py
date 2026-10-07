"""Opt-in Megatron layer-output dumps for train/rollout alignment gates."""

from __future__ import annotations

import logging
import os
import re
from functools import partial
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"^(?:.*\.)?decoder\.layers\.(\d+)$")


def _global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _selected_layers(args) -> set[int]:
    """Select Qwen3 gates first, retaining the existing GLM flag fallback."""

    dump_layers = os.getenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_LAYERS")
    if dump_layers:
        try:
            return {
                int(layer)
                for layer in re.split(r"[,\s]+", dump_layers.strip())
                if layer
            }
        except ValueError as error:
            raise ValueError(
                "SLIME_LAYERWISE_ALIGNMENT_DUMP_LAYERS must contain integer layer IDs"
            ) from error
    layers = getattr(args, "qwen3_alignment_layers", None)
    if layers is None:
        layers = getattr(args, "megatron_deepgemm_forward_layers", None)
    return {int(layer) for layer in (layers or ())}


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class _MegatronLayerwiseDumper:
    def __init__(self, dump_dir: str, selected_layers: set[int], store_prefix: str):
        self.dump_dir = Path(dump_dir) / f"rank{_global_rank():05d}"
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.selected_layers = selected_layers
        self.store_prefix = store_prefix.rstrip("_") or "actor"
        self.module_suffixes = tuple(
            suffix.strip()
            for suffix in os.getenv("SLIME_LAYERWISE_ALIGNMENT_MODULE_SUFFIXES", "").split(",")
            if suffix.strip()
        )
        self.global_module_suffixes = tuple(
            suffix.strip()
            for suffix in os.getenv(
                "SLIME_LAYERWISE_ALIGNMENT_GLOBAL_MODULE_SUFFIXES", ""
            ).split(",")
            if suffix.strip()
        )
        self.module_all_input_suffixes = tuple(
            suffix.strip()
            for suffix in os.getenv(
                "SLIME_LAYERWISE_ALIGNMENT_ALL_INPUT_SUFFIXES", ""
            ).split(",")
            if suffix.strip()
        )
        self.capture_fused_layernorm = (
            os.getenv("SLIME_LAYERWISE_ALIGNMENT_CAPTURE_FUSED_LAYERNORM", "0") == "1"
        )
        self.capture_module_params = (
            os.getenv("SLIME_LAYERWISE_ALIGNMENT_CAPTURE_MODULE_PARAMS", "0") == "1"
        )
        self.module_parameters: dict[str, torch.Tensor] = {}
        self.pass_id = 0
        self.current: dict[str, Any] = {}

    def pre_forward(self, module, args, kwargs):
        del module, args
        self.current = {
            "store_prefix": self.store_prefix,
            "layers": {},
            "layer_inputs": {},
            "modules": {},
            "module_inputs": {},
        }
        if self.capture_module_params:
            self.current["module_params"] = {
                name: parameter.detach().cpu()
                for name, parameter in self.module_parameters.items()
            }
        input_ids = kwargs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            self.current["input_ids"] = input_ids.detach().cpu()
        packed_seq_params = kwargs.get("packed_seq_params")
        cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
        if isinstance(cu_seqlens, torch.Tensor):
            self.current["cu_seqlens"] = cu_seqlens.detach().cpu()

    def record_layer_input(self, layer_id: int, module, args, kwargs):
        del module
        if "layers" not in self.current:
            return
        tensor = kwargs.get("hidden_states")
        if not isinstance(tensor, torch.Tensor):
            tensor = _first_tensor(args)
        if tensor is not None:
            self.current["layer_inputs"][layer_id] = tensor.detach().cpu()

    def record_layer(self, layer_id: int, module, args, output):
        del module, args
        if "layers" not in self.current:
            return
        tensor = _first_tensor(output)
        if tensor is not None:
            self.current["layers"][layer_id] = tensor.detach().cpu()

    def record_module_input(self, module_name: str, module, args, kwargs):
        del module
        if "modules" not in self.current:
            return
        if any(
            module_name.endswith(suffix)
            for suffix in self.module_all_input_suffixes
        ):
            tensor_index = 0
            for item in args:
                if isinstance(item, torch.Tensor):
                    self.current["module_inputs"][
                        f"{module_name}.input.{tensor_index}"
                    ] = item.detach().cpu()
                    tensor_index += 1
        tensor = kwargs.get("hidden_states")
        if not isinstance(tensor, torch.Tensor):
            tensor = _first_tensor(args)
        if tensor is not None:
            self.current["module_inputs"][module_name] = tensor.detach().cpu()

    def record_module(self, module_name: str, module, args, output):
        del module, args
        if "modules" not in self.current:
            return
        tensor = _first_tensor(output)
        if tensor is not None:
            self.current["modules"][module_name] = tensor.detach().cpu()

    def record_fused_layernorm(self, module_name: str, module, args, output):
        del module, args
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not isinstance(output[0], tuple)
            or len(output[0]) != 2
        ):
            raise TypeError(
                f"{module_name} did not return "
                "((linear_output, layernorm_output), bias)"
            )
        (linear_output, layernorm_output), bias = output
        if not isinstance(layernorm_output, torch.Tensor):
            raise TypeError(f"{module_name} did not return a tensor layernorm output")
        if "modules" in self.current:
            self.current["modules"][module_name + ".layernorm_output"] = (
                layernorm_output.detach().cpu()
            )
        return linear_output, bias

    def post_forward(self, module, args, output):
        del module, args, output
        if "input_ids" not in self.current:
            raise RuntimeError("Megatron layerwise dump did not observe model input_ids")
        observed_layers = set(self.current.get("layers", {}))
        missing_inputs = self.selected_layers - set(self.current.get("layer_inputs", {}))
        if missing_inputs:
            raise RuntimeError(
                "Megatron layerwise dump missed selected layer inputs: "
                f"{sorted(missing_inputs)}"
            )
        missing_layers = self.selected_layers - observed_layers
        if missing_layers:
            raise RuntimeError("Megatron layerwise dump missed selected layers: " f"{sorted(missing_layers)}")
        output_path = self.dump_dir / f"{self.store_prefix}_Pass{self.pass_id:05d}.pt"
        torch.save(self.current, output_path)
        logger.info("Dumped Megatron layer outputs to %s", output_path)
        self.pass_id += 1
        self.current = {}

    def register(self, model_chunk) -> int:
        model_chunk.register_forward_pre_hook(self.pre_forward, with_kwargs=True)
        model_chunk.register_forward_hook(self.post_forward)
        registered_layers = 0
        selected_prefixes = []
        for module_name, module in model_chunk.named_modules():
            match = _LAYER_RE.match(module_name)
            if match is None:
                continue
            layer_number = getattr(module, "layer_number", None)
            layer_id = int(layer_number) - 1 if layer_number is not None else int(match.group(1))
            if layer_id not in self.selected_layers:
                continue
            module.register_forward_pre_hook(
                partial(self.record_layer_input, layer_id), with_kwargs=True
            )
            module.register_forward_hook(partial(self.record_layer, layer_id))
            selected_prefixes.append(module_name)
            registered_layers += 1
        for module_name, module in model_chunk.named_modules():
            in_selected_layer = any(
                module_name == prefix or module_name.startswith(prefix + ".")
                for prefix in selected_prefixes
            )
            selected_module = in_selected_layer and any(
                module_name.endswith(suffix) for suffix in self.module_suffixes
            )
            global_module = any(
                module_name.endswith(suffix)
                for suffix in self.global_module_suffixes
            )
            if selected_module or global_module:
                if self.capture_module_params:
                    for parameter_name, parameter in module.named_parameters(recurse=False):
                        self.module_parameters[
                            module_name + ".parameter." + parameter_name
                        ] = parameter
                if (
                    self.capture_fused_layernorm
                    and module_name.endswith("self_attention.linear_qkv")
                ):
                    if not hasattr(module, "return_layernorm_output"):
                        raise TypeError(
                            f"{module_name} does not expose return_layernorm_output"
                        )
                    module.return_layernorm_output = True
                    module.register_forward_hook(
                        partial(self.record_fused_layernorm, module_name)
                    )
                module.register_forward_pre_hook(
                    partial(self.record_module_input, module_name), with_kwargs=True
                )
                module.register_forward_hook(partial(self.record_module, module_name))
        return registered_layers


def enable_megatron_layerwise_dump(args, model, store_prefix: str) -> None:
    """Register one dump hook per selected layer on every model rank."""

    dump_dir = os.getenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR")
    if not dump_dir:
        return

    selected_layers = _selected_layers(args)
    if not selected_layers:
        raise RuntimeError(
            "SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR requires "
            "--qwen3-alignment-layers or --megatron-deepgemm-forward-layers"
        )

    registered_layers = 0
    for model_chunk in model:
        if getattr(model_chunk, "_slime_layerwise_dump_registered", False):
            continue
        dumper = _MegatronLayerwiseDumper(dump_dir, selected_layers, store_prefix)
        registered_layers += dumper.register(model_chunk)
        model_chunk._slime_layerwise_dump_registered = True
        model_chunk._slime_layerwise_dumper = dumper

    if registered_layers == 0 and not any(
        getattr(model_chunk, "_slime_layerwise_dump_registered", False) for model_chunk in model
    ):
        raise RuntimeError("Could not find selected Megatron decoder layers to dump")
    if registered_layers:
        logger.info(
            "Enabled Megatron layerwise alignment dump for layers %s at %s",
            sorted(selected_layers),
            dump_dir,
        )
