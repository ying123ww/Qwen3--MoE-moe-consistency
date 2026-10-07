"""
This file provides a function `register_forward_hook_for_model` that registers a forward hook on every operator of the model.
After registration, during model inference, all tensors generated throughout the forward pass will be recorded.

Usage:
Specify the output directory for dumping tensors using the argument `--debug-tensor-dump-output-folder`.
A separate directory will be created for each GPU rank, named in the format `f"TP{tp_rank}_PP{pp_rank}_Rank{rank}_pid{pid}"`.
Each complete forward pass of the model generates a `.pt` file named `f"Pass{pass_num}.pt"`, which can be loaded using `torch.load`.
The file contains a series of key-value pairs, where the keys correspond to operator names in the model
(similar to those in model.safetensors.index.json), and the values are the outputs produced by the respective operators.
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger(__name__)


class TensorDumper:
    def __init__(
        self,
        dump_dir: str,
        dump_layers: Optional[List[int]],
        tp_size: int,
        tp_rank: int,
        pp_rank: int,
    ):
        self._dump_layers = dump_layers
        self._layer_outputs_only = os.getenv("SGLANG_TENSOR_DUMP_LAYER_OUTPUTS_ONLY", "0") == "1"
        self._dump_router_outputs = os.getenv("SGLANG_TENSOR_DUMP_ROUTER_OUTPUTS", "0") == "1"
        self._dump_moe_stages = os.getenv("SGLANG_TENSOR_DUMP_MOE_STAGES", "0") == "1"
        self._module_suffixes = tuple(suffix.strip() for suffix in os.getenv("SGLANG_TENSOR_DUMP_MODULE_SUFFIXES", "").split(",") if suffix.strip())
        self._dump_module_params = os.getenv("SGLANG_TENSOR_DUMP_MODULE_PARAMS", "0") == "1"
        self._module_all_input_suffixes = tuple(suffix.strip() for suffix in os.getenv("SGLANG_TENSOR_DUMP_ALL_INPUT_SUFFIXES", "").split(",") if suffix.strip())
        self._module_all_output_suffixes = tuple(suffix.strip() for suffix in os.getenv("SGLANG_TENSOR_DUMP_ALL_OUTPUT_SUFFIXES", "").split(",") if suffix.strip())
        self._forward_pass_id = 0
        self._pid = os.getpid()
        self._current_tensors = {}
        self._base_dir = Path(dump_dir)
        rank = tp_size * pp_rank + tp_rank
        self._process_dir = self._base_dir / f"TP{tp_rank}_PP{pp_rank}_Rank{rank}_pid{self._pid}"
        self._process_dir.mkdir(parents=True, exist_ok=True)

    def get_dump_dir(self):
        return str(self._process_dir)

    def add_tensor(self, name, tensor_item):
        if isinstance(tensor_item, (tuple, list)):
            tensors = [t.cpu() for t in tensor_item if t is not None]
            if len(tensors) == 1:
                self._current_tensors[name] = tensors[0]
            else:
                self._current_tensors[name] = tensors
        elif isinstance(tensor_item, torch.Tensor):
            self._current_tensors[name] = tensor_item.cpu()
        elif isinstance(tensor_item, LogitsProcessorOutput):
            self._current_tensors[name] = tensor_item.next_token_logits.cpu()
        elif isinstance(tensor_item, ForwardBatch):
            self._current_tensors[name + ".forward_batch_info.input_ids"] = tensor_item.input_ids.cpu()
            self._current_tensors[name + ".forward_batch_info.seq_lens"] = tensor_item.seq_lens.cpu()
            self._current_tensors[name + ".forward_batch_info.positions"] = tensor_item.positions.cpu()
            # Preserve request segmentation so an external train/rollout
            # alignment checker can map every rollout token back to the
            # matching packed Megatron sequence.
            if tensor_item.rids is not None:
                self._current_tensors[name + ".forward_batch_info.rids"] = list(tensor_item.rids)
            if tensor_item.extend_seq_lens is not None:
                self._current_tensors[name + ".forward_batch_info.extend_seq_lens"] = tensor_item.extend_seq_lens.cpu()
        elif isinstance(tensor_item, PPProxyTensors):
            for tensor_name in tensor_item.tensors.keys():
                self._current_tensors[name + ".pp_proxy_tensors." + tensor_name] = tensor_item.tensors[tensor_name].cpu()
        else:
            logger.warning(f"Unsupported type: {type(tensor_item)}: {tensor_item}")

    def dump_current_tensors(self):
        if len(self._current_tensors) == 0:
            return
        tensor_file_for_pass = self._process_dir / f"Pass{self._forward_pass_id:05d}.pt"
        logger.info(f"Dump {self._forward_pass_id:05d}th pass to {tensor_file_for_pass}")
        torch.save(self._current_tensors, str(tensor_file_for_pass))
        self._current_tensors = {}
        self._forward_pass_id += 1

    def _add_hook_recursive(self, model, prefix, top_level_module_name, layers_module_name):
        model_top_level_module_matched = False
        layers_prefix = top_level_module_name + "." + layers_module_name
        for name, module in model._modules.items():
            top_level_model = False
            if len(prefix) == 0:
                cur_name = name
                if cur_name == top_level_module_name:
                    model_top_level_module_matched = True
                    top_level_model = True
            else:
                cur_name = prefix + "." + name
            if self._dump_layers is not None and name.isdigit() and prefix == layers_prefix:
                # If we only need n layers, skip the reset layers.
                # Most models' layout is like model.layers.0.
                cur_layer = int(name)
                if cur_layer not in self._dump_layers:
                    continue
            selected_layer = name.isdigit() and prefix == layers_prefix
            if module is not None:
                if selected_layer and self._layer_outputs_only:
                    if self._dump_moe_stages:
                        self._register_moe_stage_hooks(cur_name, module)
                    module.register_forward_hook(self._dump_hook(cur_name, False))
                    if self._module_suffixes:
                        module.register_forward_pre_hook(self._dump_module_pre_hook(cur_name), with_kwargs=True)
                        for sub_name, submodule in module.named_modules():
                            if not sub_name:
                                continue
                            full_name = cur_name + "." + sub_name
                            if any(full_name.endswith(suffix) for suffix in self._module_suffixes):
                                if self._dump_module_params:
                                    for parameter_name, parameter in submodule.named_parameters(recurse=False):
                                        self.add_tensor(
                                            full_name + ".parameter." + parameter_name,
                                            parameter.detach(),
                                        )
                                submodule.register_forward_pre_hook(
                                    self._dump_module_pre_hook(full_name),
                                    with_kwargs=True,
                                )
                                submodule.register_forward_hook(self._dump_module_hook(full_name))
                    if self._dump_router_outputs:
                        mlp = getattr(module, "mlp", None)
                        if not callable(getattr(mlp, "_router_logits", None)) or not callable(getattr(mlp, "_select_experts", None)):
                            raise TypeError(f"{cur_name}.mlp does not expose router logits and top-k selection")
                        mlp.register_forward_pre_hook(self._dump_router_pre_hook(cur_name + ".mlp"))
                    continue
                _, sub_count = self._add_hook_recursive(module, cur_name, top_level_module_name, layers_module_name)
                if sub_count == 0 or top_level_model:
                    # Avoid duplicated output hooks, e.g. self_attn may contain:
                    # self_attn.qkv_proj, self_attn.attn & self_attn.o_proj.
                    # Therefore, we do not need to add output hooks for self_attn,
                    # since the output of self_attn should be the same to self_attn.o_proj.
                    module.register_forward_hook(self._dump_hook(cur_name, top_level_model))
        return model_top_level_module_matched, len(model._modules.items())

    def _dump_module_pre_hook(self, tensor_name):
        def inner_dump_module_pre_hook(module, input, kwargs):
            del module
            if any(tensor_name.endswith(suffix) for suffix in self._module_all_input_suffixes):
                tensor_index = 0
                for item in input:
                    if isinstance(item, torch.Tensor):
                        self.add_tensor(f"{tensor_name}.input.{tensor_index}", item)
                        tensor_index += 1
            value = kwargs.get("hidden_states")
            if not isinstance(value, torch.Tensor):
                tensors = [item for item in input if isinstance(item, torch.Tensor)]
                value = next((item for item in tensors if item.ndim >= 2), None)
                if value is None and tensors:
                    value = tensors[0]
            if value is not None:
                self.add_tensor(tensor_name + ".input", value)

        return inner_dump_module_pre_hook

    def _dump_module_hook(self, tensor_name):
        def inner_dump_module_hook(module, input, output):
            del module, input
            if any(tensor_name.endswith(suffix) for suffix in self._module_all_output_suffixes):
                tensor_index = 0
                for item in output if isinstance(output, (tuple, list)) else (output,):
                    if isinstance(item, torch.Tensor):
                        self.add_tensor(f"{tensor_name}.output.{tensor_index}", item)
                        tensor_index += 1
            value = output
            if isinstance(value, (tuple, list)):
                value = next(
                    (item for item in value if isinstance(item, torch.Tensor)),
                    None,
                )
            if value is not None:
                self.add_tensor(tensor_name, value)

        return inner_dump_module_hook

    def _dump_router_pre_hook(self, tensor_name):
        def inner_dump_router_pre_hook(module, input):
            hidden_states = input[0]
            router_weight = getattr(getattr(module, "gate", None), "weight", None)
            if not isinstance(router_weight, torch.Tensor):
                raise TypeError(f"{tensor_name}.gate does not expose a tensor weight")
            self.add_tensor(tensor_name + ".router_input", hidden_states)
            self.add_tensor(tensor_name + ".router_weight", router_weight)
            router_logits = module._router_logits(hidden_states)
            topk_output = module._select_experts(hidden_states, router_logits)
            self.add_tensor(tensor_name + ".router_logits", router_logits)
            self.add_tensor(tensor_name + ".topk_ids", topk_output.topk_ids)
            self.add_tensor(tensor_name + ".topk_weights", topk_output.topk_weights)

        return inner_dump_router_pre_hook

    def _add_moe_stage_value(self, tensor_name, value):
        for field in ("hidden_states", "topk_ids", "topk_weights"):
            tensor = getattr(value, field, None)
            if isinstance(tensor, torch.Tensor):
                self.add_tensor(f"{tensor_name}.{field}", tensor)
        counts = getattr(value, "num_recv_tokens_per_expert", None)
        if counts is not None:
            self.add_tensor(
                f"{tensor_name}.num_recv_tokens_per_expert",
                torch.as_tensor(counts, dtype=torch.int64),
            )

    def _register_moe_stage_hooks(self, layer_name, layer):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        dispatcher = getattr(experts, "dispatcher", None)
        if experts is None or dispatcher is None:
            raise TypeError(f"{layer_name}.mlp does not expose experts.dispatcher")
        if getattr(experts, "_slime_tensor_dump_moe_stages", False):
            return

        original_dispatch = dispatcher.dispatch
        original_run_moe_core = experts.run_moe_core
        original_combine = dispatcher.combine
        runner = getattr(getattr(experts, "quant_method", None), "runner", None)
        runner_core = getattr(runner, "runner_core", None)
        if runner_core is None or not callable(getattr(runner_core, "run", None)):
            raise TypeError(f"{layer_name}.mlp does not expose a traceable MoE runner core")
        original_runner_run = runner_core.run

        def after_down(intermediate, route_outputs, topk_weights, topk_ids):
            del intermediate
            self.add_tensor(layer_name + ".mlp.expert_routes.route_outputs", route_outputs)
            self.add_tensor(layer_name + ".mlp.expert_routes.route_topk_ids", topk_ids)
            self.add_tensor(layer_name + ".mlp.expert_routes.route_topk_weights", topk_weights)

        route_hooks = SimpleNamespace(after_gate_up=None, after_down=after_down)

        def runner_run(*args, **kwargs):
            if kwargs.get("hooks") is not None:
                raise RuntimeError("MoE route tracing cannot be combined with existing runner hooks")

            from dataclasses import replace

            unweighted_hidden_states = args[0].hidden_states.clone()
            kwargs["hooks"] = route_hooks
            output = original_runner_run(*args, **kwargs)

            def after_unweighted_down(
                intermediate,
                route_outputs,
                topk_weights,
                topk_ids,
            ):
                del intermediate, topk_weights, topk_ids
                self.add_tensor(
                    layer_name + ".mlp.expert_routes.unweighted_route_outputs",
                    route_outputs,
                )

            unweighted_hooks = SimpleNamespace(
                after_gate_up=None,
                after_down=after_unweighted_down,
            )
            unweighted_config = replace(runner_core.config, inplace=False)
            unweighted_input = replace(
                args[0],
                hidden_states=unweighted_hidden_states,
                topk_weights=torch.ones_like(args[0].topk_weights),
            )
            type(runner_core)(unweighted_config).run(
                unweighted_input,
                args[1],
                args[2],
                hooks=unweighted_hooks,
            )
            return output

        def dispatch(*args, **kwargs):
            output = original_dispatch(*args, **kwargs)
            self._add_moe_stage_value(layer_name + ".mlp.dispatch", output)
            return output

        def run_moe_core(*args, **kwargs):
            output = original_run_moe_core(*args, **kwargs)
            self._add_moe_stage_value(layer_name + ".mlp.expert_routes", output)
            return output

        def combine(*args, **kwargs):
            output = original_combine(*args, **kwargs)
            self.add_tensor(layer_name + ".mlp.combine", output)
            return output

        dispatcher.dispatch = dispatch
        experts.run_moe_core = run_moe_core
        dispatcher.combine = combine
        runner_core.run = runner_run
        experts._slime_tensor_dump_moe_stages = True

    def _dump_hook(self, tensor_name, do_dump):
        def inner_dump_hook(module, input, output):
            if do_dump:
                # This is the top-level model, so we will record the input for it.
                for item in input:
                    if isinstance(item, ForwardBatch):
                        self.add_tensor(tensor_name, item)
                self.dump_current_tensors()
            if output is not None:
                self.add_tensor(tensor_name, output)

        return inner_dump_hook


def register_forward_hook_for_model(
    model,
    dump_dir: str,
    dump_layers: Optional[List[int]],
    tp_size: int,
    tp_rank: int,
    pp_rank: int,
):
    tensor_dumper = TensorDumper(dump_dir, dump_layers, tp_size, tp_rank, pp_rank)
    # Most models have the layerout like:
    # XxxxForCausalLM
    #     (model): XxxxModel
    #         (layers): ModuleList
    # If the model is not constructed with this layout,
    # environment variable can be used to specify the module names.
    top_level_module_name = os.getenv("TENSOR_DUMP_TOP_LEVEL_MODULE_NAME", "model")
    layers_module_name = os.getenv("TENSOR_DUMP_LAYERS_MODULE_NAME", "layers")
    model_top_level_module_matched, _ = tensor_dumper._add_hook_recursive(model, "", top_level_module_name, layers_module_name)
    assert model_top_level_module_matched, f"model should have a module named {top_level_module_name}"
    return tensor_dumper
