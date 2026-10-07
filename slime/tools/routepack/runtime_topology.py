"""Call export_topology(args, model, store_prefix) from an existing training hook.

Set ROUTEPACK_PROFILE_DIR to an independent run directory. No collectives,
CUDA synchronization, tensor copies, or scheduler changes are introduced.
"""
import json
import os
from pathlib import Path


def export_topology(args, model, store_prefix='actor'):
    root = os.getenv('ROUTEPACK_PROFILE_DIR')
    if not root:
        return
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    path = Path(root) / f'topology-rank-{dist.get_rank()}.json'
    if path.exists():
        return
    getters = {
        'dense_dp': lambda: mpu.get_data_parallel_group(with_context_parallel=False),
        'ep': mpu.get_expert_model_parallel_group,
        'expert_dp': mpu.get_expert_data_parallel_group,
        'tp': mpu.get_tensor_model_parallel_group,
        'expert_tp': mpu.get_expert_tensor_parallel_group,
        'cp': mpu.get_context_parallel_group,
        'pp': mpu.get_pipeline_model_parallel_group,
    }
    groups = {name: dist.get_process_group_ranks(getter()) for name, getter in getters.items()}
    experts = {}
    for chunk in model if isinstance(model, (list, tuple)) else [model]:
        for name, module in chunk.named_modules():
            mlp = getattr(module, 'mlp', None)
            if getattr(module, 'layer_number', None) is None or not hasattr(mlp, 'local_expert_indices'):
                continue
            key = str(int(module.layer_number) - 1)
            if key in experts:
                raise ValueError(f'Duplicate global layer {key}')
            experts[key] = {'module': name, 'num_experts': int(mlp.config.num_moe_experts),
                            'local_expert_indices': list(mlp.local_expert_indices)}
    record = {'schema_version': 1, 'global_rank': dist.get_rank(), 'world_size': dist.get_world_size(),
              'store_prefix': store_prefix, 'groups': groups, 'experts': experts,
              'vpp_size': mpu.get_virtual_pipeline_model_parallel_world_size() or 1,
              'parallel_state_file': mpu.__file__,
              'data_pad_size_multiplier': args.data_pad_size_multiplier,
              'use_rollout_routing_replay': bool(getattr(args, 'use_rollout_routing_replay', False))}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def aligned_before_log_prob(args, model, store_prefix):
    """Opt-in replacement for the existing Qwen3 before-log-prob hook."""
    from slime.backends.megatron_utils.alignment.qwen3 import enable_qwen3_moe_alignment

    enable_qwen3_moe_alignment(args, model, store_prefix)
    export_topology(args, model, store_prefix)
