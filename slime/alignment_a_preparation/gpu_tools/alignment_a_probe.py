"""Opt-in complete-step measurement for fixed-data A-stage experiments."""
import functools
import inspect
import json
import os
from pathlib import Path
import time


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    import torch
    import torch.distributed as dist
    from slime.backends.megatron_utils import model as training

    variant = os.environ['ALIGN_A_VARIANT']
    if variant == 'aligned':
        from slime.backends.megatron_utils.alignment.qwen3 import enable_qwen3_moe_alignment_before_train_step
        enable_qwen3_moe_alignment_before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)
    elif variant != 'native' or getattr(args, 'qwen3_alignment_layers', None) is not None:
        raise RuntimeError('Invalid native-controlled alignment state')
    if getattr(training.train_one_step, '_alignment_a_measured', False):
        return

    rank = dist.get_rank()
    run = Path(os.environ['ALIGN_A_RUN_DIR'])
    counts = {}
    for chunk in model:
        for module in chunk.modules():
            for key, value in vars(module).items():
                if key.startswith('_slime_') and isinstance(value, bool) and value:
                    counts[key] = counts.get(key, 0) + 1
    if variant == 'aligned' and counts.get('_slime_qwen3_triton_moe_wrapped') != 48:
        raise RuntimeError(f'Expected 48 aligned MoE layers, got {counts}')
    if variant == 'native' and any('qwen3' in key or 'sglang' in key for key in counts):
        raise RuntimeError(f'Unexpected alignment wrapper in native model: {counts}')
    import sys
    from megatron.core import parallel_state as mpu
    from deep_ep import Buffer
    audit = {'rank': rank, 'variant': variant, 'torch': torch.__version__,
             'device': torch.cuda.get_device_name(), 'wrappers': counts,
             'dense_dp_size': mpu.get_data_parallel_world_size(),
             'ep_size': mpu.get_expert_model_parallel_world_size(),
             'ep_rank': mpu.get_expert_model_parallel_rank(),
             'imports': {name: getattr(sys.modules.get(name), '__file__', None)
                         for name in ['slime', 'megatron', 'megatron.core', 'sglang', 'deep_ep']},
             'deepep_num_sms': Buffer.num_sms,
             'deepep_dispatch_selector_source': inspect.getsource(Buffer.get_dispatch_config),
             'deepep_combine_selector_source': inspect.getsource(Buffer.get_combine_config),
             'deepep_config_note': 'Selectors and num_sms captured before measurement; opaque C++ Config has no public field getters.',
             'qwen3_alignment_layers': getattr(args, 'qwen3_alignment_layers', None),
             'lr': args.lr, 'timing_scope': 'train_one_step including forward/backward, optimizer, gradient zeroing and metric reduction; excluding outer data preparation/offload/dumps',
             'first_unmeasured_rollout': rollout_id}
    (run / f'audit-rank{rank}.json').write_text(json.dumps(audit, indent=2) + '\n')
    original = training.train_one_step

    @functools.wraps(original)
    def measured(*positional, **kwargs):
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_allocated = torch.cuda.memory_allocated()
        started = time.perf_counter()
        result = original(*positional, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        grad_norm = result[1]
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        row = {'rank': rank, 'variant': variant, 'rollout_id': positional[1],
               'step_id': positional[2], 'num_microbatches': positional[7],
               'global_batch_size': positional[8], 'elapsed_s': elapsed,
               'start_allocated_bytes': start_allocated,
               'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved,
               'grad_norm': grad_norm}
        with (run / f'measurements-rank{rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        return result

    measured._alignment_a_measured = True
    training.train_one_step = measured
