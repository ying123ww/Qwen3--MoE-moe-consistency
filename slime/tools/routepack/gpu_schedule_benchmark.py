"""Opt-in fixed-weight schedule gate/timing; never used by normal training.

ROUTEPACK_BENCH_MODE=correctness delegates to the separate dump-heavy B1 gate.
Timing uses real train_one_step and real CPU optimizer at lr=0. Advantages are
controlled sample-bound nonzero inputs, not rewards from a trained RL policy.
"""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
import types


def advantage_value(sample_id):
    """Stable across rank, slot, process and Python hash randomization."""
    value = int.from_bytes(hashlib.sha256(str(sample_id).encode()).digest()[:4], 'little')
    return (1 if value & 1 else -1) * (0.5 + (value % 101) / 100)


def optimizer_tree(root):
    seen, pending = set(), [root]
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        chained = getattr(obj, 'chained_optimizers', ()) or ()
        if chained:
            pending.extend(chained)
            continue
        for name in ('optimizer', 'cpu_optimizer', 'gpu_optimizer'):
            nested = getattr(obj, name, None)
            if nested is not None:
                pending.append(nested)
        for name in ('sub_optimizers', 'cpu_optimizers'):
            pending.extend(getattr(obj, name, ()) or ())


def install(args):
    mode = os.getenv('ROUTEPACK_BENCH_MODE', 'timing')
    if mode == 'correctness':
        from .b1_gate import install as install_gate
        install_gate(args)
        return
    if mode != 'timing' or not args.debug_train_only or args.lr != 0:
        raise ValueError('Timing requires debug-train-only, mode=timing and lr=0')
    if not args.use_routing_replay or not args.use_rollout_routing_replay:
        raise ValueError('Both Routing Replay flags are required')
    from megatron.core import parallel_state as mpu
    if any(getter() != 1 for getter in (mpu.get_tensor_model_parallel_world_size,
            mpu.get_context_parallel_world_size, mpu.get_pipeline_model_parallel_world_size)):
        raise ValueError('This controlled runner supports TP/CP/PP=1 only')
    from ray._private.worker import global_worker
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor
    actor = global_worker.actors.get(global_worker.actor_id)
    if not isinstance(actor, MegatronTrainRayActor):
        raise RuntimeError('Must install inside the live Megatron Ray actor')
    args.custom_megatron_before_log_prob_hook_path = None
    args.custom_megatron_before_train_step_hook_path = None
    actor.train_actor = types.MethodType(train_actor, actor)


def weight_hash(model):
    import torch
    digest = hashlib.sha256()
    for chunk_index, chunk in enumerate(model):
        for name, param in chunk.named_parameters():
            digest.update(f'{chunk_index}:{name}:{tuple(param.shape)}:{param.dtype}'.encode())
            flat = param.detach().reshape(-1)
            for start in range(0, flat.numel(), 1 << 20):
                piece = flat[start:start + (1 << 20)].contiguous().view(torch.uint8).cpu().numpy()
                digest.update(piece.tobytes())
    return digest.hexdigest()


def gradient_check(model):
    import torch
    finite, nonzero, count = True, False, 0
    parameters = []
    for chunk_index, chunk in enumerate(model):
        for name, param in chunk.named_parameters():
            grad = getattr(param, 'main_grad', None)
            if grad is None:
                grad = param.grad
            if grad is None:
                continue
            count += grad.numel()
            good = bool(torch.isfinite(grad).all().item())
            maximum = float(grad.abs().max().item()) if grad.numel() else 0.0
            l2 = float(torch.linalg.vector_norm(grad.reshape(-1), dtype=torch.float64).item())
            finite = finite and good
            nonzero = nonzero or maximum > 0
            parameters.append(dict(name=f'{chunk_index}:{name}', elements=grad.numel(),
                                   finite=good, l2=l2, absmax=maximum))
    return dict(finite=finite, nonzero=nonzero, gradient_elements=count, parameters=parameters)


def forward_with_optional_attribution(inner, phase, synchronize, clock, *args, **kwargs):
    """Formal full-step runs execute the original F/B without extra boundaries."""
    if phase != 'attribution':
        return inner(*args, **kwargs), None
    synchronize()
    started = clock()
    result = inner(*args, **kwargs)
    synchronize()
    return result, clock() - started


def cpu_metrics(value):
    if isinstance(value, dict):
        return {key: cpu_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_metrics(item) for item in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    return value


def train_actor(self, rollout_id, rollout_data, external_data=None):
    import torch
    import torch.distributed as dist
    from megatron.core.utils import get_model_config
    from slime.backends.megatron_utils import model as training
    from slime.backends.megatron_utils.data import get_data_iterator
    from slime.utils.routing_replay import RoutingReplay
    from .runtime_topology import aligned_before_log_prob

    warmups = int(os.getenv('ROUTEPACK_BENCHMARK_WARMUP', '2'))
    repeats = int(os.getenv('ROUTEPACK_BENCHMARK_REPEATS', '5'))
    if warmups < 0 or repeats < 1:
        raise ValueError('Invalid benchmark repeat counts')
    if len(rollout_data['num_microbatches']) != 1:
        raise ValueError('Exactly one optimizer step per fixed rollout is required')
    optimizers = list(optimizer_tree(self.optimizer))
    stateless = [x for x in optimizers if type(x).__name__ == 'StatelessAdam']
    if not stateless:
        raise RuntimeError('Expected real StatelessAdam suboptimizers')
    if any(float(g['lr']) != 0 for o in optimizers for g in getattr(o, 'param_groups', ())):
        raise RuntimeError('Every optimizer group must have lr=0')
    root = Path(os.environ.get('ROUTEPACK_BENCHMARK_DIR') or os.environ['ROUTEPACK_PROFILE_DIR'])
    os.environ['ROUTEPACK_PROFILE_DIR'] = str(root)
    root.mkdir(parents=True, exist_ok=True)
    aligned_before_log_prob(self.args, self.model, '')
    if any(getattr(m, '_routepack_b1', False) for c in self.model for m in c.modules()):
        raise RuntimeError('Dump observation hooks must not be installed in timing mode')
    original_routes = rollout_data['rollout_routed_experts']
    old_probs = rollout_data['rollout_log_probs']
    advantages = [torch.full_like(lp, advantage_value(sid))
                  for sid, lp in zip(rollout_data['sample_indices'], old_probs, strict=True)]
    initial_hash = weight_hash(self.model)
    original_factory = training.get_forward_backward_func
    original_step = training.train_one_step
    original_optimizer_step = self.optimizer.step
    state = {}

    def timed_factory():
        inner = original_factory()
        def wrapped(*args, **kwargs):
            result, elapsed = forward_with_optional_attribution(
                inner, state['phase'], torch.cuda.synchronize, time.perf_counter, *args, **kwargs)
            if elapsed is not None:
                state['forward_backward_s'] = elapsed
            if state['phase'] == 'gradient_validation':
                state['gradient_check'] = gradient_check(self.model)
            return result
        return wrapped

    def timed_step(*args, **kwargs):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_step(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        key = 'full_step_s' if state['phase'] in ('warmup', 'measured') else 'instrumented_step_s'
        state[key] = elapsed
        state['loss_metrics'], state['grad_norm'] = cpu_metrics(result)
        return result

    def optimizer_step(*args, **kwargs):
        state['optimizer_invocations'] = state.get('optimizer_invocations', 0) + 1
        return original_optimizer_step(*args, **kwargs)

    training.get_forward_backward_func = timed_factory
    training.train_one_step = timed_step
    self.optimizer.step = optimizer_step
    reports = []
    try:
        for iteration in range(warmups + repeats + 2):
            phase = ('warmup' if iteration < warmups else
                     'measured' if iteration < warmups + repeats else
                     'attribution' if iteration == warmups + repeats else 'gradient_validation')
            state.clear()
            state.update(phase=phase, iteration=iteration, optimizer_invocations=0)
            prep = time.perf_counter()
            RoutingReplay.clear_all()
            data = dict(rollout_data)
            data.update(rollout_routed_experts=original_routes, advantages=advantages,
                        returns=advantages, log_probs=old_probs, ref_log_probs=old_probs)
            iterator = get_data_iterator(data)
            self.fill_routing_replay(iterator, data['num_microbatches'], data)
            os.environ['ROUTING_REPLAY_STAGE'] = 'replay_backward'
            torch.cuda.synchronize()
            state['preparation_s'] = time.perf_counter() - prep
            config = get_model_config(self.model[0])
            old_no_sync = config.no_sync_func
            # train() installs no_sync anew and asserts it is initially unset.
            config.no_sync_func = None
            try:
                if phase == 'attribution':
                    from .gpu_layer_timing import layer_timing
                    observation = layer_timing(self.model)
                else:
                    observation = nullcontext()
                offload_enabled = os.getenv('ROUTEPACK_CHECKPOINT_CPU_OFFLOAD', '0') == '1'
                if offload_enabled:
                    from .checkpoint_cpu_offload import saved_hidden_cpu_offload, saved_softmax_cpu_offload
                    from slime.utils import ppo_utils
                    activation_storage = saved_hidden_cpu_offload(self.args.hidden_size)
                    protected = {p.untyped_storage().data_ptr() for c in self.model for p in c.parameters()}
                    softmax_storage = saved_softmax_cpu_offload(
                        ppo_utils, self.args.padded_vocab_size, self.args.log_probs_chunk_size,
                        protected_storages=protected)
                else:
                    activation_storage = nullcontext(None)
                    softmax_storage = nullcontext(None)
                if phase == 'attribution' and os.getenv('ROUTEPACK_COMPONENT_PROFILE', '0') == '1':
                    from .gpu_component_profile import component_profile
                    components = component_profile(self.model, self.optimizer, root / 'components', dist.get_rank())
                else:
                    components = nullcontext(None)
                with observation as trace, activation_storage as offload_stats, softmax_storage as softmax_stats, components as component_trace:
                    training.train(rollout_id, self.model, self.optimizer, self.opt_param_scheduler,
                                   iterator, data['num_microbatches'], data['global_batch_sizes'])
                if component_trace is not None:
                    state['component_profile'] = component_trace.results()
                if offload_stats is not None:
                    state['checkpoint_cpu_offload'] = dict(offload_stats)
                    state['saved_softmax_cpu_offload'] = dict(softmax_stats)
                if phase == 'attribution':
                    state['layer_attribution'] = trace.results()
            finally:
                config.no_sync_func = old_no_sync
            RoutingReplay.assert_all_consumed()
            if state['optimizer_invocations'] != 1:
                raise RuntimeError('Expected exactly one actual optimizer.step invocation')
            # All collection occurs after local elapsed time was captured.
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, dict(state, rank=dist.get_rank()))
            reports.append(gathered)
    finally:
        training.get_forward_backward_func = original_factory
        training.train_one_step = original_step
        self.optimizer.step = original_optimizer_step
        RoutingReplay.clear_all()
    final_hash = weight_hash(self.model)
    local = dict(rank=dist.get_rank(), initial_weight_sha256=initial_hash,
        final_weight_sha256=final_hash, weights_unchanged=initial_hash == final_hash,
        gradient_check=state.get('gradient_check'), stateless_optimizer_count=len(stateless))
    checks = [None] * dist.get_world_size()
    dist.all_gather_object(checks, local)
    passed = all(x['weights_unchanged'] and x['gradient_check']['finite'] and
                 x['gradient_check']['nonzero'] for x in checks)
    result = dict(passed=passed, checks=checks, iterations=reports, warmups=warmups,
        measured_repeats=repeats, attribution_repeats=1, gradient_validation_repeats=1, scope='Fixed-weight controlled-advantage training step; actual CPU optimizer at lr0',
        timing='Formal full_step_s has only outer barrier/synchronize boundaries, includes optimizer/grad zero/metrics; separately instrumented attribution and gradient_validation phases excluded',
        schedule_manifest=os.environ.get('ROUTEPACK_SCHEDULE_FILE'),
        limitations=['Layer events are from a separate instrumented pass and include stream waits; recompute forward is not backward kernel time', 'Per-parameter gradient norms are diagnostics, not a proof of cross-schedule gradient equality',
                     'Does not include rollout, schedule construction, R3 preparation or weight synchronization'],
        checkpoint_cpu_offload=os.getenv('ROUTEPACK_CHECKPOINT_CPU_OFFLOAD', '0') == '1',
        advantage_rule='sha256(sample ID), signed nonzero constant per response; not a real reward estimate')
    if dist.get_rank() == 0:
        (root / 'timing.json').write_text(json.dumps(result, indent=2) + '\n')
    if not passed:
        raise RuntimeError('Weight-freeze or gradient validity gate failed')
