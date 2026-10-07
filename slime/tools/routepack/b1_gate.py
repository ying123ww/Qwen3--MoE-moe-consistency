"""Opt-in, forward-only B1 integration gate on the original rollout schedule.

Install with --custom-megatron-init-path routepack.b1_gate.install.
Observation deliberately synchronizes/copies IDs; this is not a timing probe.
"""
import functools
import json
import os
from pathlib import Path
import types

import numpy as np

_STATE = None
_CONTEXT = None


def compare_routes(actual, expected, identities, *, real_only):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f'Dispatch shape {actual.shape} != replay layout {expected.shape}')
    selected = np.array([kind == 'real' or not real_only for _, _, kind in identities])
    bad = np.any(actual != expected, axis=1) & selected
    positions = np.flatnonzero(bad)
    return {'rows_checked': int(selected.sum()), 'mismatched_rows': len(positions),
            'first_mismatches': [{'sample_index': identities[i][0], 'token_position': identities[i][1],
                                  'kind': identities[i][2], 'expected': expected[i].tolist(),
                                  'actual': actual[i].tolist()} for i in positions[:8]]}


def layout_identities(indices, lengths, padded_length):
    rows = []
    for index, length in zip(indices, lengths, strict=True):
        rows.extend((int(index), pos, 'real' if pos < length - 1 else 'terminal') for pos in range(length))
    if padded_length < len(rows):
        raise ValueError('Layout shorter than actual token stream')
    rows.extend((-1, i, 'padding') for i in range(padded_length - len(rows)))
    return rows


def _observe(kind, layer, value):
    state = _STATE
    slot = state['calls'].get((kind, layer), 0)
    state['calls'][(kind, layer)] = slot + 1
    expected = state['expected'][slot][:, layer, :]
    actual = value.detach().cpu().numpy().copy()
    comparison = compare_routes(actual, expected, state['identities'][slot], real_only=state['stage'] == 'fallthrough')
    filename = f"{state['stage']}-{kind}-slot{slot}-layer{layer}.npy"
    np.save(state['root'] / filename, actual, allow_pickle=False)
    state['observations'].append({'stage': state['stage'], 'kind': kind, 'layer': layer, 'slot': slot,
                                  'path': filename, **comparison})


def before_log_prob(args, model, store_prefix):
    from .runtime_topology import aligned_before_log_prob
    from deep_ep import Buffer

    aligned_before_log_prob(args, model, store_prefix)
    if not getattr(Buffer.dispatch, '_routepack_b1', False):
        original = Buffer.dispatch

        @functools.wraps(original)
        def dispatch(buffer, *positional, **kwargs):
            # Observe the primary payload call inside the real manager dispatch.
            # The alignment bridge's subsequent metadata-only dispatch is separate.
            context = _CONTEXT
            if context is not None:
                context['calls'] += 1
                if context['calls'] == 1:
                    ids = kwargs.get('topk_idx')
                    if ids is None:
                        raise RuntimeError('Primary DeepEP dispatch lacks topk_idx; gate cannot establish IDs')
                    _observe('dispatch', context['layer'], ids)
            result = original(buffer, *positional, **kwargs)
            return result

        dispatch._routepack_b1 = True
        Buffer.dispatch = dispatch
    for chunk in model:
        for module in chunk.modules():
            mlp = getattr(module, 'mlp', None)
            if mlp is None or not hasattr(mlp, 'local_expert_indices') or getattr(mlp, '_routepack_b1', False):
                continue
            layer = int(module.layer_number) - 1
            router = mlp.router

            def router_hook(router, _inputs, _output, layer=layer):
                ordered = getattr(router, '_slime_ordered_topk_indices', None)
                if ordered is None:
                    raise RuntimeError('Router ordered top-k was not captured')
                _observe('router', layer, ordered)

            router.register_forward_hook(router_hook)
            manager = mlp.token_dispatcher._comm_manager
            original_dispatch = manager.dispatch

            def manager_dispatch(manager, *positional, _original=original_dispatch, _layer=layer, **kwargs):
                global _CONTEXT
                if _CONTEXT is not None:
                    raise RuntimeError('Nested dispatch unsupported by B1 scope')
                context = {'layer': _layer, 'calls': 0}
                _CONTEXT = context
                try:
                    result = _original(*positional, **kwargs)
                    if context['calls'] < 1:
                        raise RuntimeError('Actual DeepEP Buffer.dispatch was not observed')
                    return result
                finally:
                    _CONTEXT = None

            manager.dispatch = types.MethodType(manager_dispatch, manager)
            mlp._routepack_b1 = True


def install(args):
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor
    if not args.debug_train_only or not args.use_rollout_routing_replay or not args.use_routing_replay:
        raise RuntimeError('B1 requires debug-train-only and both R3 flags')
    if os.environ.get('ENABLE_ROUTING_REPLAY') != '1' or not os.environ.get('ROUTEPACK_PROFILE_DIR'):
        raise RuntimeError('Missing B1 run directory or R3 environment')
    from megatron.core import parallel_state as mpu
    for getter in (mpu.get_tensor_model_parallel_world_size, mpu.get_context_parallel_world_size,
                   mpu.get_pipeline_model_parallel_world_size, mpu.get_expert_tensor_parallel_world_size):
        if getter() != 1:
            raise RuntimeError('B1 V1 requires TP/CP/PP/expert TP = 1')
    if (mpu.get_virtual_pipeline_model_parallel_world_size() or 1) != 1:
        raise RuntimeError('B1 V1 requires VPP = 1')
    args.custom_megatron_before_log_prob_hook_path = 'routepack.b1_gate.before_log_prob'
    from ray._private.worker import global_worker
    _bind_actor(global_worker, MegatronTrainRayActor)


def _bind_actor(worker, actor_type):
    # Ray copies inherited methods into its tracing subclass. Patching only the
    # imported base class would leave that old train_actor wrapper active.
    actor = worker.actors.get(worker.actor_id)
    if actor is None or not isinstance(actor, actor_type):
        raise RuntimeError('B1 init must run inside the current Megatron Ray actor')
    actor.train_actor = types.MethodType(train_actor, actor)
    if actor.train_actor.__func__ is not train_actor:
        raise RuntimeError('B1 forward-only method was not installed on the live actor')


def train_actor(self, rollout_id, rollout_data, external_data=None):
    import torch
    import torch.distributed as dist
    from slime.backends.megatron_utils.data import get_data_iterator
    from slime.backends.megatron_utils.cp_utils import prepare_routed_experts_for_routing_replay
    from slime.utils.routing_replay import RoutingReplay
    global _STATE

    root = Path(os.environ['ROUTEPACK_PROFILE_DIR']) / f'rank-{dist.get_rank()}'
    root.mkdir(parents=True, exist_ok=False)
    iterator = get_data_iterator(rollout_data)
    expected, identities = [], []
    for local_indices in rollout_data['micro_batch_indices']:
        tokens = [rollout_data['tokens'][i] for i in local_indices]
        routes = [rollout_data['rollout_routed_experts'][i] for i in local_indices]
        padded = prepare_routed_experts_for_routing_replay(
            routes, tokens, num_experts=self.args.num_experts,
            data_pad_size_multiplier=self.args.data_pad_size_multiplier,
            sequence_parallel=self.args.sequence_parallel, allgather_cp=self.args.allgather_cp)
        expected.append(padded.cpu().numpy())
        identities.append(layout_identities([rollout_data['sample_indices'][i] for i in local_indices],
                                             [len(t) for t in tokens], padded.shape[0]))
    schedule = {key: rollout_data[key] for key in ('partition', 'sample_indices', 'micro_batch_indices',
                                                  'num_microbatches', 'global_batch_sizes', 'total_lengths')}
    schedule.update({'rank': dist.get_rank(), 'rollout_id': rollout_id, 'identities': identities})
    (root / 'schedule.json').write_text(json.dumps(schedule, indent=2) + '\n')
    _STATE = {'root': root, 'expected': expected, 'identities': identities, 'observations': []}
    self.fill_routing_replay(iterator, rollout_data['num_microbatches'], rollout_data)
    outputs = {}
    for stage in ('fallthrough', 'replay_forward'):
        _STATE.update(stage=stage, calls={})
        os.environ['ROUTING_REPLAY_STAGE'] = stage
        outputs[stage] = self.compute_log_prob(iterator, rollout_data['num_microbatches'])['log_probs']
        expected_calls = {(kind, layer): len(expected) for kind in ('router', 'dispatch') for layer in range(48)}
        if _STATE['calls'] != expected_calls:
            raise RuntimeError(f'Incomplete actual router/dispatch coverage: {_STATE["calls"]}')
    consumed = [replay.forward_index == len(expected) for replay in RoutingReplay.all_routing_replays]
    if len(consumed) != 48 or not all(consumed):
        raise RuntimeError('R3 did not consume every recorded microbatch on all 48 layers')
    comparisons = []
    for i, sample_index in enumerate(rollout_data['sample_indices']):
        baseline = outputs['fallthrough'][i].detach().cpu().float().numpy()
        replay = outputs['replay_forward'][i].detach().cpu().float().numpy()
        rollout = rollout_data['rollout_log_probs'][i].detach().cpu().float().numpy()
        for name, array in [('aligned', baseline), ('replay', replay), ('rollout', rollout)]:
            np.save(root / f'{name}-sample{sample_index}.npy', array, allow_pickle=False)
        if baseline.shape != rollout.shape or replay.shape != rollout.shape:
            raise RuntimeError('Response logprob coverage differs')
        comparisons.append({'sample_index': sample_index, 'response_tokens': len(rollout),
                            'aligned_vs_rollout_exact': bool(np.array_equal(baseline.view(np.uint32), rollout.view(np.uint32))),
                            'replay_vs_rollout_exact': bool(np.array_equal(replay.view(np.uint32), rollout.view(np.uint32))),
                            'replay_vs_aligned_exact': bool(np.array_equal(replay.view(np.uint32), baseline.view(np.uint32))),
                            'aligned_max_abs_diff': float(np.max(np.abs(baseline - rollout))),
                            'replay_max_abs_diff': float(np.max(np.abs(replay - rollout)))})
    schedule_manifest = os.environ.get('ROUTEPACK_SCHEDULE_FILE')
    report = {'rank': dist.get_rank(), 'optimizer_steps': 0,
              'scope': 'validated explicit schedule; two forward-only passes' if schedule_manifest else 'original schedule; two forward-only passes',
              'schedule_manifest': schedule_manifest,
              'observations': _STATE['observations'], 'logprobs': comparisons,
              'r3_all_48_layers_consumed': True}
    report['passed'] = (not any(row['mismatched_rows'] for row in report['observations'])
                        and all(row['aligned_vs_rollout_exact'] and row['replay_vs_rollout_exact']
                                and row['replay_vs_aligned_exact'] for row in comparisons))
    (root / 'gate.json').write_text(json.dumps(report, indent=2) + '\n')
    RoutingReplay.clear_all()
    passed = torch.tensor(int(report['passed']), device=torch.cuda.current_device())
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if not passed.item():
        raise RuntimeError('RoutePack B1 gate failed; inspect all rank gate.json evidence')


def configure_launch(options, env, *, run, rollout_path):
    """Adapt already expanded v89 options/env; caller owns resource guards/Ray."""
    run = Path(run)
    options = {key: list(value) for key, value in options.items() if not key.startswith('--sglang-')}
    for key in ('--save-debug-train-data', '--save-debug-rollout-data', '--save', '--save-interval',
                '--custom-megatron-before-train-step-hook-path'):
        options.pop(key, None)
    options.update({'--debug-train-only': [], '--load-debug-rollout-data': [str(rollout_path)],
                    '--custom-megatron-init-path': ['routepack.b1_gate.install'],
                    '--custom-megatron-before-log-prob-hook-path': ['routepack.b1_gate.before_log_prob'],
                    '--use-routing-replay': [], '--use-rollout-routing-replay': [],
                    '--lr': ['0'], '--num-rollout': ['1'], '--start-rollout-id': ['0'],
                    '--rollout-temperature': ['1.0']})
    tools_root = str(Path(__file__).resolve().parents[1])
    pythonpath = tools_root + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env = dict(env, ENABLE_ROUTING_REPLAY='1', ROUTEPACK_PROFILE_DIR=str(run / 'b1'), PYTHONPATH=pythonpath)
    return options, env
