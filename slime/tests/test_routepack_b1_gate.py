import numpy as np
import pytest

from tools.routepack.b1_gate import compare_routes, configure_launch, layout_identities


def test_identity_boundary_and_terminal_exclusion():
    identity = layout_identities([7, 9], [3, 2], 8)
    assert identity == [(7, 0, 'real'), (7, 1, 'real'), (7, 2, 'terminal'),
                        (9, 0, 'real'), (9, 1, 'terminal'), (-1, 0, 'padding'),
                        (-1, 1, 'padding'), (-1, 2, 'padding')]
    expected = np.tile([2, 5], (8, 1))
    actual = expected.copy()
    actual[[2, 4, 6]] = [4, 6]
    assert compare_routes(actual, expected, identity, real_only=True)['mismatched_rows'] == 0
    assert compare_routes(actual, expected, identity, real_only=False)['mismatched_rows'] == 3
    actual[3] = [5, 2]
    result = compare_routes(actual, expected, identity, real_only=True)
    assert result['mismatched_rows'] == 1  # ordered equality, not set equality
    assert result['first_mismatches'][0]['sample_index'] == 9
    assert result['first_mismatches'][0]['token_position'] == 0


def test_coverage_rejected():
    with pytest.raises(ValueError):
        layout_identities([0], [5], 4)
    with pytest.raises(ValueError):
        compare_routes(np.zeros((1, 8)), np.zeros((2, 8)), [(0, 0, 'real')], real_only=True)


def test_launch_is_debug_forward_only_and_preserves_training_topology():
    options, env = configure_launch({'--sglang-ep-size': ['4'], '--expert-model-parallel-size': ['4'],
                                     '--custom-megatron-before-train-step-hook-path': ['old'], '--save': ['old']},
                                    {'ENABLE_ROUTING_REPLAY': '0'}, run='/tmp/b1-test', rollout_path='/tmp/input.pt')
    assert '--sglang-ep-size' not in options and '--save' not in options
    assert options['--expert-model-parallel-size'] == ['4']
    assert options['--num-rollout'] == ['1']
    assert options['--rollout-temperature'] == ['1.0']
    assert env['ENABLE_ROUTING_REPLAY'] == '1'
    assert options['--custom-megatron-init-path'] == ['routepack.b1_gate.install']


def test_observer_reads_primary_buffer_argument_and_preserves_dispatch(monkeypatch):
    import sys
    import types
    import torch
    from tools.routepack import b1_gate, runtime_topology

    calls, observed = [], []

    class Buffer:
        def dispatch(self, payload, *, topk_idx):
            calls.append(topk_idx.clone())
            return payload + 1

    class Router(torch.nn.Module):
        def forward(self, value):
            self._slime_ordered_topk_indices = torch.tensor([[3, 5]])
            return value

    class Manager:
        def dispatch(self, value):
            # Intentionally differs from router/cache; the observer must see this.
            result = Buffer().dispatch(value, topk_idx=torch.tensor([[8, 9]]))
            Buffer().dispatch(value, topk_idx=torch.tensor([[99]]))  # metadata payload
            return result

    layer = torch.nn.Module()
    layer.layer_number = 1
    layer.mlp = torch.nn.Module()
    layer.mlp.local_expert_indices = [0]
    layer.mlp.router = Router()
    layer.mlp.token_dispatcher = types.SimpleNamespace(_comm_manager=Manager())
    monkeypatch.setitem(sys.modules, 'deep_ep', types.SimpleNamespace(Buffer=Buffer))
    monkeypatch.setattr(runtime_topology, 'aligned_before_log_prob', lambda *args: None)
    monkeypatch.setattr(b1_gate, '_observe', lambda kind, layer, value: observed.append((kind, layer, value.tolist())))
    b1_gate.before_log_prob(None, [layer], '')
    layer.mlp.router(torch.tensor([1]))
    result = layer.mlp.token_dispatcher._comm_manager.dispatch(torch.tensor([7]))
    assert result.tolist() == [8]
    assert len(calls) == 2
    assert observed == [('router', 0, [[3, 5]]), ('dispatch', 0, [[8, 9]])]
    assert b1_gate._CONTEXT is None


def test_prepare_preserves_capture_and_rejects_changed_identity(tmp_path):
    import hashlib
    import json
    import torch
    from tools.routepack.prepare_b1 import prepare

    tokens = np.array([11, 12, 13], dtype=np.int64)
    routes = np.broadcast_to(np.arange(8, dtype=np.int32), (2, 48, 8)).copy()
    logprobs = np.array([-0.0], dtype=np.float32)
    for filename, array in [('tokens.npy', tokens), ('routes.npy', routes), ('lp.npy', logprobs)]:
        np.save(tmp_path / filename, array, allow_pickle=False)
    row = {'sample_id': 'capture:0', 'token_path': 'tokens.npy', 'route_path': 'routes.npy',
           'response_logprob_path': 'lp.npy', 'response_length': 1,
           'effective_logprob_temperature': 1.0,
           'token_hash': hashlib.sha256(tokens.astype('<i8').tobytes()).hexdigest(),
           'trace_hash': hashlib.sha256(routes.astype('<i8').tobytes()).hexdigest()}
    (tmp_path / 'samples_captured.jsonl').write_text(json.dumps(row) + '\n')
    output = tmp_path / 'rollout.pt'
    assert prepare(tmp_path, output)['samples'] == 1
    sample = torch.load(output, weights_only=True)['samples'][0]
    assert sample['tokens'] == [11, 12, 13]
    assert torch.equal(sample['rollout_routed_experts'], torch.from_numpy(routes))
    assert np.asarray(sample['rollout_log_probs'], dtype=np.float32).view(np.uint32)[0] == 0x80000000
    tokens[0] += 1
    np.save(tmp_path / 'tokens.npy', tokens, allow_pickle=False)
    with pytest.raises(ValueError, match='Token hash'):
        prepare(tmp_path, tmp_path / 'changed.pt')


def test_bind_live_actor_overrides_ray_tracing_subclass_copy():
    from types import SimpleNamespace
    from tools.routepack.b1_gate import _bind_actor, train_actor

    class Actor:
        def train_actor(self):
            raise AssertionError('Original training must never run')

    class RayTracingActor(Actor):
        # Ray tracing stores a wrapper on the subclass before init executes.
        def train_actor(self):
            return Actor.train_actor(self)

    live = RayTracingActor()
    worker = SimpleNamespace(actor_id='current', actors={'current': live})
    _bind_actor(worker, Actor)
    assert live.train_actor.__self__ is live
    assert live.train_actor.__func__ is train_actor
    with pytest.raises(RuntimeError, match='current Megatron Ray actor'):
        _bind_actor(SimpleNamespace(actor_id='missing', actors={}), Actor)
