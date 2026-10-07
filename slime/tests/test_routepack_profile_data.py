"""CPU-only RoutePack B1/B2 contracts; fixtures are not runtime evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('routepack_profile_data', Path(__file__).parents[1] / 'tools/routepack/profile_data.py')
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)
NUM_GPUS = 0


def topology_rows(world=4):
    return [{'global_rank': r, 'world_size': world, 'vpp_size': 1,
             'groups': {'dense_dp': list(range(world)), 'ep': [r // 2 * 2, r // 2 * 2 + 1],
                        'expert_dp': list(range(r % 2, world, 2)),
                        **{name: [r] for name in ('tp', 'expert_tp', 'cp', 'pp')}},
             'experts': {'0': {'num_experts': 4, 'local_expert_indices': [r % 2 * 2, r % 2 * 2 + 1]}}}
            for r in range(world)]


def inputs():
    topology = profile.merge_topology(topology_rows())
    samples = [{'sample_id': str(i), 'step': 0, 'rollout_id': i, 'reward_group_id': i // 2,
                'model_version': 'fixture', 'expert_layout_version': 'fixture', 'token_count': 3,
                'response_length': 1, 'valid_loss_token_count': 1, 'layer_ids': [0]}
               for i in range(4)]
    traces = {str(i): np.array([[[i % 2 * 2, i % 2 * 2 + 1]]] * 2) for i in range(4)}
    schedule = {'cells': [{'step': 0, 'group_id': i // 2, 'slot': 0, 'source_rank': i, 'sample_ids': [str(i)]}
                          for i in range(4)]}
    return samples, traces, schedule, topology


def test_runtime_owner_export_requires_all_ranks_and_unique_experts():
    rows = topology_rows()
    assert profile.merge_topology(rows)['ep_groups'][1]['members'] == [2, 3]
    with pytest.raises(ValueError, match='Missing or duplicate'):
        profile.merge_topology(rows[:-1])
    rows[1]['experts']['0']['local_expert_indices'] = [0, 1]
    with pytest.raises(ValueError, match='duplicate expert'):
        profile.merge_topology(rows)


def test_unsupported_topology_is_rejected():
    rows = topology_rows()
    rows[0]['vpp_size'] = 2
    with pytest.raises(ValueError, match='VPP'):
        profile.merge_topology(rows)


def test_h_assignments_differ_from_unique_destination_tokens():
    h, u, experts = profile.route_histograms(np.array([[[0, 1]], [[1, 2]]]), [[0, 0, 1, 1]])
    np.testing.assert_array_equal(h, [[3, 1]])
    np.testing.assert_array_equal(u, [[2, 1]])
    np.testing.assert_array_equal(experts, [[1, 2, 1, 0]])


@pytest.mark.parametrize('routes', [np.array([[[0, 4]]]), np.array([[[0, 0]]]), np.array([[[0., 1.]]])])
def test_bad_routes_fail_closed(routes):
    with pytest.raises(ValueError):
        profile.route_histograms(routes, [[0, 0, 1, 1]])


def test_terminal_and_bin_padding_are_separate_from_real_trace():
    samples, traces, schedule, topology = inputs()
    metadata, records, arrays = profile.summarize(samples, traces, schedule, topology, 4)
    assert metadata[0]['traced_token_count'] == 2
    assert records[0]['training_tokens'] == 6
    assert records[0]['layout_padding_tokens'] == 2
    np.testing.assert_array_equal(arrays['sample_H'][0], [[4, 0]])
    np.testing.assert_array_equal(arrays['terminal_H'][0], [[2, 0]])
    np.testing.assert_array_equal(arrays['layout_padding_H'][0], [[4, 0]])
    np.testing.assert_array_equal(arrays['collective_H'][0], [[12, 4]])


def test_source_swap_within_same_collective_does_not_change_receiver_load():
    samples, traces, schedule, topology = inputs()
    baseline = profile.summarize(samples, traces, schedule, topology, 1)[2]
    schedule['cells'][0]['sample_ids'], schedule['cells'][1]['sample_ids'] = ['1'], ['0']
    swapped = profile.summarize(samples, traces, schedule, topology, 1)[2]
    np.testing.assert_array_equal(baseline['collective_H'], swapped['collective_H'])


def test_cross_replica_swap_changes_collective_inputs_even_for_singletons():
    samples, traces, schedule, topology = inputs()
    baseline = profile.summarize(samples, traces, schedule, topology, 1)[2]
    schedule['cells'][0]['sample_ids'], schedule['cells'][3]['sample_ids'] = ['3'], ['0']
    swapped = profile.summarize(samples, traces, schedule, topology, 1)[2]
    assert not np.array_equal(baseline['collective_H'], swapped['collective_H'])
    np.testing.assert_array_equal(baseline['collective_H'].sum(0), swapped['collective_H'].sum(0))


@pytest.mark.parametrize('failure', ['missing_trace', 'truncated_trace', 'step', 'duplicate', 'layer', 'topology'])
def test_data_integrity_gates(failure):
    samples, traces, schedule, topology = inputs()
    if failure == 'missing_trace':
        traces.pop('0')
    elif failure == 'truncated_trace':
        traces['0'] = traces['0'][:1]
    elif failure == 'step':
        samples[0]['step'] = 1
    elif failure == 'duplicate':
        schedule['cells'][0]['sample_ids'] = ['1']
    elif failure == 'layer':
        samples[0]['layer_ids'] = [1]
    elif failure == 'topology':
        topology['ep_groups'][0]['members'] = [1, 0]
    with pytest.raises(ValueError):
        profile.summarize(samples, traces, schedule, topology, 4)


def test_native_schedule_adapter_uses_actual_dense_dp_rank_order():
    topology = profile.merge_topology(topology_rows())
    topology['dense_dp_rank_to_global'] = [2, 0, 3, 1]
    schedule = profile.native_schedule_cells([[0], [1], [2], [3]], [[[0]]] * 4, [1], ['a', 'b', 'c', 'd'], topology)
    assert schedule['cells'][0] == {'step': 0, 'group_id': 1, 'slot': 0, 'source_rank': 2, 'sample_ids': ['a']}


@pytest.mark.parametrize('allgather_cp', [False, True])
def test_padding_matches_production_replay_on_cpu(monkeypatch, allgather_cp):
    import _cp_dist_helpers  # installs the established CPU-only Megatron stub
    import torch
    from slime.backends.megatron_utils import cp_utils

    monkeypatch.setattr(cp_utils.mpu, 'get_tensor_model_parallel_world_size', lambda: 1, raising=False)
    monkeypatch.setattr(cp_utils.mpu, 'get_context_parallel_world_size', lambda: 1)
    routes = [torch.tensor([[[0, 1]], [[2, 3]]]), torch.tensor([[[1, 2]]])]
    token_ids = [torch.zeros(3), torch.zeros(2)]
    actual = cp_utils.prepare_routed_experts_for_routing_replay(
        routes, token_ids, num_experts=4, data_pad_size_multiplier=8,
        sequence_parallel=False, allgather_cp=allgather_cp)
    expected = np.concatenate([routes[0].numpy(), profile.padding_routes(1, 1, 2, 4),
                               routes[1].numpy(), profile.padding_routes(1, 1, 2, 4),
                               profile.padding_routes(3, 1, 2, 4)])
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert not torch.cuda.is_initialized()


def test_engine_capture_checks_coverage_and_preserves_token_identity(tmp_path):
    import base64
    spec = importlib.util.spec_from_file_location('capture_routes', Path(__file__).parents[1] / 'tools/routepack/capture_routes.py')
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    routes = np.array([[[0, 1]], [[2, 3]]], dtype=np.int32)
    output = {'meta_info': {'routed_experts': base64.b64encode(routes.tobytes()).decode(),
                           'completion_tokens': 1, 'prompt_tokens': 2, 'output_token_logprobs': [[-0.5, 9, None]]}}
    kwargs = {'model_version': 'fixture', 'model_config': {'num_hidden_layers': 1, 'num_experts_per_tok': 2, 'num_experts': 4}}
    result = capture.save_engine_routes([output], [[7, 8]], [{'index': 5}], tmp_path / 'capture', **kwargs)
    assert result['actual_dispatch_route_gate'] == 'not_run'
    np.testing.assert_array_equal(np.load(tmp_path / 'capture/tokens-0.npy'), [7, 8, 9])
    np.testing.assert_array_equal(np.load(tmp_path / 'capture/routes-0.npy'), routes)
    output['meta_info']['routed_experts'] = base64.b64encode(routes[:1].tobytes()).decode()
    with pytest.raises(ValueError, match='coverage mismatch'):
        capture.save_engine_routes([output], [[7, 8]], [{'index': 5}], tmp_path / 'bad_capture', **kwargs)
    assert not (tmp_path / 'bad_capture').exists()
