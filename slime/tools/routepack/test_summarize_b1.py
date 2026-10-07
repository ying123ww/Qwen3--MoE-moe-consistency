"""Small CPU fixtures exercise B1 artifact joins; never runtime evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from profile_data import padding_routes
from summarize_b1 import build


def write_json(path, data):
    path.write_text(json.dumps(data))


def make_artifacts(tmp_path, *, world=2):
    b1, capture = tmp_path / 'b1', tmp_path / 'capture'
    b1.mkdir()
    capture.mkdir()
    samples, capture_rows = [], []
    for rank in range(world):
        ep_rank = rank if world == 2 else rank // 2
        ep_members = [0, 1] if world == 2 else [rank % 2, rank % 2 + 2]
        dense_members = [0, 1] if world == 2 else [2, 0, 3, 1]
        expert_dp_members = [rank] if world == 2 else [ep_rank * 2, ep_rank * 2 + 1]
        topology = {'global_rank': rank, 'world_size': world, 'vpp_size': 1,
                    'data_pad_size_multiplier': 4, 'use_rollout_routing_replay': True,
                    'groups': {'dense_dp': dense_members, 'ep': ep_members, 'expert_dp': expert_dp_members,
                               **{key: [rank] for key in ('tp', 'expert_tp', 'cp', 'pp')}},
                    'experts': {str(layer): {'num_experts': 4, 'local_expert_indices': [ep_rank * 2, ep_rank * 2 + 1]}
                                for layer in range(48)}}
        write_json(b1 / f'topology-rank-{rank}.json', topology)
        tokens = np.array([1, 2, 3], dtype=np.int64)
        routes = np.tile(np.array([ep_rank * 2, ep_rank * 2 + 1], dtype=np.int32), (2, 48, 1))
        np.save(capture / f'routes-{rank}.npy', routes)
        np.save(capture / f'tokens-{rank}.npy', tokens)
        np.save(capture / f'logprobs-{rank}.npy', np.array([-0.5], dtype=np.float32))
        row = {'sample_id': f'capture:{rank}', 'route_path': f'routes-{rank}.npy', 'token_path': f'tokens-{rank}.npy',
               'response_length': 1, 'token_count': 3, 'layer_ids': list(range(48)), 'model_version': 'fixture',
               'effective_logprob_temperature': 1.0, 'response_logprob_path': f'logprobs-{rank}.npy',
               'trace_hash': hashlib.sha256(routes.astype('<i8').tobytes()).hexdigest(),
               'token_hash': hashlib.sha256(tokens.astype('<i8').tobytes()).hexdigest()}
        capture_rows.append(row)
        samples.append({'index': rank, 'group_index': 0, 'rollout_id': rank, 'tokens': tokens.tolist(),
                        'rollout_routed_experts': routes, 'response_length': 1, 'loss_mask': [1],
                        'rollout_log_probs': [-0.5],
                        'metadata': {'routepack_sample_id': row['sample_id'], 'capture': str(capture)}})
        directory = b1 / f'rank-{rank}'
        directory.mkdir()
        for kind in ('aligned', 'replay', 'rollout'):
            np.save(directory / f'{kind}-sample{rank}.npy', np.array([-0.5], dtype=np.float32))
        identities = [[rank, 0, 'real'], [rank, 1, 'real'], [rank, 2, 'terminal'], [-1, 0, 'padding']]
        schedule = {'rank': rank, 'partition': [rank], 'sample_indices': [rank], 'total_lengths': [3],
                    'num_microbatches': [1], 'global_batch_sizes': [world], 'micro_batch_indices': [[0]],
                    'identities': [identities], 'rollout_id': 0}
        write_json(directory / 'schedule.json', schedule)
        expected = np.concatenate([routes, padding_routes(1, 48, 2, 4), padding_routes(1, 48, 2, 4)])
        observations = []
        for stage in ('fallthrough', 'replay_forward'):
            for kind in ('router', 'dispatch'):
                for layer in range(48):
                    path = f'{stage}-{kind}-slot0-layer{layer}.npy'
                    np.save(directory / path, expected[:, layer])
                    observations.append({'stage': stage, 'kind': kind, 'slot': 0, 'layer': layer,
                                         'path': path, 'mismatched_rows': 0,
                                         'rows_checked': 2 if stage == 'fallthrough' else 4})
        gate = {'rank': rank, 'passed': True, 'r3_all_48_layers_consumed': True, 'optimizer_steps': 0,
                'observations': observations, 'logprobs': [{'sample_index': rank, 'response_tokens': 1,
                'aligned_vs_rollout_exact': True, 'replay_vs_rollout_exact': True, 'replay_vs_aligned_exact': True}]}
        write_json(directory / 'gate.json', gate)
    (capture / 'samples_captured.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in capture_rows))
    return b1, capture, samples


@pytest.fixture
def artifacts(tmp_path):
    return make_artifacts(tmp_path)


def test_actual_dispatch_and_prediction_agree_without_fabricating_time(artifacts):
    topology, schedule, samples, collectives, arrays, validation, hashes = build(*artifacts, run_id='fixture-run')
    assert topology['ep_groups'][0]['members'] == [0, 1]
    assert len(schedule['cells']) == 2 and len(samples) == 2 and len(collectives) == 1
    np.testing.assert_array_equal(arrays['collective_H'][0, 0], [12, 4])
    np.testing.assert_array_equal(arrays['collective_U'][0, 0], [6, 2])
    np.testing.assert_array_equal(arrays['actual_dispatch_H'], arrays['collective_H'])
    assert validation['actual_route_files_checked'] == 384
    assert validation['go_decision'] == 'not_evaluated' and validation['timing_gate'] == 'not_run'
    assert not any(path.name == 'timings.jsonl' for path in artifacts[0].rglob('*'))


@pytest.mark.parametrize('failure', ['gate', 'actual_dispatch', 'prepared_identity', 'step_count', 'logprob'])
def test_corrupt_or_incomplete_evidence_is_rejected(artifacts, failure):
    b1, capture, samples = artifacts
    if failure == 'gate':
        path = b1 / 'rank-0/gate.json'
        gate = json.loads(path.read_text())
        gate['passed'] = False
        write_json(path, gate)
    elif failure == 'actual_dispatch':
        path = b1 / 'rank-0/replay_forward-dispatch-slot0-layer0.npy'
        value = np.load(path)
        value[0] = [2, 3]
        np.save(path, value)
    elif failure == 'logprob':
        np.save(b1 / 'rank-0/replay-sample0.npy', np.array([-0.25], dtype=np.float32))
    elif failure == 'prepared_identity':
        samples[0]['tokens'][0] = 99
    elif failure == 'step_count':
        for rank in range(2):
            path = b1 / f'rank-{rank}/schedule.json'
            value = json.loads(path.read_text())
            value['global_batch_sizes'] = [3]
            write_json(path, value)
    with pytest.raises(ValueError):
        build(b1, capture, samples, run_id='fixture-run')


def test_interleaved_ep_groups_and_dense_dp_order_use_actual_membership(tmp_path):
    topology, schedule, samples, collectives, arrays, validation, _ = build(
        *make_artifacts(tmp_path, world=4), run_id='fixture-interleaved')
    assert topology['dense_dp_rank_to_global'] == [2, 0, 3, 1]
    assert [group['members'] for group in topology['ep_groups']] == [[0, 2], [1, 3]]
    assert len(collectives) == 2 and len(schedule['cells']) == 4
    for group_id, members in enumerate(([0, 2], [1, 3])):
        assert sorted(cell['source_rank'] for cell in schedule['cells'] if cell['group_id'] == group_id) == members
        np.testing.assert_array_equal(arrays['collective_H'][group_id, 0], [12, 4])
    assert validation['actual_logprob_values_checked'] == 12
    assert validation['actual_logprob_bitwise_match']


def test_relative_preparation_provenance_does_not_depend_on_summary_cwd(artifacts, monkeypatch):
    b1, capture, samples = artifacts
    for sample in samples:
        sample['metadata']['capture'] = str(Path(capture.parent.name) / capture.name)
    monkeypatch.chdir(b1)
    result = build(b1, capture, samples, run_id='fixture-other-cwd')
    assert result[5]['actual_dispatch_H_U_per_expert_match']
    assert result[5]['actual_logprob_bitwise_match']
