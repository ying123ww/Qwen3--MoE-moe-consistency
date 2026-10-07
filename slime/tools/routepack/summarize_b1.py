"""Build B2 load artifacts only from a completed, independently rechecked B1 gate.

No GPU calls, optimizer steps, timing estimates or scheduling decisions.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from profile_data import merge_topology, padding_routes, require, route_histograms, summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(b1, capture, prepared_samples, *, run_id):
    """Accept actual prepare_b1 samples separately to keep the core CPU-testable."""
    rank_paths = sorted(b1.glob('topology-rank-*.json'))
    topology_rows = [json.loads(path.read_text()) for path in rank_paths]
    topology = merge_topology(topology_rows)
    require(len({row['data_pad_size_multiplier'] for row in topology_rows}) == 1, 'Rank padding multiplier mismatch')
    require(all(row['use_rollout_routing_replay'] for row in topology_rows), 'Runtime R3 flag missing')
    multiplier = topology_rows[0]['data_pad_size_multiplier']
    owners_dict = topology['ep_groups'][0]['owners']
    layers = sorted(map(int, owners_dict))
    require(layers == list(range(48)), 'B1 Qwen3 gate requires all 48 layers')
    owners = [owners_dict[str(layer)] for layer in layers]
    layout_version = hashlib.sha256(json.dumps([g['owners'] for g in topology['ep_groups']], sort_keys=True).encode()).hexdigest()
    capture_path = capture / 'samples_captured.jsonl'
    captured = [json.loads(line) for line in capture_path.read_text().splitlines() if line.strip()]
    by_capture_id = {row['sample_id']: row for row in captured}
    require(len(by_capture_id) == len(captured), 'Duplicate capture IDs')
    by_index = {s['index']: s for s in prepared_samples}
    require(len(by_index) == len(prepared_samples), 'Duplicate prepared sample indices')
    metadata, traces, capture_logprobs = {}, {}, {}
    source_hashes = {str(path.resolve()): digest(path) for path in rank_paths + [capture_path]}
    for index, sample in by_index.items():
        provenance = sample['metadata']
        declared_capture = Path(provenance['capture'])
        if declared_capture.is_absolute():
            same_capture = declared_capture.resolve() == capture.resolve()
        else:
            # prepare_b1 preserves its invocation-relative path without a cwd.
            # Match that suffix against the explicit input, then verify contents
            # below; resolving it against this process's cwd would reject valid data.
            suffix = declared_capture.parts
            same_capture = bool(suffix) and '..' not in suffix and capture.resolve().parts[-len(suffix):] == suffix
        require(same_capture, 'Prepared rollout references a different capture')
        row = by_capture_id[provenance['routepack_sample_id']]
        require(row.get('effective_logprob_temperature') == 1.0, 'B1 requires effective logprob temperature 1.0')
        route_path, token_path = capture / row['route_path'], capture / row['token_path']
        route = np.load(route_path, allow_pickle=False)
        tokens = np.load(token_path, allow_pickle=False)
        require(hashlib.sha256(route.astype('<i8').tobytes()).hexdigest() == row['trace_hash'], 'Capture route hash mismatch')
        require(hashlib.sha256(tokens.astype('<i8').tobytes()).hexdigest() == row['token_hash'], 'Capture token hash mismatch')
        require(np.array_equal(np.asarray(sample['tokens']), tokens), 'Prepared token identities differ from capture')
        require(np.array_equal(np.asarray(sample['rollout_routed_experts']), route), 'Prepared routes differ from capture')
        require(sample['response_length'] == row['response_length'], 'Response length mismatch')
        logprob_path = capture / row['response_logprob_path']
        logprobs = np.load(logprob_path, allow_pickle=False)
        require(logprobs.dtype == np.float32 and logprobs.shape == (row['response_length'],)
                and np.isfinite(logprobs).all(), 'Invalid captured response logprobs')
        prepared_logprobs = np.asarray(sample['rollout_log_probs'], dtype=np.float32)
        require(prepared_logprobs.shape == logprobs.shape
                and np.array_equal(prepared_logprobs.view(np.uint32), logprobs.view(np.uint32)),
                'Prepared logprobs differ from capture')
        capture_logprobs[index] = logprobs
        mask = np.asarray(sample['loss_mask'])
        require(mask.shape == (row['response_length'],) and np.isin(mask, [0, 1]).all(), 'Invalid loss mask')
        sample_id = f'{run_id}:sample{index}'
        metadata[index] = {**row, 'sample_id': sample_id, 'prepared_sample_index': index,
                           'capture_local_sample_id': row['sample_id'],
                           'grouping_note': 'rollout/group IDs and loss mask are forward-only fixture fields; not production reward semantics',
                           'fixture_identifiers': True, 'route_path': str(route_path.resolve()),
                           'token_path': str(token_path.resolve()), 'response_logprob_path': str(logprob_path.resolve()),
                           'rollout_id': sample['rollout_id'],
                           'reward_group_id': sample['group_index'], 'valid_loss_token_count': int(mask.sum()),
                           'expert_layout_version': layout_version, 'model_version': row['model_version']}
        traces[sample_id] = route
        for path in (route_path, token_path, logprob_path):
            source_hashes[str(path.resolve())] = digest(path)
    cells, seen, dispatch_loads = [], [], {}
    rank_to_group = {rank: g['group_id'] for g in topology['ep_groups'] for rank in g['members']}
    batch_contract = None
    actual_route_files_checked = actual_rows_checked = actual_logprob_values_checked = 0
    for rank in topology['dense_dp_rank_to_global']:
        directory = b1 / f'rank-{rank}'
        schedule_path, gate_path = directory / 'schedule.json', directory / 'gate.json'
        schedule = json.loads(schedule_path.read_text())
        gate = json.loads(gate_path.read_text())
        require(schedule['rank'] == gate['rank'] == rank, 'Rank metadata mismatch')
        require(gate['passed'] and gate['r3_all_48_layers_consumed'], f'B1 rank {rank} did not pass')
        require(gate['optimizer_steps'] == 0, 'Expected forward-only B1 gate')
        for path in (schedule_path, gate_path):
            source_hashes[str(path.resolve())] = digest(path)
        indices = schedule['sample_indices']
        require(len(indices) == len(schedule['partition']) == len(schedule['total_lengths']), 'Rank partition metadata mismatch')
        require(len(indices) == len(set(indices)), 'Duplicate rank sample indices')
        for partition, index, length in zip(schedule['partition'], indices, schedule['total_lengths'], strict=True):
            require(prepared_samples[partition]['index'] == index, 'Rank partition does not map to prepared sample index')
            require(len(by_index[index]['tokens']) == length, 'Rank token length mismatch')
        contract = (schedule['num_microbatches'], schedule['global_batch_sizes'], schedule['rollout_id'])
        require(batch_contract is None or contract == batch_contract, 'Ranks disagree on step/microbatch contract')
        batch_contract = contract
        require(len(schedule['num_microbatches']) == len(schedule['global_batch_sizes']), 'Step metadata mismatch')
        require(sum(schedule['num_microbatches']) == len(schedule['micro_batch_indices']) == len(schedule['identities']), 'Slot coverage mismatch')
        require(Counter(x['sample_index'] for x in gate['logprobs']) == Counter(indices), 'Incomplete gate logprob samples')
        for row in gate['logprobs']:
            require(row['response_tokens'] == by_index[row['sample_index']]['response_length'], 'Gate logprob length mismatch')
            require(all(row[key] for key in ('aligned_vs_rollout_exact', 'replay_vs_rollout_exact', 'replay_vs_aligned_exact')), 'Gate logprob exact check failed')
            index = row['sample_index']
            expected_logprobs = capture_logprobs[index]
            for kind in ('aligned', 'replay', 'rollout'):
                path = directory / f'{kind}-sample{index}.npy'
                actual = np.load(path, allow_pickle=False)
                require(actual.dtype == np.float32 and actual.shape == expected_logprobs.shape
                        and np.isfinite(actual).all(), 'Invalid actual response logprobs')
                require(np.array_equal(actual.view(np.uint32), expected_logprobs.view(np.uint32)),
                        'Actual logprob file differs from capture')
                source_hashes[str(path.resolve())] = digest(path)
                actual_logprob_values_checked += len(actual)
        observations = {(row['stage'], row['kind'], row['slot'], row['layer']): row for row in gate['observations']}
        expected_keys = {(stage, kind, slot, layer) for stage in ('fallthrough', 'replay_forward')
                         for kind in ('router', 'dispatch') for slot in range(len(schedule['micro_batch_indices'])) for layer in layers}
        require(len(observations) == len(gate['observations']) and set(observations) == expected_keys, 'Incomplete/duplicate route observations')
        offset = 0
        for step, count in enumerate(schedule['num_microbatches']):
            for slot in range(count):
                absolute_slot = offset + slot
                local = schedule['micro_batch_indices'][absolute_slot]
                selected_indices = [indices[i] for i in local]
                sample_ids = [metadata[i]['sample_id'] for i in selected_indices]
                cells.append({'step': step, 'group_id': rank_to_group[rank], 'slot': slot,
                              'source_rank': rank, 'sample_ids': sample_ids})
                seen.extend(selected_indices)
                expected_parts, identities = [], []
                for index in selected_indices:
                    sample_id = metadata[index]['sample_id']
                    require(metadata[index].get('step', step) == step, 'Sample crosses optimizer step')
                    metadata[index]['step'] = step
                    route = traces[sample_id]
                    expected_parts.extend([route, padding_routes(1, len(layers), route.shape[2], len(owners[0]))])
                    length = len(by_index[index]['tokens'])
                    identities.extend([[index, pos, 'real' if pos < length - 1 else 'terminal'] for pos in range(length)])
                padded = (-len(identities)) % multiplier
                expected_parts.append(padding_routes(padded, len(layers), expected_parts[0].shape[2], len(owners[0])))
                identities.extend([[-1, pos, 'padding'] for pos in range(padded)])
                require(identities == schedule['identities'][absolute_slot], 'Actual B1 identity layout differs from reconstructed layout')
                expected = np.concatenate(expected_parts)
                replay_dispatch = np.empty_like(expected)
                real = np.array([row[2] == 'real' for row in identities])
                for stage in ('fallthrough', 'replay_forward'):
                    selected = real if stage == 'fallthrough' else np.ones(len(real), dtype=bool)
                    for kind in ('router', 'dispatch'):
                        for layer in layers:
                            row = observations[stage, kind, absolute_slot, layer]
                            require(row['mismatched_rows'] == 0 and row['rows_checked'] == int(selected.sum()), 'B1 observation mismatch')
                            path = directory / row['path']
                            actual = np.load(path, allow_pickle=False)
                            require(actual.shape == expected[:, layer].shape and np.issubdtype(actual.dtype, np.integer), 'Actual route shape/dtype mismatch')
                            require(np.array_equal(actual[selected], expected[:, layer][selected]), 'Actual route file differs from capture/replay layout')
                            source_hashes[str(path.resolve())] = digest(path)
                            actual_route_files_checked += 1
                            actual_rows_checked += int(selected.sum())
                            if stage == 'replay_forward' and kind == 'dispatch':
                                replay_dispatch[:, layer] = actual
                key = (step, rank_to_group[rank], slot)
                loads = route_histograms(replay_dispatch, owners)
                if key not in dispatch_loads:
                    dispatch_loads[key] = [np.zeros_like(array) for array in loads]
                for target, values in zip(dispatch_loads[key], loads, strict=True):
                    target += values
            offset += count
    require(Counter(seen) == Counter(by_index.keys()), 'Actual schedule duplicates/drops prepared samples')
    for step, expected_rollouts in enumerate(batch_contract[1]):
        require(len({m['rollout_id'] for m in metadata.values() if m['step'] == step}) == expected_rollouts, 'Actual step rollout count differs from global_batch_sizes')
    schedule = {'source': 'actual_b1_rank_schedules', 'cells': cells, 'num_microbatches': batch_contract[0],
                'global_batch_sizes': batch_contract[1], 'rollout_execution_id': batch_contract[2]}
    samples, collectives, arrays = summarize([metadata[i] for i in sorted(metadata)], traces, schedule, topology, multiplier)
    for column, name in enumerate(('H', 'U', 'per_expert')):
        actual = np.stack([dispatch_loads[r['step'], r['group_id'], r['slot']][column] for r in collectives])
        require(np.array_equal(actual, arrays[f'collective_{name}']), f'Actual dispatch {name} differs from collective model')
        arrays[f'actual_dispatch_{name}'] = actual
    validation = {'b1_forward_gate': 'passed_and_route_files_rechecked', 'sample_count': len(samples),
                  'collective_count': len(collectives), 'actual_route_files_checked': actual_route_files_checked,
                  'actual_route_rows_checked': actual_rows_checked, 'data_pad_size_multiplier': multiplier,
                  'actual_dispatch_H_U_per_expert_match': True, 'optimizer_steps': 0,
                  'actual_logprob_values_checked': actual_logprob_values_checked,
                  'actual_logprob_bitwise_match': True,
                  'timing_gate': 'not_run', 'schedule_intervention_gate': 'not_run', 'go_decision': 'not_evaluated',
                  'scope': 'one original schedule; raw routes + terminal + bin padding; counts are not communication bytes or time'}
    return topology, schedule, samples, collectives, arrays, validation, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b1-directory', type=Path, required=True)
    parser.add_argument('--capture-directory', type=Path, required=True)
    parser.add_argument('--prepared-rollout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    import torch
    prepared = torch.load(args.prepared_rollout, map_location='cpu', weights_only=True)['samples']
    topology, schedule, samples, collectives, arrays, validation, hashes = build(
        args.b1_directory, args.capture_directory, prepared, run_id=args.run_id)
    hashes[str(args.prepared_rollout.resolve())] = digest(args.prepared_rollout)
    hashes[str(Path(__file__).resolve())] = digest(Path(__file__))
    profile_source = Path(__file__).with_name('profile_data.py')
    hashes[str(profile_source.resolve())] = digest(profile_source)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [('topology.json', topology), ('schedule.json', schedule), ('validation.json', validation), ('source_hashes.json', hashes)]:
        (args.output / name).write_text(json.dumps(data, indent=2) + '\n')
    for name, rows in [('samples.jsonl', samples), ('collectives.jsonl', collectives)]:
        (args.output / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    np.savez_compressed(args.output / 'route_loads.npz', **arrays)
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
