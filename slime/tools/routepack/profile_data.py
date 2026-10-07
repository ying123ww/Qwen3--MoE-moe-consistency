"""Offline B1/B2 validation; no scheduling decisions or GPU work.

Topology is accepted only from runtime rank exports. Input traces are .npy
integer arrays [tokens-1, layers, topk]; schedule cells name stable sample IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merge_topology(rows):
    require(bool(rows), 'No runtime topology records')
    world = rows[0]['world_size']
    require(sorted(r['global_rank'] for r in rows) == list(range(world)), 'Missing or duplicate global ranks')
    by_rank = {r['global_rank']: r for r in rows}
    for row in rows:
        require(row['world_size'] == world, 'world_size mismatch')
        require(row['vpp_size'] == 1, 'VPP != 1 unsupported')
        for name in ('dense_dp', 'ep', 'expert_dp', 'tp', 'expert_tp', 'cp', 'pp'):
            members = row['groups'][name]
            require(row['global_rank'] in members and len(set(members)) == len(members), f'Invalid {name} membership')
            require(all(rank in by_rank for rank in members), f'Unknown {name} rank')
            require(all(by_rank[rank]['groups'][name] == members for rank in members), f'Inconsistent ordered {name} membership')
        for name in ('tp', 'expert_tp', 'cp', 'pp'):
            require(len(row['groups'][name]) == 1, f'{name} != 1 unsupported')
        require(row['groups']['dense_dp'] == rows[0]['groups']['dense_dp'], 'Expected one dense DP group')
    groups = []
    for members in sorted({tuple(r['groups']['ep']) for r in rows}):
        layer_sets = [set(by_rank[r]['experts']) for r in members]
        require(all(s == layer_sets[0] for s in layer_sets), 'EP ranks disagree on layer IDs')
        owners = {}
        for layer in sorted(layer_sets[0], key=int):
            num_experts = by_rank[members[0]]['experts'][layer]['num_experts']
            owner = [-1] * num_experts
            for local_rank, rank in enumerate(members):
                entry = by_rank[rank]['experts'][layer]
                require(entry['num_experts'] == num_experts, 'num_experts mismatch')
                for expert in entry['local_expert_indices']:
                    require(0 <= expert < num_experts and owner[expert] == -1, 'Invalid/duplicate expert owner')
                    owner[expert] = local_rank
            require(-1 not in owner, 'Incomplete expert owner layout')
            owners[layer] = owner
        require(bool(owners), 'No actual MoE expert layout exported')
        groups.append({'group_id': len(groups), 'members': list(members), 'owners': owners})
    require(all(g['owners'] == groups[0]['owners'] for g in groups), 'Different owner layout across EP replicas unsupported')
    return {'schema_version': 1, 'source': 'runtime_model_and_process_groups', 'world_size': world,
            'ranks': sorted(rows, key=lambda r: r['global_rank']), 'ep_groups': groups,
            'dense_dp_rank_to_global': rows[0]['groups']['dense_dp'], 'v1_supported': True}


def native_schedule_cells(partitions, micro_batch_indices, num_microbatches, sample_ids, topology):
    """Translate actual build_dp_schedule outputs without changing placement."""
    ranks = topology['dense_dp_rank_to_global']
    require(len(partitions) == len(micro_batch_indices) == len(ranks), 'DP rank count mismatch')
    owner_group = {rank: g['group_id'] for g in topology['ep_groups'] for rank in g['members']}
    cells = []
    for dp_rank, global_rank in enumerate(ranks):
        require(len(micro_batch_indices[dp_rank]) == sum(num_microbatches), 'Microbatch count mismatch')
        offset = 0
        for step, count in enumerate(num_microbatches):
            for slot in range(count):
                local_indices = micro_batch_indices[dp_rank][offset + slot]
                ids = [sample_ids[partitions[dp_rank][i]] for i in local_indices]
                cells.append({'step': step, 'group_id': owner_group[global_rank], 'slot': slot,
                              'source_rank': global_rank, 'sample_ids': ids})
            offset += count
    return {'source': 'native_build_dp_schedule_outputs', 'cells': cells}


def route_histograms(routes, owners):
    """H counts assignments; U counts token/destination pairs, never bytes."""
    routes = np.asarray(routes)
    require(routes.ndim == 3 and np.issubdtype(routes.dtype, np.integer), 'Routes must be integer [token,layer,topk]')
    require(routes.shape[1] == len(owners) and routes.shape[2] > 0, 'Incomplete layer/topk coverage')
    num_experts = len(owners[0])
    require(all(len(o) == num_experts for o in owners), 'Inconsistent expert counts')
    require(routes.size == 0 or (routes.min() >= 0 and routes.max() < num_experts), 'Route ID out of range')
    require(routes.shape[2] <= num_experts, 'topk exceeds expert count')
    if routes.shape[2] > 1:
        require(not np.any(np.diff(np.sort(routes, axis=-1), axis=-1) == 0), 'Duplicate expert in token topk')
    ep = max(max(o) for o in owners) + 1
    h = np.zeros((len(owners), ep), dtype=np.int64)
    u = np.zeros_like(h)
    per_expert = np.zeros((len(owners), num_experts), dtype=np.int64)
    for layer, owner in enumerate(owners):
        per_expert[layer] = np.bincount(routes[:, layer].reshape(-1), minlength=num_experts)
        destinations = np.asarray(owner)[routes[:, layer]]
        for rank in range(ep):
            h[layer, rank] = np.count_nonzero(destinations == rank)
            u[layer, rank] = np.count_nonzero(np.any(destinations == rank, axis=-1))
    return h, u, per_expert


def padding_routes(count, layers, topk, num_experts):
    # cp_utils._pad_routed_experts restarts arange for EACH padding call.
    return np.arange(count * layers * topk, dtype=np.int64).reshape(count, layers, topk) % num_experts


def summarize(samples, traces, schedule, topology, pad_multiplier):
    require(bool(samples), 'No samples')
    require(pad_multiplier > 0, 'Invalid data_pad_size_multiplier')
    require(topology.get('source') == 'runtime_model_and_process_groups', 'Runtime topology required')
    checked = merge_topology(topology['ranks'])
    require(checked['ep_groups'] == topology['ep_groups'], 'Topology groups inconsistent with rank evidence')
    require(checked['dense_dp_rank_to_global'] == topology['dense_dp_rank_to_global'], 'Dense DP map inconsistent with rank evidence')
    owners_dict = topology['ep_groups'][0]['owners']
    layer_ids = sorted(owners_dict, key=int)
    owners = [owners_dict[l] for l in layer_ids]
    groups = {g['group_id']: g for g in topology['ep_groups']}
    sample_by_id = {s['sample_id']: s for s in samples}
    require(len(sample_by_id) == len(samples), 'Duplicate sample IDs')
    raw, terminal, metadata = {}, {}, []
    for sample_id, sample in sample_by_id.items():
        require(sample_id in traces, f'Missing trace: {sample_id}')
        routes = np.asarray(traces[sample_id])
        require(routes.ndim == 3, 'Routes must have token/layer/topk axes')
        require(routes.shape[0] == sample['token_count'] - 1, f'tokens-1 mismatch: {sample_id}')
        require(sample['layer_ids'] == [int(l) for l in layer_ids], 'Trace layer IDs differ from runtime model')
        require(0 <= sample['valid_loss_token_count'] <= sample['response_length'] <= sample['token_count'], 'Invalid sample lengths')
        for field in ('rollout_id', 'reward_group_id', 'model_version', 'expert_layout_version'):
            require(field in sample, f'Missing sample provenance: {field}')
        raw[sample_id] = route_histograms(routes, owners)
        terminal[sample_id] = route_histograms(padding_routes(1, *routes.shape[1:], len(owners[0])), owners)
        digest = hashlib.sha256(np.ascontiguousarray(routes, dtype='<i8').tobytes()).hexdigest()
        require(sample.get('trace_hash', digest) == digest, 'trace_hash mismatch (canonical little-endian int64)')
        metadata.append({**sample, 'trace_hash': digest, 'traced_token_count': routes.shape[0],
                         'synthetic_terminal_count': 1, 'summary_index': len(metadata)})
    require(len({s['model_version'] for s in samples}) == 1, 'Mixed model versions')
    require(len({s['expert_layout_version'] for s in samples}) == 1, 'Mixed expert layout versions')
    cells = schedule['cells']
    seen, collective, rollout_steps, cell_keys = [], defaultdict(list), {}, set()
    for cell in cells:
        key = (cell['step'], cell['group_id'], cell['slot'], cell['source_rank'])
        require(key not in cell_keys, 'Duplicate schedule cell')
        cell_keys.add(key)
        require(cell['group_id'] in groups and cell['source_rank'] in groups[cell['group_id']]['members'], 'Invalid source rank/group')
        require(bool(cell['sample_ids']), 'Empty microbatch unsupported')
        for sample_id in cell['sample_ids']:
            require(sample_id in sample_by_id, 'Unknown scheduled sample')
            sample = sample_by_id[sample_id]
            require(sample['step'] == cell['step'], 'Sample crosses optimizer step')
            rid = sample['rollout_id']
            require(rid not in rollout_steps or rollout_steps[rid] == cell['step'], 'Rollout crosses optimizer step')
            rollout_steps[rid] = cell['step']
        seen.extend(cell['sample_ids'])
        collective[key[:3]].append(cell)
    require(Counter(seen) == Counter(sample_by_id.keys()), 'Schedule duplicates or drops samples')
    for step in {c['step'] for c in cells}:
        slots_by_rank = [{c['slot'] for c in cells if c['step'] == step and c['source_rank'] == r}
                         for r in range(topology['world_size'])]
        require(bool(slots_by_rank[0]) and all(s == slots_by_rank[0] for s in slots_by_rank), 'Unequal rank slot sets')
        require(slots_by_rank[0] == set(range(len(slots_by_rank[0]))), 'Slots must be zero-based contiguous')
    arrays = {f'sample_{name}': np.stack([raw[s['sample_id']][i] for s in samples])
              for i, name in enumerate(('H', 'U', 'per_expert'))}
    arrays.update({f'terminal_{name}': np.stack([terminal[s['sample_id']][i] for s in samples])
                   for i, name in enumerate(('H', 'U', 'per_expert'))})
    records, collective_arrays, pad_arrays = [], [], []
    for key, bins in sorted(collective.items()):
        require({b['source_rank'] for b in bins} == set(groups[key[1]]['members']), 'Incomplete EP collective')
        totals = [np.zeros_like(a) for a in next(iter(raw.values()))]
        pads = [np.zeros_like(a) for a in totals]
        real_tokens = layout_padding = 0
        for bin_ in bins:
            token_count = sum(sample_by_id[s]['token_count'] for s in bin_['sample_ids'])
            real_tokens += token_count
            pad = (-token_count) % pad_multiplier
            layout_padding += pad
            topk = traces[bin_['sample_ids'][0]].shape[2]
            require(all(traces[s].shape[2] == topk for s in sample_by_id), 'Inconsistent topk')
            ph = route_histograms(padding_routes(pad, len(owners), topk, len(owners[0])), owners)
            for i in range(3):
                pads[i] += ph[i]
                totals[i] += ph[i]
                for sid in bin_['sample_ids']:
                    totals[i] += raw[sid][i] + terminal[sid][i]
        h = totals[0]
        records.append({'step': key[0], 'group_id': key[1], 'slot': key[2], 'training_tokens': real_tokens,
                        'layout_padding_tokens': layout_padding,
                        'EP_excess': float((h.max(axis=1) - h.mean(axis=1)).sum()),
                        'max_EP_peak_ratio': float((h.max(axis=1) / h.mean(axis=1)).max())})
        collective_arrays.append(totals)
        pad_arrays.append(pads)
    for i, name in enumerate(('H', 'U', 'per_expert')):
        arrays[f'collective_{name}'] = np.stack([a[i] for a in collective_arrays])
        arrays[f'layout_padding_{name}'] = np.stack([a[i] for a in pad_arrays])
    return metadata, records, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    merge = sub.add_parser('merge-topology')
    merge.add_argument('rank_directory', type=Path)
    merge.add_argument('output', type=Path)
    profile = sub.add_parser('summarize')
    profile.add_argument('input_directory', type=Path)
    profile.add_argument('--pad-multiplier', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'merge-topology':
        rows = [json.loads(p.read_text()) for p in sorted(args.rank_directory.glob('topology-rank-*.json'))]
        args.output.write_text(json.dumps(merge_topology(rows), indent=2) + '\n')
        return
    root = args.input_directory
    topology = json.loads((root / 'topology.json').read_text())
    samples = [json.loads(line) for line in (root / 'samples_input.jsonl').read_text().splitlines() if line.strip()]
    traces = {s['sample_id']: np.load(root / s['route_path'], allow_pickle=False) for s in samples}
    schedule = json.loads((root / 'schedule.json').read_text())
    metadata, records, arrays = summarize(samples, traces, schedule, topology, args.pad_multiplier)
    (root / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metadata))
    (root / 'collectives.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    np.savez_compressed(root / 'route_loads.npz', **arrays)
    print(json.dumps({'samples': len(metadata), 'collectives': len(records), 'timing_gate': 'pending',
                      'actual_dispatch_route_gate': 'pending', 'Go': False}))


if __name__ == '__main__':
    main()
