"""Strict opt-in adapter from explicit B2 cells to Slime's DP schedule tuple."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def input_identity(lengths, tokens, rollout_indices):
    require(len(lengths) == len(tokens) == len(rollout_indices), 'Input identity coverage mismatch')
    require(all(type(n) is int and n > 0 for n in lengths), 'Invalid sample length')
    require(all(len(t) == n for t, n in zip(tokens, lengths)), 'Token length mismatch')
    return {'lengths': list(lengths), 'rollout_indices': list(rollout_indices),
            'tokens_sha256': [digest(list(map(int, t))) for t in tokens]}


def make_manifest(native_schedule, candidate_schedule, *, sample_to_global_index,
                  source_rank_to_dp_rank, total_lengths, token_sequences, rollout_indices,
                  pad_multiplier, max_tokens_per_bin, mode='candidate'):
    """Caller supplies explicit provenance mapping; never parses a sample_id string."""
    require(mode in ('native', 'candidate'), 'Invalid mode')
    return {'schema': 'routepack-runtime-v1', 'mode': mode,
            'sample_to_global_index': dict(sample_to_global_index),
            'source_rank_to_dp_rank': {str(k): v for k, v in source_rank_to_dp_rank.items()},
            'input_identity': input_identity(total_lengths, token_sequences, rollout_indices),
            'pad_multiplier': pad_multiplier, 'max_tokens_per_bin': max_tokens_per_bin,
            'native_cells': native_schedule['cells'], 'candidate_cells': candidate_schedule['cells']}


def _cells(cells, mapping, rank_mapping, n):
    result, group_by_rank = {}, {}
    for c in cells:
        require(all(type(c[k]) is int and c[k] >= 0 for k in ('step', 'slot', 'source_rank', 'group_id')), 'Invalid cell coordinates')
        require(str(c['source_rank']) in rank_mapping, 'Unknown source rank')
        rank = rank_mapping[str(c['source_rank'])]
        key = (c['step'], rank, c['slot'])
        require(key not in result, 'Duplicate cell')
        require(bool(c['sample_ids']), 'Empty cell')
        require(c['source_rank'] not in group_by_rank or group_by_rank[c['source_rank']] == c['group_id'], 'Source rank changes EP group')
        group_by_rank[c['source_rank']] = c['group_id']
        require(all(s in mapping for s in c['sample_ids']), 'Unknown sample ID')
        indices = [mapping[s] for s in c['sample_ids']]
        require(all(type(i) is int and 0 <= i < n for i in indices), 'Invalid global index')
        result[key] = (c['group_id'], indices)
    return result


def apply_schedule(native_result, *, total_lengths, token_sequences, rollout_indices,
                   dp_size, max_tokens_per_bin, pad_multiplier, path):
    """Validate against freshly computed native schedule; return a new DP tuple.

    max_tokens_per_bin must be live args.max_tokens_per_gpu * cp_size;
    pad_multiplier must be the live data padding multiple, not inferred here.
    The caller must pass actual current tokens (including responses) and rollout IDs.
    """
    m = json.loads(Path(path).read_text())
    require(m.get('schema') == 'routepack-runtime-v1', 'Unsupported manifest schema')
    require(m.get('mode') in ('native', 'candidate'), 'Invalid mode')
    require(type(pad_multiplier) is int and pad_multiplier > 0, 'Invalid padding multiple')
    require(type(max_tokens_per_bin) is int and max_tokens_per_bin > 0, 'Invalid capacity')
    require(m['pad_multiplier'] == pad_multiplier and m['max_tokens_per_bin'] == max_tokens_per_bin, 'Runtime capacity/padding config mismatch')
    require(m['input_identity'] == input_identity(total_lengths, token_sequences, rollout_indices), 'Stale input identity')
    ranks = m['source_rank_to_dp_rank']
    require(len(ranks) == dp_size and all(type(v) is int for v in ranks.values()) and set(ranks.values()) == set(range(dp_size)), 'DP rank mapping is not bijective')
    mapping = m['sample_to_global_index']
    require(all(type(i) is int for i in mapping.values()) and len(set(mapping.values())) == len(mapping), 'Sample mapping is not injective')
    base = _cells(m['native_cells'], mapping, ranks, len(total_lengths))
    candidate = _cells(m['candidate_cells'], mapping, ranks, len(total_lengths))
    partitions, micro, counts, batches = native_result
    require(len(partitions) == len(micro) == dp_size and len(counts) == len(batches), 'Invalid native dimensions')
    live = {}
    for rank in range(dp_size):
        offset = 0
        for step, count in enumerate(counts):
            for slot in range(count):
                local = micro[rank][offset+slot]
                live[(step, rank, slot)] = [partitions[rank][i] for i in local]
            offset += count
        require(offset == len(micro[rank]), 'Native microbatch coverage mismatch')
    require(set(base) == set(live) == set(candidate), 'Step/rank/slot cells changed')
    require(all(base[k][1] == live[k] for k in live), 'Manifest native does not match fresh native schedule')
    kept = [i for indices in live.values() for i in indices]
    require(len(set(kept)) == len(kept) and set(mapping.values()) == set(kept), 'Mapping does not cover kept samples exactly')
    require(Counter(i for _, inds in candidate.values() for i in inds) == Counter(kept), 'Samples dropped or duplicated')
    original_step = {i: k[0] for k, inds in live.items() for i in inds}
    old_padding, new_padding = defaultdict(int), defaultdict(int)
    for key, (group, indices) in candidate.items():
        require(group == base[key][0], 'EP group changed')
        require(len(indices) == len(live[key]), 'Cell sample count changed')
        require(all(original_step[i] == key[0] for i in indices), 'Sample crosses optimizer step')
        size = sum(total_lengths[i] for i in indices)
        old_size = sum(total_lengths[i] for i in live[key])
        require(size <= max_tokens_per_bin, 'Candidate exceeds token capacity')
        require(size + (-size) % pad_multiplier <= old_size + (-old_size) % pad_multiplier,
                'Candidate exceeds native cell padded capacity')
        old_padding[key[0]] += (-old_size) % pad_multiplier
        new_padding[key[0]] += (-size) % pad_multiplier
    require(all(new_padding[s] <= old_padding[s] for s in old_padding), 'Step padding increased')
    if m['mode'] == 'native':
        require(all(candidate[k] == base[k] for k in base), 'Native mode contains intervention')
    new_partitions, new_micro = [[] for _ in range(dp_size)], [[] for _ in range(dp_size)]
    for step, count in enumerate(counts):
        for rank in range(dp_size):
            for slot in range(count):
                indices = candidate[(step, rank, slot)][1]
                start = len(new_partitions[rank])
                new_partitions[rank].extend(indices)
                new_micro[rank].append(list(range(start, start+len(indices))))
    return new_partitions, new_micro, list(counts), list(batches)
