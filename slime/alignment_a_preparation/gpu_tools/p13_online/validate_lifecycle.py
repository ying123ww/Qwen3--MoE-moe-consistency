"""Independent audit of actual online optimizer, sync, and graph lifecycle evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit(run):
    state = json.loads((run / 'state.json').read_text())
    check(state['status'] == 'online-logprob-complete', 'Online coordinator has not passed')
    manifest = json.loads((run / 'manifest.json').read_text())
    for source, digest in manifest['source_sha256'].items():
        check(hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest,
              f'Runtime source changed: {source}')
    bits = json.loads((run / 'target-logprob-bit-gate.json').read_text())
    check(set(bits) == {'0', '1', '2'}, 'Missing version target-token gate')
    check(all(x['target_tokens'] == 1024 and x['bit_mismatches'] == 0 and
              x['bitwise_exact'] for x in bits.values()), 'Target token bit gate failed')
    comparisons = [json.loads((run / 'replay-validation' / f'version-{v}.json').read_text())
                   for v in range(3)]
    for version, row in enumerate(comparisons):
        check(row['rollout_id'] == version and row['model_equal'] and
              row['old_replay'] == row['recaptured'], 'Old/re-captured outputs differ')
        check(len(row['old_replay']) == 8 and all(len(x['bits']) == 16 and
              len(x['tokens']) == 16 for x in row['old_replay']), 'Incomplete greedy probes')
    check(comparisons[2]['replacement_recovered'] and
          comparisons[2]['replacement_reference'] == comparisons[2]['recaptured'],
          'Unchanged storage replacement recovery differs')
    changed_versions = [v for v in (1, 2) if
                        comparisons[v]['old_replay'] != comparisons[v - 1]['old_replay']]
    # Optimizer master deltas and full synchronization are separate proofs;
    # identical BF16 greedy outputs alone cannot prove that graph values changed.
    check(changed_versions, 'No actual post-update greedy output change observed')
    ranks = {}
    for rank in range(8):
        updates = [json.loads(x) for x in
                   (run / 'training-update-evidence' / f'updates-rank-{rank}.jsonl')
                   .read_text().splitlines()]
        two = [x for x in updates if x['rollout_id'] in (0, 1)]
        check(len(two) == 2 and {x['rollout_id'] for x in two} == {0, 1},
              f'rank{rank}: missing two optimizer updates')
        check(all(x['rank'] == rank and x['lr'] == 1e-6 and math.isfinite(x['gradient_norm'])
                  and x['gradient_norm'] > 0 and 0 < x['sampled_master_changed_elements']
                  <= x['sampled_master_elements'] for x in two),
              f'rank{rank}: nonzero optimizer witness failed')
        events = [json.loads(x) for x in
                  (run / 'lifecycle' / f'lifecycle-rank-{rank}.jsonl').read_text().splitlines()]
        active = {}
        captures = 0
        replacements = 0
        sync_calls = 0
        sync_epoch_calls = 0
        synced_epochs = []
        old_graph_after_sync = []
        for row in events:
            check(row['rank'] == rank, 'Cross-rank evidence')
            event = row['event']
            if event == 'capture':
                check(row['parameter_count'] == 435 and row['kv_tensor_count'] == 96,
                      f'rank{rank}: incomplete whole-model pointer coverage')
                check(row['size'] in (2, 4) and not row['asleep'] and
                      not row['invalidated'], 'Invalid capture state')
                active[row['size']] = row
                captures += 1
            elif event.startswith('update_weights_from_'):
                sync_calls += 1
                sync_epoch_calls += 1
                check(not row['address_changes'] and not row['invalidated'],
                      f'rank{rank}: weight synchronization replaced captured storage')
            elif event == 'release':
                check(row['asleep'], 'Release not reflected in ledger')
                sync_epoch_calls = 0
            elif event == 'resume':
                check(not row['asleep'] and not row['invalidated'] and
                      not row['address_changes'], 'Unsafe memory resume')
                # Actual v89 full synchronization has two non-expert buckets
                # followed by 48 expert buckets; count every successful worker call.
                check(sync_epoch_calls == 50, f'rank{rank}: incomplete full-sync epoch')
                synced_epochs.append(row['update_calls'])
            elif event == 'first_replay':
                check(row['size'] in active and not row['asleep'] and
                      not row['invalidated'], 'Replay before capture/restoration')
                captured = active[row['size']]
                check(row['graph_object_id'] == captured['graph_object_id'] and
                      row['pool_id'] == captured['pool_id'], 'Replay graph/pool identity differs')
                if row['update_calls'] in synced_epochs and row['update_calls'] > captured['update_calls']:
                    old_graph_after_sync.append(row['update_calls'])
            elif event == 'replacement_negative':
                check(row['old_replay_rejected'] and row['invalidated'] and
                      len(row['address_changes']) == 1, 'Replacement did not reject old graph')
                replacements += 1
        check(len(synced_epochs) >= 3 and len(set(old_graph_after_sync)) >= 3,
              f'rank{rank}: initial/two updated old-graph replay epochs missing')
        check(replacements == 1 and captures >= 10 and set(active) == {2, 4},
              f'rank{rank}: replacement/re-capture coverage incomplete')
        for size in (2, 4):
            addresses = json.loads((run / 'lifecycle' /
                                   f'addresses-size-{size}-rank-{rank}.json').read_text())
            check(sum(k.startswith('weight:') for k in addresses) == 435 and
                  sum(k.startswith('kv:') for k in addresses) == 96,
                  'Stored pointer ledger lacks full model/KV tensors')
            check(any(k.endswith('lm_head.weight') for k in addresses) and
                  'request:req_to_token' in addresses, 'LM head/request pointers missing')
            digest = hashlib.sha256(json.dumps(addresses, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()
            check(digest == active[size]['address_digest'], 'Final address snapshot differs from capture')
        ranks[rank] = dict(nonzero_updates=two, successful_weight_bucket_calls=sync_calls,
                           full_sync_update_calls=synced_epochs,
                           old_graph_after_sync_update_calls=sorted(set(old_graph_after_sync)),
                           capture_calls=captures, rejected_storage_replacements=replacements)
    return dict(passed=True, ranks=ranks, versions=bits,
                changed_greedy_versions=changed_versions,
                scope='Actual v89 colocated full tensor sync, two real StatelessAdam/reset '
                      'updates, 3x1024 target bits, whole-model pointer/graph/pool lifecycle; '
                      'not persistent Adam or long-run training validation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    (args.run / 'independent-online-lifecycle-validation.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print('P13_INDEPENDENT_ONLINE_LIFECYCLE_PASSED')
