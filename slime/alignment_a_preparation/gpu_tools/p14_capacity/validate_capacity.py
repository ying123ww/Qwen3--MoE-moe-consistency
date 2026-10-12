"""Independent finite P14 model, allocator, guard, padding and routing audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(path.read_text())


def source_gate(run):
    check(load(run / 'state.json')['status'] == 'complete', f'{run}: coordinator incomplete')
    for name, digest in load(run / 'manifest.json')['source_sha256'].items():
        check(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest,
              f'Source changed after run: {name}')


def workloads(data):
    names = [f'batch{n}' for n in (1, 2, 3, 4, 7, 8, 12)] + [
        'variable4', 'mixed-response4', 'repeat-prompt8']
    check(data['complete'] and [x['case'] for x in data['cases']] == names,
          'Incomplete finite multibatch workload')
    targets = 0
    requests = 0
    for case in data['cases']:
        rows = case['records']
        count = int(case['case'][5:]) if case['case'].startswith('batch') else (
            8 if case['case'] == 'repeat-prompt8' else 4)
        check(len(rows) == count, 'Incomplete request group')
        requests += count
        for row in rows:
            n = row['response_length']
            check(n > 0 and len(row['tokens']) == n and len(row['logprob_bits']) == n,
                  'Incomplete target tokens/logprobs')
            check(all(math.isfinite(struct.unpack('<f', struct.pack('<i', b))[0])
                      for b in row['logprob_bits']), 'Nonfinite FP32 target bits')
            check(row['route_shape'] == [len(row['prompt']) + n - 1, 48, 8] and
                  len(row['route_sha256']) == 64, 'Incomplete actual route coverage')
            targets += n
        if case['case'] == 'variable4':
            baseline = next(c['records'] for c in data['cases'] if c['case'] == 'batch4')
            # The probe slices existing frozen prompts; [:192] cannot extend
            # a source prompt of length137. Validate actual tokens, not an
            # invented length implied by the requested slice limit.
            check(all(x['prompt'] == base['prompt'][:limit] for x, base, limit in
                      zip(rows, baseline, (17, 63, 129, 192), strict=True)) and
                  len({len(x['prompt']) for x in rows}) == 4, 'Variable input slices absent')
        if case['case'] == 'mixed-response4':
            check([x['response_length'] for x in rows] == [1, 4, 16, 32], 'Mixed responses absent')
        if case['case'] == 'repeat-prompt8':
            check(all(x['prompt'] == rows[0]['prompt'] for x in rows), 'Repeated prompt case differs')
    check(requests == 53 and targets == 837, 'Unexpected finite workload size')
    return dict(requests=requests, target_tokens=targets, cases=len(names))


def same_workload(left, right):
    for a, b in zip(left['cases'], right['cases'], strict=True):
        check(a['case'] == b['case'], 'Case identity differs')
        for x, y in zip(a['records'], b['records'], strict=True):
            check(all(x[k] == y[k] for k in ('prompt', 'response_length', 'tokens',
                      'logprob_bits', 'route_sha256', 'route_shape')), 'Model/eager/capacity result differs')


def audit(model_run, shadow_run, routing_run):
    for run in (model_run, shadow_run, routing_run):
        source_gate(run)
        check(load(run / 'results.json')['complete'], 'Run result incomplete')
    models = {}
    summary = {}
    for cap in (16, 8):
        root = model_run / f'cap-{cap}'
        models[cap] = load(root / 'results.json')
        coverage = workloads(models[cap])
        check(models[cap]['engine_initialization_s'] > 0, 'Initialization cost absent')
        records = [load(p) for p in (root / 'memory-evidence').glob('memory-rank-*.json')]
        check({x['rank'] for x in records} == set(range(8)), 'Missing allocator rank')
        required = {(r, s) for r in range(8) for s in (2, 4)}
        proofs = [load(p) for p in (root / 'graph-evidence').glob('fixed-capture-size-*-rank-*.json')]
        replays = [load(p) for p in (root / 'graph-evidence').glob('fixed-full-replay-size-*-rank-*.json')]
        check({(x['rank'], x['size']) for x in proofs} == required and
              {(x['rank'], x['size']) for x in replays} == required, 'Missing graph bucket/rank')
        for row in proofs:
            check(row['capacity'] == cap and row['dispatches_this_shape'] == 48 and
                  row['captured_layers'] == list(range(48)) and row['source_rows'] and
                  all(int(k) <= cap // 4 for k in row['source_rows']), 'Physical receive bound unproved')
        check(all(x['successful_replay'] and x['backend_class'] == 'FullCudaGraphBackend'
                  for x in replays), 'Native full replay absent')
        for row in records:
            counts = row['counts']
            check(counts.get('unsafe_physical_send_rejected') == 1 and
                  counts.get('pure_eager_decode', 0) == 0 and counts['full_replays'] > 0 and
                  set(row['replay_shapes']) == {'2', '4'}, 'Guard/full/no-fallback coverage absent')
            check({x['size'] for x in row['captures']} == {2, 4} and
                  len(row['captures']) == 2, 'Capture cost records missing')
            for capture in row['captures']:
                check(capture['capture_and_backend_warmup_s'] > 0 and
                      capture['graph_count'] in (1, 2), 'Invalid capture cost')
            m = row['final_memory']
            check(0 < m['allocated_bytes'] <= m['reserved_bytes'] <= m['peak_reserved_bytes']
                  and m['peak_allocated_bytes'] >= m['allocated_bytes'], 'Invalid allocator memory')
        resident = load(root / 'nvml-resident.json')['observations']
        check(resident and all({g['index'] for g in x['gpus']} == set(range(8))
                              for x in resident), 'Whole-device resident samples missing')
        reference = load(shadow_run / f'cap-{cap}' / 'results.json')
        workloads(reference)
        same_workload(models[cap], reference)
        shadow = [load(p) for p in (shadow_run / f'cap-{cap}' / 'graph-evidence')
                  .glob('fixed-shadow-rank-*.json')]
        check({x['rank'] for x in shadow} == set(range(8)) and
              all(x['counts']['shadow_calls'] > 0 and x['counts']['shadow_elements'] > 0
                  for x in shadow), 'No all-rank actual-M expert shadow')
        summary[cap] = dict(coverage=coverage, memory=records,
                           engine_initialization_s=models[cap]['engine_initialization_s'],
                           max_rank_capture_and_warmup_sum_s=max(
                               sum(c['capture_and_backend_warmup_s'] for c in x['captures'])
                               for x in records), resident_samples=len(resident),
                           nvml_sampled_peak_mib={i: max(g['used_mib'] for x in resident
                             for g in x['gpus'] if g['index'] == i) for i in range(8)},
                           expert_shadow_calls=sum(x['counts']['shadow_calls'] for x in shadow),
                           expert_shadow_elements=sum(x['counts']['shadow_elements'] for x in shadow))
    same_workload(models[16], models[8])
    routing = [load(p) for p in routing_run.glob('controlled-rank-*.json')]
    check({x['rank'] for x in routing} == set(range(8)), 'Missing controlled routing rank')
    for row in routing:
        ep = row['rank'] % 4
        check(row['complete'] and len(row['cases']) == 5, 'Routing cases missing')
        for case in row['cases']:
            name = case['case']
            expected = {'all-to-rank0': 8 if ep == 0 else 0,
                        'balanced-local': 2, 'spread-all-ranks': 8,
                        'one-idle-sender': 6 if ep == 0 else 0, 'all-idle': 0}[name]
            logical_send = 0 if name == 'all-idle' or (name == 'one-idle-sender' and ep == 3) else 2
            check(case['physical_send_rows'] == 2 and case['logical_send_rows'] == logical_send,
                  'Logical idle sender not observed')
            for cap in ('8', '16'):
                value = case['capacities'][cap]
                check(value['real_receive_rows'] == expected and value['zero_receive'] == (expected == 0)
                      and value['physical_receive_shape'] == [int(cap), 2048] and
                      value['output_bit_mismatches'] == 0, 'Routing count/output gate failed')
    return dict(passed=True, capacities=summary, controlled_rank_cases=40,
                scope='Finite cap8/16 model/eager/padding equivalence, EP4 physical send bound, '
                      'zero/idle/skew routing, allocator/resident/capture costs; not cross-node '
                      'or latency confirmation. NVML sampled peaks may miss transients.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--shadow-run', type=Path, required=True)
    parser.add_argument('--routing-run', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.model_run, args.shadow_run, args.routing_run)
    (args.model_run / 'independent-capacity-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('P14_INDEPENDENT_CAPACITY_AUDIT_PASSED')
