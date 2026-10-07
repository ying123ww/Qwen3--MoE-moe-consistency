"""Opt-in CPU candidate schedules; never changes the live training scheduler."""
from copy import deepcopy
import argparse
import hashlib
import json
import random
from time import perf_counter
from pathlib import Path

import numpy as np

try:
    from .profile_data import summarize, padding_routes, route_histograms, require
except ImportError:
    from profile_data import summarize, padding_routes, route_histograms, require


def excess(h):
    """Sum of per-layer receiver peaks above the mean, not a layer histogram."""
    return float((h.max(axis=-1) - h.mean(axis=-1)).sum())


def refine(samples, traces, schedule, topology, pad_multiplier, *, max_candidates=10000,
           max_swaps=100, max_bin_tokens=None, mode="sampled_best", seed=0,
           max_iterations=100, max_wall_ms=None, candidate_batch_size=25):
    started = perf_counter()
    require(mode in ("random", "sampled_best"), "Invalid mode")
    require(max_iterations >= 0, "Negative iteration budget")
    require(max_wall_ms is None or (np.isfinite(max_wall_ms) and max_wall_ms >= 0), "Invalid wall budget")
    require(candidate_batch_size > 0, "Invalid candidate batch size")
    rng = random.Random(seed)
    deadline = float("inf") if max_wall_ms is None else started + max_wall_ms / 1000
    require(max_candidates >= 0 and max_swaps >= 0, 'Negative search budget')
    require(max_bin_tokens is None or max_bin_tokens > 0, 'Invalid bin capacity')
    metadata, baseline_records, baseline = summarize(samples, traces, schedule, topology, pad_multiplier)
    result = deepcopy(schedule)
    cells = result['cells']
    by_id = {s['sample_id']: i for i, s in enumerate(metadata)}
    lengths = {s['sample_id']: s['token_count'] for s in metadata}
    owners_map = topology['ep_groups'][0]['owners']
    owners = [owners_map[k] for k in sorted(owners_map, key=int)]
    topk = next(iter(traces.values())).shape[2]
    pad_cache = {}

    def bin_load(ids):
        count = sum(lengths[s] for s in ids)
        pad = (-count) % pad_multiplier
        if pad not in pad_cache:
            pad_cache[pad] = route_histograms(padding_routes(pad, len(owners), topk, len(owners[0])), owners)[:2]
        loads = []
        for j, name in enumerate(('H', 'U')):
            loads.append(pad_cache[pad][j] + sum(
                (baseline['sample_' + name][by_id[s]] + baseline['terminal_' + name][by_id[s]] for s in ids),
                np.zeros_like(pad_cache[pad][j])))
        return count, pad, loads

    keys = [(c['step'], c['group_id'], c['slot']) for c in cells]
    loads = [bin_load(c['sample_ids']) for c in cells]
    caps = [max_bin_tokens if max_bin_tokens is not None else n+p for n,p,_ in loads]
    require(all(n+p <= cap for (n,p,_),cap in zip(loads,caps)), 'Native bin exceeds capacity')
    totals = {k: [sum((loads[i][2][j] for i in range(len(cells)) if keys[i] == k),
                          np.zeros_like(loads[0][2][j])) for j in range(2)] for k in sorted(set(keys))}
    examined = 0
    swaps = []
    search_started = perf_counter()
    iterations = 0
    def pairs():
        for a in range(len(cells)):
            for b in range(a+1, len(cells)):
                if keys[a][0] != keys[b][0] or keys[a][2] == keys[b][2]: continue
                for ia in range(len(cells[a]['sample_ids'])):
                    for ib in range(len(cells[b]['sample_ids'])):
                        yield a,b,ia,ib
    # Both modes use seeded candidate order; sampled-best ties use stable indices.
    while examined < max_candidates and len(swaps) < max_swaps and iterations < max_iterations and perf_counter() < deadline:
        iterations += 1
        accepted = False
        proposals = pairs()
        best = None
        if mode in ('random', 'sampled_best'):
            proposals = list(proposals)
            rng.shuffle(proposals)
            if mode == "sampled_best": proposals = proposals[:candidate_batch_size]
        for a,b,ia,ib in proposals:
            if examined >= max_candidates or perf_counter() >= deadline: break
            sa,sb = cells[a]['sample_ids'][ia],cells[b]['sample_ids'][ib]
            examined += 1
            aa, bb = list(cells[a]['sample_ids']), list(cells[b]['sample_ids'])
            aa[ia], bb[ib] = sb, sa
            la, lb = bin_load(aa), bin_load(bb)
            if la[0]+la[1] > caps[a] or lb[0]+lb[1] > caps[b]: continue
            if la[1]+lb[1] > loads[a][1]+loads[b][1]: continue
            updated = {}
            for idx, new in ((a,la),(b,lb)):
                updated[keys[idx]] = [totals[keys[idx]][j] - loads[idx][2][j] + new[2][j] for j in range(2)]
            old_e = sum(excess(totals[k][0]) for k in updated)
            new_e = sum(excess(v[0]) for v in updated.values())
            if mode != "random" and new_e >= old_e: continue
            if mode == "sampled_best":
                order = (new_e-old_e, a,b,ia,ib)
                if best is None or order < best[0]:
                    best = (order, a,b,sa,sb,aa,bb,la,lb,updated,old_e,new_e)
                continue
            swaps.append({'cells': [a,b], 'samples': [sa,sb], 'old_affected_E':old_e,
                          'new_affected_E':new_e, 'padding_delta':la[1]+lb[1]-loads[a][1]-loads[b][1]})
            cells[a]['sample_ids'], cells[b]['sample_ids'] = aa, bb
            loads[a], loads[b] = la, lb
            totals.update(updated)
            accepted = True
            break
        if best is not None:
            _,a,b,sa,sb,aa,bb,la,lb,updated,old_e,new_e = best
            swaps.append({'cells':[a,b], 'samples':[sa,sb], 'old_affected_E':old_e,
                          'new_affected_E':new_e, 'padding_delta':la[1]+lb[1]-loads[a][1]-loads[b][1]})
            cells[a]['sample_ids'], cells[b]['sample_ids'] = aa,bb
            loads[a],loads[b] = la,lb
            totals.update(updated)
            accepted = True
        if not accepted and mode != 'sampled_best': break
    search_finished = perf_counter()
    stop_reason = ('wall_budget' if search_finished >= deadline else
                   'swap_budget' if len(swaps) >= max_swaps else
                   'candidate_budget' if examined >= max_candidates else
                   'iteration_budget' if iterations >= max_iterations else 'no_acceptable_pair')
    _, records, arrays = summarize(samples, traces, result, topology, pad_multiplier)
    # Independent full reconstruction checks the incremental candidate accounting.
    for j,name in enumerate(('H','U')):
        np.testing.assert_array_equal(arrays['collective_'+name], np.stack([totals[k][j] for k in sorted(totals)]))
    def metrics(records, arrays):
        return {'E': sum(r['EP_excess'] for r in records),
                'padding_tokens': sum(r['layout_padding_tokens'] for r in records),
                'collectives': [{**r, 'U_excess':excess(arrays['collective_U'][i])} for i,r in enumerate(records)]}
    report = {'objective':'sum_group_slot_layer(max_rank H - mean_rank H), including terminal and layout padding',
              'scope':'offline candidate only; H is a compute proxy; U counts token/destination pairs, not bytes or time',
              'mode':mode, 'seed':seed, 'candidate_batch_size':candidate_batch_size, 'iterations':iterations, 'stop_reason':stop_reason,
              'tie_break':'input cell order, then current sample position; random uses seeded pair permutation; sampled_best ties use gain then cell/sample indices',
              'determinism':'fixed count budgets are reproducible; wall-clock cutoff can change the stopping point',
              'cpu_wall_ms':{'prepare':(search_started-started)*1000, 'search':(search_finished-search_started)*1000,
                             'validation_and_report':(perf_counter()-search_finished)*1000, 'total':(perf_counter()-started)*1000},
              'candidates_examined':examined, 'swaps':swaps, 'bin_capacities':caps,
              'before':metrics(baseline_records,baseline), 'after':metrics(records,arrays),
              'runtime_intervention':'not_run', 'correctness_gate':'not_run', 'timing_gate':'not_run'}
    return result, report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--pad-multiplier',type=int,required=True)
    parser.add_argument('--max-candidates',type=int,default=10000)
    parser.add_argument('--max-swaps',type=int,default=100)
    parser.add_argument('--max-bin-tokens',type=int)
    parser.add_argument('--mode', choices=['random','sampled_best'], default='sampled_best')
    parser.add_argument('--candidate-batch-size',type=int,default=25)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--max-iterations',type=int,default=100)
    parser.add_argument('--max-wall-ms',type=float)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output already exists')
    paths = [args.input/n for n in ('samples.jsonl','schedule.json','topology.json')]
    samples = [json.loads(l) for l in paths[0].read_text().splitlines() if l.strip()]
    trace_paths = {s['sample_id']:args.input/s['route_path'] for s in samples}
    traces = {k:np.load(p,allow_pickle=False) for k,p in trace_paths.items()}
    schedule, report, arrays = refine(samples,traces,json.loads(paths[1].read_text()),json.loads(paths[2].read_text()),
        args.pad_multiplier,max_candidates=args.max_candidates,max_swaps=args.max_swaps,max_bin_tokens=args.max_bin_tokens,mode=args.mode,seed=args.seed,
        max_iterations=args.max_iterations,max_wall_ms=args.max_wall_ms,candidate_batch_size=args.candidate_batch_size)
    report['sha256'] = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in paths+list(trace_paths.values())+[Path(__file__),Path(__file__).with_name('profile_data.py')]}
    report['settings'] = {'pad_multiplier':args.pad_multiplier,'max_candidates':args.max_candidates,
                          'max_swaps':args.max_swaps,'max_bin_tokens':args.max_bin_tokens,
                          'mode':args.mode,'seed':args.seed,'max_iterations':args.max_iterations,'max_wall_ms':args.max_wall_ms}
    args.output.mkdir(parents=True)
    (args.output/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
    (args.output/'exchange.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'route_loads.npz',**arrays)
    print(json.dumps({k:report[k] for k in ('candidates_examined','swaps','timing_gate')}))


if __name__ == '__main__':
    main()
