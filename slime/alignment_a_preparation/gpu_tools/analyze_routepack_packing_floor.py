"""CPU-only bounds and expert-placement counterfactual on saved RoutePack counts.

Usage: python analyze_routepack_packing_floor.py PLANS_DIRECTORY NEW_OUTPUT_JSON
No training state, schedules, or GPU processes are modified.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def metrics(h):
    aggregate = h.sum(axis=0)
    excess = float((h.max(axis=-1) - h.mean(axis=-1)).sum())
    floor = float((aggregate.max(axis=-1) - aggregate.mean(axis=-1)).sum())
    peak = int(h.max(axis=-1).sum())
    return {
        "assignments": int(h.sum()),
        "sum_collective_layer_rank_peaks": peak,
        "excess": excess,
        "fixed_owner_excess_lower_bound": floor,
        "lower_bound_fraction_of_excess": floor / excess if excess else None,
        "optimistic_remaining_peak_reduction_fraction": (excess - floor) / peak,
    }


def project(per_expert, owners, ranks):
    h = np.zeros((*per_expert.shape[:2], ranks), dtype=np.int64)
    for layer in range(per_expert.shape[1]):
        for rank in range(ranks):
            h[:, layer, rank] = per_expert[:, layer, owners[layer] == rank].sum(axis=-1)
    np.testing.assert_array_equal(h.sum(axis=-1), per_expert.sum(axis=-1))
    return h


def main():
    root, output = map(Path, sys.argv[1:])
    topology = json.loads((root / "input/topology.json").read_text())
    groups = topology["ep_groups"]
    ranks = len(groups[0]["members"])
    assert all(g["owners"] == groups[0]["owners"] for g in groups)
    layer_ids = sorted(groups[0]["owners"], key=int)
    original_owners = np.array([groups[0]["owners"][l] for l in layer_ids])
    schedule = json.loads((root / "native/schedule.json").read_text())
    assert len({c["step"] for c in schedule["cells"]}) == 1
    inputs = {}
    arrays = {}
    for method in ("native", "routepack"):
        path = root / method / "route_loads.npz"
        inputs[method] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with np.load(path, allow_pickle=False) as z:
            h = z["collective_H"] - z["layout_padding_H"]
            pe = z["collective_per_expert"] - z["layout_padding_per_expert"]
            np.testing.assert_array_equal(h.sum(0), (z["sample_H"] + z["terminal_H"]).sum(0))
            np.testing.assert_array_equal(pe.sum(0), (z["sample_per_expert"] + z["terminal_per_expert"]).sum(0))
        np.testing.assert_array_equal(project(pe, original_owners, ranks), h)
        arrays[method] = (h, pe)
    np.testing.assert_array_equal(arrays["native"][1].sum(0), arrays["routepack"][1].sum(0))

    demand = arrays["native"][1].sum(0)
    owners = np.zeros_like(original_owners)
    for layer, weights in enumerate(demand):
        capacity = np.bincount(original_owners[layer], minlength=ranks)
        assigned = np.zeros(ranks, dtype=np.int64)
        loads = np.zeros(ranks, dtype=np.int64)
        for expert in np.argsort(-weights, kind="stable"):
            eligible = np.flatnonzero(assigned < capacity)
            rank = eligible[np.argmin(loads[eligible])]
            owners[layer, expert] = rank
            loads[rank] += weights[expert]
            assigned[rank] += 1
        np.testing.assert_array_equal(assigned, capacity)

    results = {}
    for method, (h, pe) in arrays.items():
        placed = project(pe, owners, ranks)
        results[method] = {"existing_placement": metrics(h), "greedy_placement_counterfactual": metrics(placed)}
        results[method]["counterfactual_peak_reduction_fraction"] = 1 - placed.max(-1).sum() / h.max(-1).sum()
    result = {
        "scope": "CPU count analysis of one saved 128-sample optimizer step; layout padding excluded; fixed synthetic terminal assignments retained",
        "bound": "sum_c max_r H[c,l,r] >= max_r sum_c H[c,l,r], with c spanning EP groups and slots having the same owner map",
        "interpretation": "The fixed-owner bound relaxes sample indivisibility and capacity; it bounds this assignment proxy, not step time. Padding-dependent savings are excluded.",
        "placement_method": "Per-layer descending aggregate demand, greedily assign to least-loaded rank with a free expert slot; same expert count on every rank as original; one common mapping across EP replicas",
        "limitations": [
            "No GPU placement implemented or timed; no migration, communication, optimizer-state, padding, or numerical costs included.",
            "Same-batch routing foresight diagnostic, not evidence for reuse on future batches.",
            "Aggregating collective maxima is not the training critical path.",
        ],
        "inputs": inputs,
        "results": results,
        "candidate_owner_by_layer": owners.tolist(),
    }
    with output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"output": str(output.resolve()), "results": results}, indent=2))


if __name__ == "__main__":
    main()
