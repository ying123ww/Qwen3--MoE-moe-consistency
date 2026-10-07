#!/usr/bin/env python3
"""Attribute GPU transfers to optimizer CPU launch scopes using Kineto IDs.

Read-only, standard-library analysis; no CUDA or model import. Runtime durations
and GPU event durations are separate inclusive sums, never added together.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def aggregate(events):
    events = list(events)
    return {"count": len(events), "duration_s": sum(e.get("dur", 0) for e in events) / 1e6,
            "bytes": sum(e.get("args", {}).get("bytes", 0) for e in events)}


def analyze(events):
    spans = [e for e in events if e.get("cat") == "user_annotation" and e.get("name") == "P1/optimizer_step"]
    if len(spans) != 1:
        raise ValueError(f"Expected one CPU optimizer span, found {len(spans)}")
    span = spans[0]
    lo, hi = span["ts"], span["ts"] + span["dur"]
    cpu = [e for e in events if e.get("cat") in ("cpu_op", "cuda_runtime", "user_annotation")
           and e.get("pid") == span["pid"] and lo <= e.get("ts", -1) < hi]
    ops = {e.get("args", {}).get("External id"): e for e in cpu if e.get("cat") == "cpu_op"}
    runtime = {e.get("args", {}).get("correlation"): e for e in cpu if e.get("cat") == "cuda_runtime"}
    steps = defaultdict(list)
    for e in cpu:
        if e.get("name") == "Optimizer.step#StatelessAdam.step":
            steps[e["tid"]].append(e)
    for seq in steps.values():
        seq.sort(key=lambda e: e["ts"])
    starts = {tid: [e["ts"] for e in seq] for tid, seq in steps.items()}

    def adam_scope(runtime_event):
        seq = steps.get(runtime_event["tid"], [])
        idx = bisect_right(starts.get(runtime_event["tid"], []), runtime_event["ts"]) - 1
        return idx >= 0 and runtime_event["ts"] + runtime_event.get("dur", 0) <= seq[idx]["ts"] + seq[idx]["dur"]

    copies = defaultdict(list)
    launches = defaultdict(dict)
    linked_ops = defaultdict(Counter)
    adam_counts = Counter()
    bytes_hist = defaultdict(Counter)
    window_unlinked = Counter()
    examples = []
    for e in events:
        if e.get("cat") != "gpu_memcpy":
            continue
        args = e.get("args", {})
        r = runtime.get(args.get("correlation"))
        if r is None or r.get("name") != "cudaMemcpyAsync":
            if lo <= e.get("ts", -1) < hi:
                window_unlinked[e["name"]] += 1
            continue
        name = e["name"]
        copies[name].append(e)
        launches[name][args["correlation"]] = r
        op = ops.get(args.get("External id"))
        linked_ops[name][op["name"] if op else "unmatched"] += 1
        adam_counts[name] += int(adam_scope(r))
        bytes_hist[name][args.get("bytes", 0)] += 1
        if "Pageable -> Device" in name and len(examples) < 3:
            examples.append({"gpu": e, "runtime": r, "cpu_operator": op})
    runtime_groups = defaultdict(list)
    for e in cpu:
        if e.get("cat") == "cuda_runtime":
            runtime_groups[e["name"]].append(e)
    operator_groups = defaultdict(list)
    for e in cpu:
        if e.get("cat") == "cpu_op" and e["name"] in ("aten::addcdiv_", "aten::mul", "aten::abs", "aten::add_", "aten::copy_"):
            operator_groups[e["name"]].append(e)
    return {
        "optimizer_wall_s": span["dur"] / 1e6,
        "adam_scopes": aggregate(e for seq in steps.values() for e in seq),
        "copies_by_launch_scope": {name: {
            "gpu": aggregate(seq), "runtime_api": aggregate(launches[name].values()),
            "linked_cpu_operators": dict(linked_ops[name]), "launches_inside_adam_scope": adam_counts[name],
            "bytes_histogram": dict(sorted(bytes_hist[name].items())),
        } for name, seq in copies.items()},
        "window_memcpy_without_optimizer_launch": dict(window_unlinked),
        "cuda_runtime_inclusive": {name: aggregate(seq) for name, seq in sorted(runtime_groups.items())},
        "selected_cpu_operator_inclusive": {name: aggregate(seq) for name, seq in operator_groups.items()},
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = []
    for path in args.traces:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            events = json.load(f)["traceEvents"]
        result = analyze(events)
        result.update(trace=str(path.resolve()), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        results.append(result)
        print(path.name, result["optimizer_wall_s"], flush=True)
        del events
    output = {"method": "GPU memcpy -> CUDA runtime by correlation; CPU copy by External id. Include launches in CPU optimizer scope. CPU durations inclusive; GPU durations additive, not critical-path savings.",
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "results": results}
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
