"""Replay real residual RMSNorm captures against original, manual and compile kernels.

Run with the experiment SGLang python directory on PYTHONPATH. This is an
operator replay, not an end-to-end model benchmark. M4 duplicates captured M2
rows; provenance distinguishes this from independently captured M4 activations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import time
import traceback

import torch

PROJECT = Path(__file__).resolve().parents[3]
PREP = PROJECT / "slime/alignment_a_preparation"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def compare(actual, expected):
    rows = []
    for a, b in zip(actual, expected, strict=True):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Output shape/dtype changed")
        av, bv = a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
        mismatch = av != bv
        locations = mismatch.nonzero()[:8]
        rows.append(dict(shape=list(a.shape), dtype=str(a.dtype), elements=a.numel(),
                         mismatched_elements=int(mismatch.sum().item()),
                         max_abs_error=float((a.float()-b.float()).abs().max().item()),
                         nonfinite_actual=int((~torch.isfinite(a)).sum().item()),
                         nonfinite_expected=int((~torch.isfinite(b)).sum().item()),
                         first_mismatch_indices=locations.cpu().tolist()))
    return dict(exact=all(r["mismatched_elements"] == 0 for r in rows), outputs=rows)


def math_expression(x, residual, weight, eps):
    value = x.float() + residual.float()
    visible_residual = value.to(x.dtype)
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    return ((value * torch.rsqrt(variance + eps)) * weight).to(x.dtype), visible_residual


def measure_direct(fn, calls):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        output = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / calls


def measure_graph(graph, replays, calls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / (calls * replays)


def capture(fn, calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Keep every output alive, with the same allocation lifetime for all candidates.
        outputs = [fn() for _ in range(calls)]
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


def kernel_trace(fn, path):
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(path))
        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        return dict(count=len(events), names=[e.name for e in events], trace=str(path))
    except Exception as exc:
        return dict(error=repr(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=20)
    args = parser.parse_args()
    if args.rounds < 5 or args.calls < 1 or args.graph_replays < 1:
        parser.error("Require rounds >= 5, calls >= 1 and graph-replays >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", "0")
    os.environ.setdefault("SLIME_QWEN3_RESIDUAL_RMS_MODE", "off")
    from sglang.srt.batch_invariant_ops import fused_rmsnorm_candidate as manual
    from sglang.srt.batch_invariant_ops import batch_invariant_ops as original
    from torch._inductor.utils import run_and_get_code
    import triton

    # Explicit mean_dim reproduces the project's enabled batch-invariant dispatch.
    # All remaining operations exactly follow RMSNorm.forward_native's eligible branch.
    def reference(x, residual, weight, eps):
        value = x.float() + residual.float()
        visible_residual = value.to(x.dtype)
        variance = original.mean_dim(value.pow(2), dim=-1, keepdim=True)
        return ((value * torch.rsqrt(variance + eps)) * weight).to(x.dtype), visible_residual

    files = args.inputs or sorted((PREP / "gpu_runs/rms-repro-hgx05-20261008-01/exact-shadow/audit").glob("failure-*.pt"))
    if not files:
        raise RuntimeError("No real model residual captures found")
    torch.cuda.set_device(0)
    result = dict(status="running", host=socket.gethostname(), pid=os.getpid(),
                  environment=dict(torch=torch.__version__, triton=triton.__version__,
                                   python=platform.python_version(), cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0),
                                   capability=list(torch.cuda.get_device_capability(0))),
                  source_sha256={str(p): sha(p) for p in [Path(__file__), Path(manual.__file__), Path(original.__file__),
                      PROJECT / "experiments/sglang-main-sm80/python/sglang/srt/layers/layernorm.py"]},
                  compile_options=dict(backend="inductor", fullgraph=True, dynamic=False),
                  semantics=dict(original="forward_native eligible branch with explicit batch-invariant mean_dim",
                      manual="production _candidate(..., allow_cuda_graph=True)",
                      compile_original="torch.compile of original explicit mean_dim path; fullgraph errors retained",
                      compile_math="torch.compile of same arithmetic using ordinary torch.mean; reduction tree may differ",
                      residual="FP32 x+residual, BF16 visible residual; norm uses unrounded FP32 sum",
                      norm="FP32 squared values/reduction/rsqrt/weight multiply, then BF16 output"),
                  timing_protocol=dict(rounds=args.rounds, calls=args.calls, graph_replays=args.graph_replays,
                      order="cyclic rotation each round", direct="synchronized host wall time per call",
                      graph="CUDA event time per operation, 100-call graph by default, repeated 20 times",
                      exclusions="Compilation, profiling, warmup and graph capture excluded",
                      graph_outputs="All outputs retained during capture for every candidate"),
                  scope="Finite real-activation operator replay. M1 prefix; M2 original; M4 cyclic repeated real rows. No full model run.",
                  inputs=[], cases=[], batch_prefix=[], errors=[])
    save(args.output_dir / "results.json", result)
    try:
        for file_index, file in enumerate(files):
            data = torch.load(file, map_location="cpu", weights_only=True)
            if tuple(data["x"].shape) != (2, 2048) or data["x"].dtype != torch.bfloat16:
                raise ValueError("Expected captured M2 H2048 BF16 tensors")
            source = dict(path=str(file.resolve()), sha256=sha(file), context=data.get("context"),
                          x_shape=list(data["x"].shape), x_dtype=str(data["x"].dtype),
                          residual_dtype=str(data["residual"].dtype), weight_dtype=str(data["weight"].dtype), eps=data["eps"])
            result["inputs"].append(source)
            previous = {}
            for m in (1, 2, 4):
                indices = torch.arange(m) % data["x"].shape[0]
                x = data["x"][indices].contiguous().cuda()
                residual = data["residual"][indices].contiguous().cuda()
                weight = data["weight"].contiguous().cuda()
                eps = data["eps"]
                call_args = (x, residual, weight, eps)
                expected = reference(*call_args)
                captured_expected = tuple(v[indices].cuda() for v in data["reference"])
                capture_check = compare(expected, captured_expected)
                if not capture_check["exact"]:
                    raise RuntimeError("Current reference disagrees with historical real-model output: " + str(file))
                case = dict(input_index=file_index, shape=list(x.shape), row_indices=indices.tolist(),
                            provenance="original_capture" if m == 2 else "captured_prefix" if m == 1 else "repeated_captured_rows",
                            captured_reference_check=capture_check, candidates={})
                result["cases"].append(case)
                fns = dict(original=lambda: reference(*call_args),
                           manual=lambda: manual._candidate(*call_args, allow_cuda_graph=True))
                for name, function in (("compile_original", reference), ("compile_math", math_expression)):
                    try:
                        compiled = torch.compile(function, backend="inductor", fullgraph=True, dynamic=False)
                        _, codes = run_and_get_code(compiled, *call_args)
                        paths = []
                        for code_index, code in enumerate(codes):
                            target = args.output_dir / f"source{file_index}-m{m}-{name}-{code_index}.py"
                            target.write_text(code)
                            paths.append(dict(path=str(target), sha256=sha(target),
                                              triton_definitions=code.count("async_compile.triton(")))
                        case["candidates"][name] = dict(generated_code=paths)
                        fns[name] = lambda compiled=compiled: compiled(*call_args)
                    except Exception:
                        case["candidates"][name] = dict(eligible=False, error=traceback.format_exc())
                graphs = {}
                graph_outputs = {}
                # run_and_get_code resets Dynamo, so warm every candidate only after all code collection.
                for name in list(fns):
                    record = case["candidates"].setdefault(name, {})
                    try:
                        for _ in range(30):
                            actual = fns[name]()
                        actual = fns[name]()
                        record["comparison"] = compare(actual, expected)
                        record["eligible"] = record["comparison"]["exact"]
                        cpu_actual = tuple(t.detach().cpu() for t in actual)
                        for smaller in (1, 2):
                            if (name, smaller) in previous:
                                check = compare(tuple(t[:smaller] for t in cpu_actual), previous[name, smaller])
                                result["batch_prefix"].append(dict(input_index=file_index, candidate=name, smaller=smaller, larger=m, comparison=check))
                                record["eligible"] &= check["exact"]
                        previous[name, m] = cpu_actual
                        record["kernels"] = kernel_trace(fns[name], args.output_dir / f"source{file_index}-m{m}-{name}-trace.json")
                        graph, outputs = capture(fns[name], args.calls)
                        graphs[name], graph_outputs[name] = graph, outputs
                        record["graph_comparison"] = compare(outputs[-1], expected)
                        record["eligible"] &= record["graph_comparison"]["exact"]
                        record["samples"] = []
                    except Exception:
                        record.update(eligible=False, error=traceback.format_exc())
                        fns.pop(name)
                names = list(fns)
                for repeat in range(args.rounds):
                    order = names[repeat % len(names):] + names[:repeat % len(names)]
                    for name in order:
                        sample = dict(round=repeat, order=order,
                                      direct_us=measure_direct(fns[name], args.calls),
                                      graph_us=measure_graph(graphs[name], args.graph_replays, args.calls))
                        case["candidates"][name]["samples"].append(sample)
                for name in names:
                    record = case["candidates"][name]
                    record["median_us"] = {key: statistics.median(s[key] for s in record["samples"])
                                            for key in ("direct_us", "graph_us")}
                save(args.output_dir / "results.json", result)
                print(json.dumps(dict(input_index=file_index, m=m, candidates={name: dict(eligible=r["eligible"], median_us=r.get("median_us"), comparison=r.get("comparison")) for name, r in case["candidates"].items()})), flush=True)
                del graphs, graph_outputs
        result["status"] = "complete"
        result["all_original_manual_exact"] = all(case["candidates"][name]["eligible"] for case in result["cases"] for name in ("original", "manual"))
    except BaseException:
        result["status"] = "failed"
        result["errors"].append(traceback.format_exc())
        raise
    finally:
        save(args.output_dir / "results.json", result)


if __name__ == "__main__":
    main()
