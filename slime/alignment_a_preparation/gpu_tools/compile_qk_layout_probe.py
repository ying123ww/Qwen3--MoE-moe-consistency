"""Compare captured model Q/K layout+norm, hand fusion, and two compile routes.

This is an operator replay, not an end-to-end model latency measurement. The
driver owns GPU scheduling and keepalive transitions. Run with TORCH_LOGS=output_code
to retain the actual generated kernels alongside the JSON and profiler traces.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import socket
import statistics
import sys
import time
import traceback

import torch
import triton

PREP = Path(__file__).resolve().parents[1]
PROJECT = PREP.parent.parent
DEFAULT_EVIDENCE = PREP / 'gpu_runs/remaining-profile-hgx03-qk-20261011-01/profile-candidate/qk-evidence'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(x):
    return dict(shape=list(x.shape), dtype=str(x.dtype), stride=list(x.stride()),
                contiguous=x.is_contiguous(), storage_offset=x.storage_offset())


def compare(a, b):
    compatible = a.shape == b.shape and a.dtype == b.dtype
    result = dict(shape_dtype_equal=compatible, actual=describe(a), expected=describe(b))
    if not compatible:
        return dict(result, bit_equal=False, layout_equal=False)
    # Integer views compare all bits, including signed zero and NaN encodings.
    integer_dtype = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    different = a.contiguous().view(integer_dtype) != b.contiguous().view(integer_dtype)
    count = int(different.sum().item())
    maximum = float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.
    return dict(result, elements=a.numel(), different_elements=count,
                bit_equal=count == 0, max_abs_error=maximum,
                layout_equal=a.stride() == b.stride() and a.is_contiguous() == b.is_contiguous())


def pair_compare(actual, expected):
    comparisons = {name: compare(a, b) for name, a, b in zip(('q', 'k'), actual, expected)}
    return dict(outputs=comparisons,
                bit_equal=all(v['bit_equal'] for v in comparisons.values()),
                layout_equal=all(v['layout_equal'] for v in comparisons.values()))


def restore(data, batch, device='cuda'):
    # Restore the original packed-QKV storage offsets and row strides: .cuda()
    # alone would silently replace the layout-copy workload with contiguous data.
    specs = [data['metadata'][key] for key in ('q', 'k')]
    capacity = max(s['storage_offset'] + (batch - 1) * s['stride'][0]
                   + (s['shape'][1] - 1) * s['stride'][1] + 1 for s in specs)
    packed = torch.empty(capacity, dtype=data['q'].dtype, device=device)
    inputs = []
    for key, spec in zip(('q', 'k'), specs):
        x = packed.as_strided((batch, spec['shape'][1]), tuple(spec['stride']), spec['storage_offset'])
        x.copy_(data[key][:batch])
        inputs.append(x)
    return (*inputs, data['q_weight'].to(device), data['k_weight'].to(device),
            data['metadata']['head_dim'], data['q_eps'], data['k_eps'])


def validate_capture_paths(evidence, files):
    """Require recorded runtime dispatch, rather than inferring it from names."""
    provenance = {}
    for rank in sorted({row[0] for row in files}):
        metadata_file = evidence / f'rank-{rank}-metadata.jsonl'
        rows = [json.loads(line) for line in metadata_file.read_text().splitlines() if line.strip()]
        paths = {(row['layer'], row['component']): row['implementation']
                 for row in rows if row.get('kind') == 'norm_path'}
        for _, layer, _, file in (item for item in files if item[0] == rank):
            for component in ('q_norm', 'k_norm'):
                if paths.get((layer, component)) != 'rms_norm_batch_invariant':
                    raise RuntimeError(f'Capture {file} has no verified rms_norm_batch_invariant '
                                       f'runtime path for {component}; use deterministic profile-candidate evidence')
        provenance[str(metadata_file)] = dict(sha256=sha(metadata_file),
            verified_implementation='rms_norm_batch_invariant', norm_path_entries=len(paths))
    return provenance


def math_qk(q, k, qw, kw, h, qe, ke):
    # Match the upstream Triton RMS formula, including its explicit sqrt then
    # reciprocal (not native RMSNorm's rsqrt). Inductor chooses its own reduction.
    def norm(x, weight, eps):
        xf = x.reshape(-1, h).float()
        mean = (xf * xf).mean(dim=-1, keepdim=True)
        inverse = 1.0 / torch.sqrt(mean + eps)
        return ((xf * inverse) * weight.float()).to(x.dtype).reshape(x.shape).contiguous()
    return norm(q, qw, qe), norm(k, kw, ke)


def direct_measure(fn, calls):
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    begin.record()
    for _ in range(calls):
        output = fn()
    end.record()
    end.synchronize()
    return dict(wall_us=(time.perf_counter() - start) * 1e6 / calls,
                gpu_interval_us=begin.elapsed_time(end) * 1e3 / calls)


def make_graph(fn, calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    # Holding every output avoids allocator/address reuse adding different
    # artificial output-dependency chains to different implementations.
    outputs = []
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            outputs.append(fn())
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


def graph_measure(graph, calls, replays):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    denominator = calls * replays
    return dict(wall_us=(time.perf_counter() - start) * 1e6 / denominator,
                gpu_interval_us=begin.elapsed_time(end) * 1e3 / denominator)


def profile_one(fn, path):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())['traceEvents']
    kernels = [dict(name=e['name'], duration_us=e.get('dur')) for e in events if e.get('cat') == 'kernel']
    copies = [dict(name=e['name'], category=e.get('cat'), duration_us=e.get('dur'))
              for e in events if e.get('cat') in ('gpu_memcpy', 'gpu_memset')]
    return dict(trace=str(path), kernel_count=len(kernels), kernels=kernels, copy_events=copies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--ranks', type=int, nargs='*', default=None)
    parser.add_argument('--layers', type=int, nargs='*', default=None)
    parser.add_argument('--timing-layers', type=int, nargs='+', default=[0, 24, 47])
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--calls', type=int, default=100)
    parser.add_argument('--graph-calls', type=int, default=100)
    parser.add_argument('--graph-replays', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=20)
    args = parser.parse_args()
    if args.rounds < 5 or min(args.calls, args.graph_calls, args.graph_replays, args.warmup) < 1:
        parser.error('Use at least five rounds and positive iteration counts')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.set_grad_enabled(False)
    sys.path.insert(0, str(PROJECT / 'experiments/sglang-main-sm80/python'))
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import rms_norm_batch_invariant
    import qk_norm_prototype

    def original(q, k, qw, kw, h, qe, ke):
        return (rms_norm_batch_invariant(q.reshape(-1, h), qw, qe).view(q.shape),
                rms_norm_batch_invariant(k.reshape(-1, h), kw, ke).view(k.shape))

    def hand(q, k, qw, kw, h, qe, ke):
        return qk_norm_prototype.forward(q, k, qw, kw, h, qe, ke, 'joint')

    functions = dict(original=original, hand_fused=hand,
                     compile_original=torch.compile(original, fullgraph=True, dynamic=False),
                     compile_math=torch.compile(math_qk, fullgraph=True, dynamic=False))
    module_path = Path(inspect.getsourcefile(rms_norm_batch_invariant))
    sources = [Path(__file__), Path(qk_norm_prototype.__file__), module_path,
               PROJECT / 'experiments/sglang-main-sm80/python/sglang/srt/models/utils.py',
               PROJECT / 'experiments/sglang-main-sm80/python/sglang/srt/layers/layernorm.py']
    manifest = dict(task='Q/K RMSNorm + packed layout copy operator replay',
        environment=dict(host=socket.gethostname(), pid=os.getpid(), python=sys.version,
            platform=platform.platform(), torch=torch.__version__, triton=triton.__version__,
            cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            TORCH_LOGS=os.environ.get('TORCH_LOGS'),
            TORCHINDUCTOR_CACHE_DIR=os.environ.get('TORCHINDUCTOR_CACHE_DIR')),
        source_sha256={str(p): sha(p) for p in sources}, arguments=vars(args).copy(),
        reference='Extracted actual apply_qk_norm deterministic batch-invariant branch: '
                  'q/k.reshape(-1,head_dim), upstream rms_norm_batch_invariant, view(original shape). '
                  'Must match each captured original model output; dispatch overhead is excluded.',
        compile_original='torch.compile(fullgraph=True,dynamic=False) on the original callable including upstream Triton RMSNorm; does not promise fusion across that kernel.',
        compile_math='torch.compile(fullgraph=True,dynamic=False) on pure PyTorch sqrt/reciprocal RMS formula; '
                     'replaces upstream Triton reduction and may change bits.',
        deterministic_mean_note='This Q/K branch uses upstream rms_norm_batch_invariant Triton RMSNorm directly; '
                                'the deterministic mean custom-op path used by residual native RMSNorm is not invoked.',
        evidence_selection='profile-candidate is the deterministic model capture (candidate refers to the earlier '
                           'aligned model, not the hand-fused Q/K kernel). profile-baseline uses another normalization '
                           'path and is rejected by runtime norm_path metadata validation.',
        timing='Microseconds per invocation. Compile and warmup excluded. Direct includes allocation/launch and Python overhead; '
               'CUDA Graph includes graph-calls independent invocations per replay. Orders rotate each round. '
               'Same persistent real inputs reused across calls, outputs retained inside graph. '
               'This is an operator replay, not full-model speedup.')
    manifest['arguments'] = {k: str(v) if isinstance(v, Path) else v for k, v in manifest['arguments'].items()}
    save(args.output_dir / 'manifest.json', manifest)
    files = []
    for file in args.evidence.glob('rank-*-layer-*-m-*.pt'):
        rank, layer, batch = map(int, re.search(r'rank-(\d+)-layer-(\d+)-m-(\d+)\.pt$', file.name).groups())
        if (args.ranks is None or rank in args.ranks) and (args.layers is None or layer in args.layers):
            files.append((rank, layer, batch, file))
    files.sort()
    if not files:
        raise RuntimeError('No captured Q/K evidence found')
    manifest['capture_runtime_provenance'] = validate_capture_paths(args.evidence, files)
    save(args.output_dir / 'manifest.json', manifest)
    checks, timings, prefix_checks, errors = [], [], [], []
    disabled = set()
    compilation = {}
    aggregate = {name: dict(cases=0, elements=0, different_elements=0, bit_failures=0,
                           layout_failures=0, max_abs_error=0., prefix_failures=0)
                 for name in functions}
    first_rank = min(row[0] for row in files)
    for rank, layer, captured_batch, file in files:
        data = torch.load(file, map_location='cpu', weights_only=False)
        file_sha = sha(file)
        full_outputs = {}
        # Largest prefix first establishes the cross-batch comparison target.
        for batch in sorted((b for b in args.batches if b <= captured_batch), reverse=True):
            inputs = restore(data, batch)
            expected = (data['q_output'][:batch].cuda(), data['k_output'][:batch].cuda())
            reference = original(*inputs)
            captured_gate = pair_compare(reference, expected)
            if not captured_gate['bit_equal'] or not captured_gate['layout_equal']:
                save(args.output_dir / 'reference-failure.json', dict(file=str(file), batch=batch, gate=captured_gate))
                raise RuntimeError('Upstream replay differs from captured model output: ' + str(file))
            row = dict(source=str(file), source_sha256=file_sha, rank=rank, layer=layer,
                       captured_batch=captured_batch, batch=batch,
                       input_origin='unaltered real capture' if batch == captured_batch else 'row prefix of real captured batch',
                       inputs={name: describe(t) for name, t in zip(('q', 'k', 'q_weight', 'k_weight'), inputs[:4])},
                       head_dim=inputs[4], q_eps=inputs[5], k_eps=inputs[6],
                       captured_reference_gate=captured_gate, candidates={})
            callable_functions = {}
            for name, function in functions.items():
                if name in disabled:
                    row['candidates'][name] = dict(error='disabled after prior compile/runtime failure')
                    continue
                try:
                    started = time.perf_counter()
                    output = function(*inputs)
                    torch.cuda.synchronize()
                    compilation.setdefault(f'{name}/m={batch}', dict(first_call_s=time.perf_counter() - started))
                    gate = pair_compare(output, reference)
                    gate['eligible_bit_and_layout'] = gate['bit_equal'] and gate['layout_equal']
                    row['candidates'][name] = gate
                    total = aggregate[name]
                    total['cases'] += 1
                    total['bit_failures'] += not gate['bit_equal']
                    total['layout_failures'] += not gate['layout_equal']
                    for value in gate['outputs'].values():
                        total['elements'] += value.get('elements', 0)
                        total['different_elements'] += value.get('different_elements', 0)
                        total['max_abs_error'] = max(total['max_abs_error'], value.get('max_abs_error', 0.))
                    if name in full_outputs:
                        prior_batch, prior_output = full_outputs[name]
                        prefix_gate = pair_compare(output, tuple(t[:batch].contiguous() for t in prior_output))
                        prefix_checks.append(dict(source=str(file), candidate=name, batch=batch,
                            full_batch=prior_batch, comparison=prefix_gate))
                        total['prefix_failures'] += not prefix_gate['bit_equal']
                    else:
                        full_outputs[name] = (batch, tuple(t.clone() for t in output))
                    callable_functions[name] = lambda function=function, inputs=inputs: function(*inputs)
                except Exception as exc:
                    if name in ('original', 'hand_fused'):
                        raise
                    disabled.add(name)
                    failure = dict(candidate=name, source=str(file), batch=batch,
                                   error=str(exc), traceback=traceback.format_exc())
                    errors.append(failure)
                    row['candidates'][name] = failure
                    save(args.output_dir / 'candidate-errors.json', errors)
            checks.append(row)
            if rank == first_rank and layer in args.timing_layers:
                names = list(callable_functions)
                graphs, graph_errors, graph_gates, profiles = {}, {}, {}, {}
                for name, fn in callable_functions.items():
                    for _ in range(args.warmup):
                        fn()
                    torch.cuda.synchronize()
                    try:
                        profiles[name] = profile_one(fn, args.output_dir / f'trace-layer-{layer}-m-{batch}-{name}.json')
                    except Exception:
                        profiles[name] = dict(error=traceback.format_exc())
                    try:
                        graphs[name] = make_graph(fn, args.graph_calls)
                        graph_gate = pair_compare(graphs[name][1][-1], reference)
                        graph_gates[name] = dict(against_original=graph_gate,
                            against_same_candidate_direct=pair_compare(graphs[name][1][-1], fn()))
                        if not graph_gates[name]['against_same_candidate_direct']['bit_equal']:
                            raise RuntimeError('CUDA graph changed the same candidate output')
                    except Exception:
                        graphs.pop(name, None)
                        graph_errors[name] = traceback.format_exc()
                trials = {name: dict(direct=[], cuda_graph=[]) for name in names}
                orders = []
                for trial in range(args.rounds):
                    order = names[trial % len(names):] + names[:trial % len(names)]
                    orders.append(order)
                    for name in order:
                        trials[name]['direct'].append(direct_measure(callable_functions[name], args.calls))
                        if name in graphs:
                            trials[name]['cuda_graph'].append(graph_measure(graphs[name][0], args.graph_calls, args.graph_replays))
                medians = {name: {mode: {key: statistics.median(v[key] for v in values)
                                        for key in ('wall_us', 'gpu_interval_us')} if values else None
                                  for mode, values in modes.items()} for name, modes in trials.items()}
                timing = dict(source=str(file), rank=rank, layer=layer, batch=batch,
                              medians=medians, trials=trials, orders=orders, profiles=profiles,
                              graph_errors=graph_errors, graph_correctness=graph_gates,
                              eligible={name: row['candidates'][name]['eligible_bit_and_layout'] for name in names})
                timings.append(timing)
                save(args.output_dir / 'timings.json', timings)
                print(json.dumps(dict(event='timing', layer=layer, batch=batch, medians=medians)), flush=True)
                del graphs
        if layer % 8 == 0:
            print(json.dumps(dict(event='correctness_progress', rank=rank, layer=layer, cases=len(checks))), flush=True)
        save(args.output_dir / 'progress.json', dict(rank=rank, layer=layer, cases=len(checks), aggregate=aggregate))
    for name, totals in aggregate.items():
        totals['eligible_bit_layout_and_batch'] = (name not in disabled and totals['cases'] > 0 and
            totals['bit_failures'] == totals['layout_failures'] == totals['prefix_failures'] == 0)
    save(args.output_dir / 'correctness.json', checks)
    save(args.output_dir / 'cross-batch-prefix.json', prefix_checks)
    save(args.output_dir / 'summary.json', dict(complete=True, real_source_files=len(files),
        case_count=len(checks), aggregate=aggregate, errors=errors, first_calls=compilation,
        timing_cases=len(timings), prefix_comparisons=len(prefix_checks)))
    print(json.dumps(dict(event='complete', aggregate=aggregate)), flush=True)


if __name__ == '__main__':
    main()
