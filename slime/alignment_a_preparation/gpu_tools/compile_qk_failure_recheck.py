"""Fresh-process direct/graph replay of first observed failures; no kernel edits."""
import argparse
import json
from pathlib import Path
import sys
import torch

PROJECT = Path('/public/home/yingxy2024/moe-consistency')
sys.path.insert(0, str(PROJECT / 'slime/alignment_a_preparation/gpu_tools'))
sys.path.insert(0, str(PROJECT / 'experiments/sglang-main-sm80/python'))
from compile_qk_layout_probe import restore, pair_compare, math_qk, make_graph
from sglang.srt.batch_invariant_ops.batch_invariant_ops import rms_norm_batch_invariant


def original(q, k, qw, kw, h, qe, ke):
    return (rms_norm_batch_invariant(q.reshape(-1, h), qw, qe).view(q.shape),
            rms_norm_batch_invariant(k.reshape(-1, h), kw, ke).view(k.shape))


def details(actual, expected):
    result = {}
    for label, a, b in zip(('q', 'k'), actual, expected):
        ai, bi = a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
        indices = (ai != bi).nonzero().cpu().tolist()
        result[label] = [dict(index=idx, actual=float(a[tuple(idx)].item()),
                             reference=float(b[tuple(idx)].item()),
                             actual_bits=int(ai[tuple(idx)].item()),
                             reference_bits=int(bi[tuple(idx)].item())) for idx in indices]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    checks = json.loads((args.run_dir / 'correctness.json').read_text())
    functions = dict(compile_original=torch.compile(original, fullgraph=True, dynamic=False),
                     compile_math=torch.compile(math_qk, fullgraph=True, dynamic=False))
    rows = []
    for name, function in functions.items():
        first = next(row for row in checks if not row['candidates'][name]['bit_equal'])
        data = torch.load(first['source'], map_location='cpu', weights_only=False)
        inputs = restore(data, first['batch'])
        before = tuple(t.clone() for t in inputs[:4])
        reference = tuple(t.clone() for t in original(*inputs))
        captured = tuple(data[key][:first['batch']].cuda() for key in ('q_output', 'k_output'))
        assert pair_compare(reference, captured)['bit_equal']
        direct = tuple(t.clone() for t in function(*inputs))
        for _ in range(10):
            function(*inputs)
        repeat = tuple(t.clone() for t in function(*inputs))
        graph, outputs = make_graph(lambda: function(*inputs), 1)
        graph.replay()
        torch.cuda.synchronize()
        graph_output = tuple(t.clone() for t in outputs[-1])
        row = dict(candidate=name, source=first['source'], rank=first['rank'],
            layer=first['layer'], batch=first['batch'],
            direct_vs_reference=pair_compare(direct, reference),
            graph_vs_reference=pair_compare(graph_output, reference),
            graph_vs_direct=pair_compare(graph_output, direct),
            repeated_direct=pair_compare(repeat, direct),
            inputs_unchanged=all(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
                for a, b in zip(inputs[:4], before)),
            direct_differences=details(direct, reference),
            graph_differences=details(graph_output, reference))
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
