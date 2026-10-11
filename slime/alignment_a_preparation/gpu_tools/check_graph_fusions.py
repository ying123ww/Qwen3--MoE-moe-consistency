"""Check changing inputs and weights across native graph fusion replays."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from sglang.srt.batch_invariant_ops import fused_rmsnorm_candidate as rms
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import rms_norm_batch_invariant
    from residual_rms_candidate import reference
    import qk_norm_prototype as qk
    rows = []
    elements = 0
    for m in (1, 2, 4):
        for weight_dtype in (torch.bfloat16, torch.float32):
            x = torch.zeros(m, 2048, dtype=torch.bfloat16, device='cuda')
            residual = torch.zeros_like(x)
            weight = torch.ones(2048, dtype=weight_dtype, device='cuda')
            packed = torch.zeros(m, 2560, dtype=torch.bfloat16, device='cuda')
            q, k = packed[:, :2048], packed[:, 2048:2304]
            qw = torch.ones(128, dtype=weight_dtype, device='cuda')
            kw = torch.ones_like(qw)
            for _ in range(2):
                rms._candidate(x, residual, weight, allow_cuda_graph=True)
                qk.forward(q, k, qw, kw, 128, 1e-6, 1e-6)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result_rms = rms._candidate(x, residual, weight, allow_cuda_graph=True)
                result_qk = qk.forward(q, k, qw, kw, 128, 1e-6, 1e-6)
            for seed, scale in enumerate((0., .01, 1., 10., 1.)):
                torch.manual_seed(20261011 + seed)
                for target in (x, residual, packed):
                    target.copy_((torch.randn_like(target.float()) * scale).to(target.dtype))
                x[:, 0] = -0.
                residual[:, 0] = -0.
                residual[:, 1] = -x[:, 1]
                for target in (weight, qw, kw):
                    target.copy_(torch.randn_like(target))
                expected_rms = reference(x, residual, weight)
                expected_qk = tuple(rms_norm_batch_invariant(v.reshape(-1,128), w, 1e-6).reshape(v.shape)
                                    for v, w in ((q,qw),(k,kw)))
                graph.replay()
                torch.cuda.synchronize()
                comparisons = [int((actual.view(torch.int16) != expected.view(torch.int16)).sum().item())
                    for actual, expected in zip((*result_rms,*result_qk),(*expected_rms,*expected_qk))]
                elements += sum(t.numel() for t in (*result_rms,*result_qk))
                rows.append(dict(m=m, weight_dtype=str(weight_dtype), seed=seed, scale=scale,
                                 mismatches=comparisons))
                if any(comparisons):
                    raise RuntimeError('Graph fusion bit mismatch: ' + json.dumps(rows[-1]))
    args.output.write_text(json.dumps(dict(passed=True, graph_replays=len(rows),
        compared_elements=elements, rows=rows), indent=2)+'\n')
    print(json.dumps(dict(passed=True, graph_replays=len(rows), compared_elements=elements)), flush=True)


if __name__ == '__main__':
    main()
