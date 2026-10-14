"""Add + ReLU compile probe. CPU is the default; CUDA requires --device cuda.

Run: TORCH_LOGS=output_code python gpu_tools/compile_add_relu_cpu_probe.py
"""

import argparse
import statistics

import torch


def eager(x, residual):
    return torch.relu(x + residual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"torch={torch.__version__}; device={device}", flush=True)
    torch.manual_seed(1234)
    failures = 0
    cases = 0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        torch.compiler.reset()
        compiled = torch.compile(
            eager, backend="inductor", fullgraph=True, dynamic=True
        )
        for batch in (1, 2, 4, 8):
            for strided in (False, True):
                width = 4096 if strided else 2048
                x = torch.randn(batch, width, dtype=dtype, device=device)
                residual = torch.randn_like(x)
                if strided:
                    x, residual = x[:, ::2], residual[:, ::2]
                expected = eager(x, residual)
                actual = compiled(x, residual)
                a = expected.contiguous().view(torch.uint8)
                b = actual.contiguous().view(torch.uint8)
                mismatched_bytes = int((a != b).sum())
                cases += 1
                failures += mismatched_bytes != 0
                print(f"dtype={dtype} batch={batch} strided={strided} "
                      f"mismatched_bytes={mismatched_bytes}", flush=True)
    print(f"cases={cases} failed_cases={failures}", flush=True)
    if args.benchmark:
        if device.type != "cuda":
            raise ValueError("--benchmark requires --device cuda")
        if failures:
            raise RuntimeError("Skip benchmark: correctness check failed")
        benchmark()
    raise SystemExit(1 if failures else 0)


def benchmark():
    # Tiny operator timings, excluding compilation; not model throughput.
    torch.compiler.reset()
    compiled = torch.compile(eager, backend="inductor", fullgraph=True)
    for batch in (1, 2, 4, 8):
        x = torch.randn(batch, 2048, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn_like(x)
        for _ in range(20):
            eager(x, residual)
            compiled(x, residual)
        torch.cuda.synchronize()
        graphs = []
        outputs = []
        for fn in (eager, compiled):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn(x, residual)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = fn(x, residual)
            graphs.append(graph)
            outputs.append(out)
        for graph, out in zip(graphs, outputs):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(
                out.view(torch.uint8), eager(x, residual).view(torch.uint8)
            ), "CUDA Graph replay differs from eager"
        calls = (
            ("eager", lambda: eager(x, residual)),
            ("compile", lambda: compiled(x, residual)),
            ("eager_graph", graphs[0].replay),
            ("compile_graph", graphs[1].replay),
        )
        timings = {name: [] for name, _ in calls}
        for repeat in range(5):
            # Rotate order to reduce consistent ordering bias.
            order = calls[repeat % 4:] + calls[:repeat % 4]
            for name, fn in order:
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(200):
                    fn()
                end.record()
                end.synchronize()
                timings[name].append(start.elapsed_time(end) * 1000 / 200)
        print(f"BENCH bf16 batch={batch} width=2048 " + " ".join(
            f"{name}_us={statistics.median(samples):.4f}"
            for name, samples in timings.items()
        ), flush=True)
        # A graph containing 100 calls amortizes Python replay overhead. This
        # measures repeated GPU work rather than single-call Python latency.
        unrolled = []
        for fn in (eager, compiled):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(100):
                    out = fn(x, residual)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(
                out.view(torch.uint8), eager(x, residual).view(torch.uint8)
            )
            unrolled.append(graph)
        samples = [[], []]
        for repeat in range(5):
            for index in (repeat % 2, 1 - repeat % 2):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    unrolled[index].replay()
                end.record()
                end.synchronize()
                samples[index].append(start.elapsed_time(end) * 1000 / 2000)
        print(f"GPU_GRAPH_100 bf16 batch={batch} width=2048 "
              f"eager_us={statistics.median(samples[0]):.4f} "
              f"compile_us={statistics.median(samples[1]):.4f}", flush=True)


if __name__ == "__main__":
    main()
