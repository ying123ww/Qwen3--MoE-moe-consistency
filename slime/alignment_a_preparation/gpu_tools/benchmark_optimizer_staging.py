#!/usr/bin/env python3
"""Standalone pageable FP32 -> GPU BF16 copy versus bounded pinned staging.

Does not import or patch the training optimizer. End-to-end timing includes CPU
staging copies and completion of GPU casts. This is a transfer microbenchmark,
not an estimate of model training speedup. Run on an allocated, idle GPU.
"""
import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes-mib', type=int, nargs='+', default=[6, 12, 64])
    parser.add_argument('--buffer-mib', type=int, default=32)
    parser.add_argument('--buffers', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes_mib + [args.buffer_mib, args.buffers, args.repeats]) <= 0:
        parser.error('Sizes, buffer count and repeat count must be positive')
    import torch
    torch.cuda.set_device(0)
    torch.manual_seed(20260912)
    sources = [torch.randn(mib * 1024 * 1024 // 4, dtype=torch.float32) for mib in args.sizes_mib]
    # Include BF16 rounding boundaries and special values in exact comparisons.
    edge = torch.tensor([0., -0., float('inf'), -float('inf'), float('nan'), 1.00390625, 1.01171875])
    for source in sources:
        source[:edge.numel()] = edge
    direct = [torch.empty(x.numel(), device='cuda', dtype=torch.bfloat16) for x in sources]
    staged = [torch.empty_like(x) for x in direct]
    capacity = args.buffer_mib * 1024 * 1024 // 4
    buffers = [torch.empty(capacity, dtype=torch.float32, pin_memory=True) for _ in range(args.buffers)]
    events = [torch.cuda.Event() for _ in buffers]
    used = [False for _ in buffers]
    stream = torch.cuda.Stream()

    def copy_direct():
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for source, destination in zip(sources, direct):
                destination.copy_(source, non_blocking=True)
        torch.cuda.current_stream().wait_stream(stream)

    def copy_staged():
        stream.wait_stream(torch.cuda.current_stream())
        index = 0
        for source, destination in zip(sources, staged):
            for offset in range(0, source.numel(), capacity):
                slot = index % len(buffers)
                if used[slot]:
                    events[slot].synchronize()  # CPU must not overwrite in-flight source.
                length = min(capacity, source.numel() - offset)
                buffer = buffers[slot][:length]
                buffer.copy_(source[offset:offset + length])
                with torch.cuda.stream(stream):
                    destination[offset:offset + length].copy_(buffer, non_blocking=True)
                    events[slot].record(stream)  # Covers transfer and GPU FP32 -> BF16 cast.
                used[slot] = True
                index += 1
        torch.cuda.current_stream().wait_stream(stream)  # Parameter consumer dependency.

    timings = {'direct_s': [], 'staged_s': []}
    for iteration in range(args.repeats + 2):
        cases = [('direct_s', copy_direct), ('staged_s', copy_staged)]
        if iteration % 2:
            cases.reverse()
        for name, fn in cases:
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if iteration >= 2:
                timings[name].append(elapsed)
        for expected, actual in zip(direct, staged):
            # Integer bit-pattern comparison includes NaNs and signed zero.
            if not torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
                raise RuntimeError('Staging changed FP32 -> BF16 bit patterns')
    output = {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'sizes_mib': args.sizes_mib, 'staging_bytes': args.buffers * args.buffer_mib * 1024 * 1024,
              'source_pinned': [x.is_pinned() for x in sources], 'bf16_bitwise_equal': True,
              'times': timings, 'median_s': {k: statistics.median(v) for k, v in timings.items()},
              'limitation': 'Transfer-only, no CPU Adam overlap, model state, distributed collectives or profiler. Not a training speedup result.'}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
