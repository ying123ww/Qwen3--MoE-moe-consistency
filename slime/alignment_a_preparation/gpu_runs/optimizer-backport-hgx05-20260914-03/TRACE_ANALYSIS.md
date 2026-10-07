# Isolated optimizer trace analysis (2026-09-14)

## Scope and accounting

Actual optimizer with synthetic gradients, 128 parameters, 1.125 GiB FP32 master data; no forward/backward. Each trace covers one profiled optimizer step. Speed comparisons use separate device-synchronized, unprofiled repetitions. CUDA APIs take priority in exclusive main-thread accounting, then the innermost CPU op; parent/child durations are not added together. GPU transfer intervals are linked by correlation and intersected with these disjoint CPU intervals. CPU wall intervals include worker waits, not core utilization.

Runs 01/02 inherited CUDA_LAUNCH_BLOCKING=1 (confirmed by coordinator); they cannot test asynchronous overlap recovery. Their pinned H2D intervals fall almost entirely inside the following blocking cudaLaunchKernel (GPU dtype conversion), not CPU math. Run 03 explicitly disables launch blocking and is the appropriate overlap experiment.

## Run 03

| Variant | Unprofiled median of 3 repetition medians (ms) | H2D API (ms) | H2D GPU union (ms) | H2D / exclusive CPU math overlap (ms) | H2D / fill overlap (ms) |
|---|---:|---:|---:|---:|---:|
| old | 372.765 | 140.495 | 90.954 | 0.000 | 0.000 |
| pin_only | 218.182 | 1.729 | 48.537 | 18.231 | 14.700 |
| event_only | 325.590 | 102.443 | 93.630 | 0.000 | 0.000 |
| both | 220.663 | 1.766 | 49.559 | 13.922 | 17.549 |

Both fixes restore observed partial overlap: 13.922 ms with math alone (28.09% of H2D); including CPU fill work gives 31.472 ms (63.50%). Pin-only is an ablation, not a deployable recommendation: its event dependency remains incorrect. Differences between variants cannot all be assigned causally to overlap; pinned transfer bandwidth, pageable staging, dependency scheduling, and measurement variability also matter.

Both-fix median 220.663 ms versus old 372.765 ms is a 40.80% isolated optimizer reduction. Do not extrapolate it to the historical 15.4-second optimizer or full training. Single-step profile durations are not the timing baseline. The old historical job environment must be audited separately before attributing its missing overlap to launch blocking.

Machine-readable per-op attribution: `exclusive_trace_analysis.json`. Reproduction adapter: `../optimizer-backport-hgx05-20260914-01/analyze_traces.py <experiment-directory>`.
