# RoutePack CPU checks — 2026-09-13

Added seeded Random-Swap with identical legality rules, matched accepted-swap comparison, candidate/iteration/swap budgets, cooperative wall cutoff, phase timings and deterministic tie-break checks. 33 CPU tests passed in psamp Python 3.12. The warning from the existing Megatron CPU stub is expected; no CUDA execution took place.

Synthetic fixture: 128 samples, 32 bins, 48 layers, EP2, 17 tokens/sample. Fixed input seed 19; random seed 0; candidate budget 500, iteration budget 20, swap budget 20. One warm-up, 20 reset-to-Native repetitions. Both algorithms accepted 18 swaps. All 20 RoutePack final schedule hashes matched.

| Method | Excess |
|---|---:|
| Native | 68574 |
| Random-Swap | 58846 |
| RoutePack | 64858 |

Random outperformed this first-improvement proxy search on this synthetic case. Do not cherry-pick a more favorable seed or treat this as production workload evidence. There is no proof of superiority to random, communication improvement, or GPU speedup.

| CPU phase | p50 ms | p95 ms |
|---|---:|---:|
| Prepare | 341.40 | 356.03 |
| Search | 70.47 | 74.37 |
| Validation/report | 344.16 | 353.36 |
| Total | 758.65 | 780.41 |

Timings exclude disk I/O but include route summaries and full reconstruction in the total. This offline callable is not a low-millisecond production scheduler. Search time alone must not be used to claim end-to-end overhead. Quantiles per phase do not add exactly to total quantiles.

Wall limits stop search cooperatively, not preparation/validation mid-call, and can yield different stopping points under varying load. Reproducibility uses fixed count budgets; random uses a private seeded generator. A random baseline failing to reach the RoutePack swap count is flagged and excluded from the matched excess comparison.

Artifacts: `exchange_cpu_benchmark_20260913.json` contains all repeated timings, swaps, before/after H/U metrics, seed/settings and schedule hash. Entrypoints: `tools/routepack/benchmark_exchange.py` for B2 data and `tools/routepack/benchmark_exchange_synthetic.py` for this reproducible synthetic fixture. Algorithm work stops here as requested; real representative data, training integration, numerical gates and GPU timings remain outstanding.
