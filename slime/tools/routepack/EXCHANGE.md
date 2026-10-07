# RoutePack offline sample exchange

Retained method: seeded sampled best-improvement. Native remains unchanged unless this tool is explicitly called; live training integration and GPU correctness/performance gates are not implemented.

Each iteration samples up to 25 structurally eligible pairs from a seeded permutation. Exchanges are complete samples within one optimizer step and across different slots. The best legal strictly improving pair is accepted; gain ties use cell and sample-position indices. The objective is sum over group/slot/layer of max-rank H minus mean-rank H, including terminal and recomputed bin padding. Fixed rank/slot cells and sample counts are preserved; padded bin capacity cannot be exceeded and padding cannot increase. Default capacity is each native bin's padded size. H/U are independently reconstructed after search; U is a token/destination count, not time or bytes.

```bash
python tools/routepack/exchange.py INPUT_B2_DIR NEW_OUTPUT_DIR --pad-multiplier 512 --seed 0
python tools/routepack/benchmark_exchange.py INPUT_B2_DIR NEW_RESULT.json --pad-multiplier 512 --repeats 20 --max-candidates 500 --max-iterations 20
python tools/routepack/benchmark_exchange_synthetic.py
PYTHONPATH=. python -m pytest -q tests/test_routepack_exchange.py
```

Use the actual runtime padding multiplier. Inputs: samples.jsonl with route_path, schedule.json, topology.json and original route arrays. The profile_data.py validator requires actual topology provenance and rejects invalid traces/schedules. Outputs: candidate schedule, exchange report with provenance/settings and H/U arrays. Existing outputs are rejected.

Random-Swap is retained only as a baseline (`--mode random`): same legality rules, no excess filter. Benchmark resets to Native each run and matches random accepted-swap count to sampled-best; unmatched random results are marked and excluded. Fixed seed/input/count budgets reproduce schedules. Time-based cutoff can alter the stopping point and is cooperative, not a hard end-to-end deadline. Candidate budget counts evaluations, not pair enumeration/shuffling. CPU total includes preparation/full reconstruction but excludes disk I/O; one warm-up is excluded from benchmark quantiles.

On the fixed synthetic EP2/48-layer/128-sample/32-bin fixture, five search seeds gave excess 48,802–48,978 versus matched random 53,334–57,566 and Native 68,574. Search cost was 133–152 ms; this is not evidence of low-latency scheduling or GPU benefit. Previous first-improvement and construction implementations were removed after comparison; historical experiment outputs remain evidence, not supported algorithms. The synthetic benchmark writes a new sampled_best_cpu_benchmark.json without overwriting historical runs.
