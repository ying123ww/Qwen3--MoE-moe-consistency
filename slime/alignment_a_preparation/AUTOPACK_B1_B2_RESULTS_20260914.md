# AutoPack B1/B2 GPU validation — 2026-09-14

## Verified native schedule

ai_hgx_05, 8 A100 80GB; CUDA_LAUNCH_BLOCKING=0. Reused capture-01: 32 distinct DAPO prompts, 128 response tokens each. B1-03 exited 0; all eight rank gates passed; each rank has three slots, yielding two EP groups × three slots (six collectives). No optimizer updates. GPU workers shut down after completion.

4096 response tokens have exact FP32 logprob bits across SGLang Engine capture, aligned Megatron fallthrough, and Megatron R3 replay_forward. Router and dispatch mismatched rows are zero. B2 independently reread 4608 route arrays (2018016 rows) and 12288 logprob values; H/U/per-expert reconstruction matched actual dispatch. Evidence: [gate-summary](gpu_runs/autopack-hgx05-b1-20260914-03/gate-summary.json), [B2](gpu_runs/autopack-hgx05-b1-20260914-03/b2-verified/).

## Offline intervention comparison

Retained sampled-best algorithm; 500 candidate / 20 iteration / 20 swap limits; padding multiplier 512. Random uses matching accepted swap count independently per seed. CPU execution on the current workspace host, not hgx05 GPU timing.

| Seed | Native E | RoutePack E | Random E | Matched swaps | RoutePack reduction |
|---|---:|---:|---:|---:|---:|
| 0 | 1013700 | 1010099 | 1013538 | 12 | 0.3552% |
| 1 | 1013700 | 1010331 | 1012744 | 14 | 0.3323% |
| 2 | 1013700 | 1010151 | 1013655 | 8 | 0.3501% |
| 3 | 1013700 | 1010378 | 1013060 | 10 | 0.3277% |
| 4 | 1013700 | 1010476 | 1011762 | 11 | 0.3180% |

Seed 0: 20 measured repeats after warm-up; identical final schedule hashes. Total CPU p50/p95 2361.79/2575.98 ms; search alone 1294.99/1445.20 ms. Other seeds have one measured repeat each and cannot establish within-seed repeatability. Disk I/O excluded; preparation and independent reconstruction included in total.

All five seeds beat their matched Random baseline, but only reduce this batch’s E by approximately 0.32–0.36%. Native and seed-0 candidate both contain 3523 padding tokens out of 12288 padded tokens (28.67%). E includes terminal and layout padding assignments. No attribution to padding or communication time was established.

## Scope and next gate

This is one short fixed batch, forward-only, with native-schedule correctness verified. Candidate schedules have only offline structural/proxy checks; no exchanged-schedule GPU correctness gate, slow-rank timing, or training-step speedup has been measured. Current seconds-scale offline CPU time is not suitable evidence of low-overhead online scheduling. Next: integrate candidate schedule replay and verify exact correctness, then compare Native/Random/RoutePack on multiple representative batches with matched token/FLOP budgets, per-collective layer timing and uninstrumented step timing. Do not claim the native routing-skew → slow-rank → step-time causal chain has been established.
