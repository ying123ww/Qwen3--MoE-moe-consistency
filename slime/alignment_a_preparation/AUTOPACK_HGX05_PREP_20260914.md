# AutoPack hgx05 first multi-slot capture preparation

CPU preparation only; no GPU experiment was launched by this preparation task.

- Node: ai_hgx_05; verified 8 × A100-SXM4-80GB.
- HF: `/storage/data/yingxy2024/Models/Qwen3-30B-A3B`.
- Training checkpoint: `/storage/data/yingxy2024/Models/Qwen3-30B-A3B_torch_dist`.
- Data: `/storage/data/yingxy2024/data/dapo-math-17k/dapo-math-17k.jsonl`.
- Python: `/public/home/yingxy2024/.conda/envs/slime/bin/python`.

## Prepared entrypoints

`gpu_tools/autopack_capture_hgx05.py` consumes `--prompts-json`, `--reference-manifest`, `--output`, `--response-tokens 128`. Reference: `gpu_runs/router-exact-off-20260912-02/capture/manifest.json`. Prompts: `gpu_runs/autopack-hgx05-20260914-input/prompts.json`. Runs one Engine, saves true routes/tokens/FP32 response logprobs and bits, produces restricted-load-readable `rollout.pt` for existing B1. Requires 32 distinct prompts, total sequence <=512. Capture outputs are not performance measurements. Use an existing validated SGLang runtime env with node-local CUDA paths repaired. The wrapper explicitly sets CUDA_LAUNCH_BLOCKING=0 before SGLang import.

`gpu_tools/autopack_b1_hgx05.py --rollout CAPTURE/rollout.pt --output NEW_RUN --cuda-home VALID_CUDA_HOME` reuses the previously passed B1 options/environment, changes global batch to32, rollout batch to16 and max tokens per GPU to512. It creates a local Ray cluster rather than attaching to other services and shuts that cluster down in finally. No optimizer/backward. --dry-run passed on hgx05; output `gpu_runs/autopack-hgx05-b1-dryrun-20260914/` is an inspection artifact with a placeholder input, not a real launch configuration.

B1 launch checks empty GPUs, available system RAM >=512GiB and disk >=20GiB. These are preflight checks, not cgroup peak guarantees: supervisor should monitor allocation memory and apply bounded timeout. Runtime env and argv are persisted before execution. Caches are isolated per output directory. CUDA_LAUNCH_BLOCKING=0 is explicit.

## Acceptance and limits

All eight rank gates must pass; inspect schedules to prove each EP group has multiple slots (do not infer only from32 samples). Run `tools/routepack/summarize_b1.py` with B1/capture/prepared rollout and fresh output. Only then compare native/random/AutoPack on true H/U.

The first32 short unique prompts are a chain-integrity test, not representative whole-dataset performance evidence. Multiple batches and uninstrumented timing still needed. Existing capture_engine.py alone does not persist the logprob fields required by prepare_b1.py, which is why the independent wrapper was necessary. Existing launch.py hardcodes ai_hgx_03 and its memory guard, so it was not used or modified.

## Actual execution

Capture `gpu_runs/autopack-hgx05-capture-20260914-01` succeeded: 32 distinct prompts,128 response tokens each, true routes and FP32 logprobs persisted; `capture/rollout.pt` SHA256 `665c97cfcba60a854a7712a8c5cfb41f9fe2edd807045107e06040409439f6fd`. Engine shutdown and empty GPU list verified before B1.

B1-01 stopped at disk preflight (19GiB result disk versus20GiB initial gate). B1-02 moved cache to/tmp, retained2GiB result/20GiB temporary disk checks, started independent Ray and initialized training. An unrelated Wan21 job appeared after the empty-GPU check, using ~45GiB on every GPU; B1 failed to allocate a13.5GiB DDP buffer. This is resource contention, not a numerical or routing failure. No backward/optimizer or dispatch gate was reached. Ray shutdown completed and GPU list contains only external Wan21 workers. No external process was stopped.

Evidence: `gpu_runs/autopack-hgx05-b1-20260914-02/status.json` and adjacent `.log`. More GPU attempts paused until an exclusive8GPU window. CPU input/capture evidence remains reusable.

## Resumed B1-03 result

After the user stopped the external task, GPU availability was rechecked and B1-03 reused the successful capture. All8rank gates passed, each with3slots:2EPgroups×3slots=6collectives. All4096response tokens agree bitwise across SGLang Engine, aligned Megatron fallthrough and R3 replay_forward. Router/actual dispatch mismatched rows=0; optimizer steps=0. The process exited0 and GPU compute list was empty after Ray shutdown.

`gpu_runs/autopack-hgx05-b1-20260914-03/gate-summary.json` preserves rank-by-rank results. `b2-verified/validation.json` reports4608actual route files/2018016route rows checked, dispatchH/U/per-expert matches and12288saved logprob values checked. This is the first real multi-slot original-schedule correctness/load reconstruction result; it does not validate a changed schedule or performance.
