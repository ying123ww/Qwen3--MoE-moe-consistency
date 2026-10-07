# RoutePack B1/B2 observation tools

These tools do not change the online scheduler. CPU fixtures are tests, never runtime topology or benchmark evidence. The current status is recorded in `alignment_a_preparation/routepack_b_20260912/REPORT.md`.

## B1: export actual training topology

Prepend the standalone package directory to the launch environment and each Ray worker's `PYTHONPATH`:

```bash
export PYTHONPATH="/public/home/yingxy2024/moe-consistency/slime/tools:${PYTHONPATH:-}"
```

Use module names beginning with `routepack.`, not `tools.routepack.`: the latter conflicts with Megatron's ordinary `tools` package. `routepack.b1_gate.configure_launch` performs the worker-environment insertion for the full B1 gate. Its init hook is `routepack.b1_gate.install`, and its observation hook is `routepack.b1_gate.before_log_prob`.

For a new independent aligned training run, set `ROUTEPACK_PROFILE_DIR` to its output directory and replace the existing `--custom-megatron-before-log-prob-hook-path` with:

```text
routepack.runtime_topology.aligned_before_log_prob
```

The wrapper invokes the original Qwen alignment hook, then reads initialized process groups and actual `mlp.local_expert_indices`. It adds no collective, GPU copy or synchronization. Keep all other training hooks and numerical arguments unchanged. When composing with another custom hook, call `export_topology(args, model, store_prefix)` after that hook instead. All training ranks must execute the hook. Use a fresh output directory; each rank writes only its first topology snapshot.

After the run, merge the actual rank exports:

```bash
python tools/routepack/profile_data.py merge-topology RUN_DIR RUN_DIR/topology.json
```

The merger rejects missing/duplicate ranks, inconsistent ordered group membership, non-singleton dense TP/expert TP/CP/PP, VPP>1, incomplete/duplicate expert ownership, and differing owner layouts between EP replicas. Dense DP rank order comes from the process group, never from GPU numbering. This proves topology only; it does not prove that replayed routes reach the dispatch path.

## B2: inspect existing rollout coverage

```bash
CUDA_VISIBLE_DEVICES='' python tools/routepack/inspect_rollout.py ROLLOUT.pt inventory.json
```

This uses `torch.load(weights_only=True, map_location='cpu')`. It does not retry with unrestricted pickle. The inspected v89 rollout has 8 samples and **zero routing traces**.

## Guarded GPU route capture

Run `routepack-capture-20260912-02` has completed successfully after the Qwen capture fix: 8/8 valid traces, 48 layers × top8, 2,055,168 raw bytes. Evidence is in `alignment_a_preparation/routepack_b_20260912/capture02_verified/`. Training topology, dispatch and replay exact gates remain pending. Use a new run ID for further captures; the successful run is retained.

Preferred guarded launcher on the allocated `ai_hgx_03` node:

```bash
/public/home/yingxy2024/.conda/envs/slime/bin/python \
  alignment_a_preparation/gpu_tools/run_matmul_attribution.py --capture-routes \
  --run-id routepack-capture-20260912-03
```

It reuses the A-stage allocation/GPU-idle checks, memory guard, launcher lock and isolated caches. The default capture run ID is `routepack-capture-20260912-01`; supply a fresh lowercase run ID to write `gpu_runs/<run-id>/capture`. Run IDs accept only lowercase letters, digits, underscores and hyphens, with an alphanumeric first character. Existing output directories are rejected. The launcher retains its original matmul mode when this flag is omitted. Source hashes include both capture scripts, the reference manifest, Qwen model, TopK and route/base capturer. The first capture revealed a missing capture callback in the deterministic Qwen path; the minimal runtime fix and CPU regression evidence are in the status report.

Run from the slime repository on the allocated GPU node, using the same environment/PYTHONPATH/CUDA dependencies and exclusive GPU allocation as the A-stage reference. The reference manifest must come from an aligned inference run and contain `requested_server_args` and `sampling`. The script preserves its topology, adds route capture and generates one batch of 16 response tokens per prompt. A new Engine is required; an Engine started without route capture cannot be retroactively used.

```bash
python tools/routepack/capture_engine.py \
  --reference-manifest alignment_a_preparation/gpu_runs/a-matmul-attribution-20260912-01/manifest.json \
  --frozen-rollout /public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt \
  --output alignment_a_preparation/gpu_runs/routepack-capture-UNIQUE_RUN_ID \
  --response-tokens 16
```

This is an 8-GPU startup with the reference's TP=8 / DP=2 / attention CP=2 / EP=4 inference configuration, not a CPU command. Budget is dominated by model load; elapsed capture time is diagnostic only. The new flag allocates route buffers and adds collection cost. Do not mix its timings into the original inference overhead comparison.

Exact source API: `Engine(enable_return_routed_experts=True, ...)` and `generate(return_routed_experts=True, routed_experts_start_len=0, return_logprob=True, ...)`. The response's `meta_info.routed_experts` is base64 int32 data, expected shape `[prompt_tokens + completion_tokens - 1, 48, 8]`. The last generated token has no forward route yet. `save_engine_routes` validates coverage/range/topk, preserves source IDs, token identities and hashes, and writes safe `.npy` files. Source reward/rollout IDs are provenance only; the standalone run computes no reward or training step assignment.

Capture output is **B2 preflight**, not a complete profiler run. Actual training topology, model-weight match, sample/position route equality at router/dispatch, alignment+replay exact logprob, real schedule, and GPU timings remain separate gates.

## Summarize an actual native schedule

Call `native_schedule_cells(partitions, micro_batch_indices, num_microbatches, sample_ids, topology)` on the actual outputs of `build_dp_schedule`. This converts dense DP rank to global rank and EP group without changing any placement. Save its result as `schedule.json`. `step` is zero-based within this schedule.

Supply `samples_input.jsonl` with one record per kept sample:

```json
{"sample_id":"run:0","step":0,"rollout_id":0,"reward_group_id":0,"model_version":"frozen-checkpoint-id","expert_layout_version":"frozen-layout-id","token_count":280,"response_length":128,"valid_loss_token_count":128,"layer_ids":[0,1],"route_path":"routes-0.npy"}
```

`layer_ids` must contain every actual exported layer (the abbreviated example has only two); `route_path` contains integer `[token_count-1,layers,topk]`. `model_version` and `expert_layout_version` are required declarations; their equality to the actual checkpoint/layout must be established by the run manifest and B1 gate. The tool checks within-run consistency, not checkpoint identity.

```bash
python tools/routepack/profile_data.py summarize RUN_DIR --pad-multiplier 512
```

Use the actual training `data_pad_size_multiplier`; 512 is an example, not a universal default. The V1 scope fixes CP/TP/PP/expert TP/VPP to 1. Outputs are `samples.jsonl`, `collectives.jsonl`, `route_loads.npz` (sample H/U/per-expert, synthetic terminal, per-bin layout padding, and complete group/slot aggregation). Accumulators are int64. H counts assignments; U counts unique token/destination pairs. Neither is communication bytes or GPU time. No `timings.jsonl` is manufactured. `Go` stays false until separate runtime and intervention gates pass.

## CPU validation

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" python -m pytest -q tests/test_routepack_profile_data.py
```

Tests include invalid topology and traces, missing/duplicate samples, optimizer-step boundaries, H vs U, native rank mapping, real production replay padding with CP=TP=1, the same-collective source-swap counterexample, and singleton exchanges across independent EP replicas. The replay-layout tests use the existing CPU Megatron stub and do not run dispatch.

## Summarize completed B1 evidence into B2 artifacts

`tools/routepack/summarize_b1.py` is a CPU-only entry point for the newer forward-only `b1_gate.py` outputs. Run it only after every B1 rank has successfully written `gate.json`. Use the exact prepared rollout loaded by that B1 run, because it supplies the IDs and masks actually used by this forward fixture; do not infer them from the earlier standalone capture's null rollout IDs. In `prepare_b1.py`, `index=i`, `rollout_id=i`, and `group_index=i//2` are constructed fixture fields, response loss masks are ones, and `reward=0` is an unused placeholder. These are not preserved production reward/grouping semantics. Source capture IDs remain in metadata for traceability.

```bash
CUDA_VISIBLE_DEVICES='' python tools/routepack/summarize_b1.py \
  --b1-directory B1_RUN/b1 \
  --capture-directory alignment_a_preparation/gpu_runs/router-exact-off-20260912-02/capture/case-batch8/captured \
  --prepared-rollout B1_ACTUAL_INPUT.pt \
  --output B2_NEW_OUTPUT_DIRECTORY \
  --run-id B1_RUN_ID
```

Required evidence: all `topology-rank-N.json`, all `rank-N/schedule.json`, all passed `rank-N/gate.json`, every router/dispatch `.npy` referenced by those gates, the source capture JSONL/tokens/routes, and the actual `prepare_b1` rollout `.pt`. The `.pt` is read with the restricted CPU loader. The capture must declare effective logprob temperature 1.0, as used by the B1 conversion.

The tool maps the real execution of the fixture's native rank schedule to EP groups, validates the constructed sample/step/rollout identity and runtime padding, and rechecks both forward stages' actual route arrays and saved FP32 logprob bits against the capture. A `step` here is a schedule partition; B1 executes no optimizer update. Fallthrough checks original trace rows; replay checks all rows including synthetic terminals and layout padding. It then separately sums the real replay dispatch arrays and requires their H/U/per-expert totals to equal the counts reconstructed from source traces and actual layout. Missing, failed or inconsistent evidence stops the tool before creating its output directory.

Outputs are `topology.json`, `schedule.json`, `samples.jsonl`, `collectives.jsonl`, `route_loads.npz`, `validation.json`, and `source_hashes.json`. The NPZ includes both reconstructed and independently counted actual dispatch loads. `timing_gate=not_run` and `go_decision=not_evaluated`: this is one original schedule's data evidence, not a timing experiment, an optimizer update or a scheduling intervention.

CPU fixture tests: `python -m pytest -q tools/routepack/test_summarize_b1.py` (8 tests, all individually verified). They check two-rank and noncontiguous EP groups with reordered dense-DP membership, rejection of failed gates/corrupted dispatch IDs or logprob bits/changed tokens/invalid global-batch counts, and invocation-independent relative capture provenance. These fixtures are not runtime evidence. No real B1 result was assumed while preparing this entry point.

The resulting B2 H/U and per-expert counts describe real routes dispatched under this small forward fixture. They do not establish production workload diversity, preservation of original RL rewards/GRPO grouping, gradient/update equivalence or performance benefit. Future workload analysis must separately capture naturally generated training batches and retain their original identities, rewards, masks and optimizer-step boundaries.

## Offline cross-slot sample exchange

The retained method is seeded sampled best-improvement, with a Random-Swap control, CPU timing and determinism checks. See [EXCHANGE.md](EXCHANGE.md). First-improvement and construction experiments have been removed; historical results remain available as evidence.
