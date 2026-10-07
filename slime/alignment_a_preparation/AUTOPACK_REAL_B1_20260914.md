# AutoPack real-length original-schedule B1/B2 validation

## Result

`autopack-real-hgx05-b1-20260914-02` passed all eight ranks. The original schedule has12slots per rank,2EPgroups and24collectives. All1,599,412response tokens match bitwise across SGLang Engine capture, aligned Megatron fallthrough and Megatron R3 replay_forward. Actual router/dispatch mismatched rows=0. No optimizer or backward was run.

B2 independently reread18,432route arrays,316,896,384route rows and4,798,236saved logprob values. Actual dispatch H/U/per-expert counts match reconstructed data. This verifies real-length original-schedule routes and forward numerical consistency, not exchanged-schedule correctness or training speed.

## Workload and provenance

Repository recipe `scripts/run-qwen3-30B-A3B.sh:47` supplies32prompts×8samples=256, response cap8192, sampling temperature1, global batch256, balance-data and max_tokens_per_gpu20480. We retain the previously verified alignment topology TP1/EP4, which differs from that recipe's TP4/EP8. The recipe is a workload source, not evidence of an earlier production deployment.

The corrected capture02 uses explicit within-group seeds42..49, producing256distinct complete token sequences. Capture01 was discarded because omitted seeds could duplicate sibling responses. Total tokens1,639,308; response tokens1,599,412;145responses stopped naturally and111hit8192. Declared context cap16384; no sample truncation in B1. Data-pad multiplier512.

## OOM and unchanged-workload retry

B1-01 failed when full logits were converted to FP32:11.30GiB allocation requested with10.84GiB free. The error also reported allocator-reserved unallocated memory. B1-02 set only `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and completed; precision, token shapes, batch, max_tokens20480 and topology remained unchanged. This is consistent with an allocation-fragmentation contribution, not a separate measured performance claim.

Both runs have logs and supervisor exit records. The supervisor sampled process-tree RSS, available RAM and disk every5seconds, with owned-process-only termination thresholds. B1-02 exited0 and GPUs were empty after Ray shutdown. Root subsequently restored the authorized keepalive workload.

## Artifacts

Shared root: `alignment_a_preparation/gpu_runs/autopack-real-hgx05-b1-20260914-02/`.

- `gate-summary.json`: rank-level pass flags,12slots each, three-way exact checks.
- `b2-validation.json`: independent array/count/logprob verification.
- `b2-source_hashes.json`, `launch-source-hashes.json`: source provenance.
- `argv.json`, `runtime_env.json`: actual workload and allocator settings.
- `resource-summary.json`, adjacent `.memory.jsonl`: sampled resource use.
- `supervisor-exit.json`, adjacent `.log`: execution status/full logs.

Node-local heavy data on ai_hgx_05:

- Corrected capture: `/tmp/ap-real-capture-20260914-02/capture/`.
- Actual B1 arrays: `/tmp/ap-real-b1-arrays-20260914-02/`.
- Verified B2 data: `/tmp/ap-real-b2-verified-20260914-02/`.

Offline Native/Random/AutoPack comparison follows separately. No schedule intervention, backward/parameter-update consistency or GPU timing claim is made here.
