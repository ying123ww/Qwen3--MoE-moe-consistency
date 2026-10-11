# Baseline native full CUDA Graph capture attempt

On ai_hgx_03, the existing baseline cannot directly use native full decode CUDA
Graph with its current SGLang DeepEP-normal integration. An actual capture
attempt on all eight A100 GPUs failed inside DeepEP dispatch. No formal generation
batch ran, so this experiment has no inference timing or speedup result.

## Configuration and scope

- Model: Qwen3-30B-A3B, BF16; TP=8, DP=2, attention CP=2, EP=4, MoE DP=2.
- Same baseline Triton attention/MoE and DeepEP normal path; DeepEP uses 20 SMs.
- Deterministic inference, unified decode attention, experimental RMS/QK fusion
  and router tile optimizations are off.
- Native decode backend `full`, capture bucket `[4]`, maximum batch size 4.
  Prefill remains eager as in the preceding benchmark. Even full model decode
  capture does not make scheduling, sampling, IPC or the entire request a graph.
- Only the experiment entry point bypasses `ServerArgs._handle_a2a_moe`'s
  prohibition. Production defaults and the dispatcher implementation are unchanged.
- No breakable backend or eager fallback was substituted after failure.
- Requested benchmark protocol: 3 warmup batches, 1 formal batch, 8 frozen prompts
  with 1218 input tokens total and 128 output tokens per prompt. Engine initialization
  failed, so none of these generate batches executed.

## Observed failure

All eight ranks logged `Capture target decode CUDA graph begin. backend=full,
num_tokens_per_req=1, bs=[4]` at 21:51:39 CST. Native full capture performs two
eager forward warmups before opening `torch.cuda.graph`; the failure stack is
from `forward_fn()` inside that capture context, not its warmup loop.

All eight ranks subsequently reported the primary exception:

```text
FullCudaGraphBackend.capture_one -> forward_fn
Qwen3 MoE -> DeepEPMoE.forward_impl
DeepEP normal _dispatch_core -> Buffer.dispatch -> runtime.intranode_dispatch
RuntimeError: DeepEP error: CPU recv timeout
```

The subsequent `Capture must end on the same stream it began on` is an exception
while unwinding the failed graph context. It follows the dispatch timeout in the
trace and is not the first failure. Rank initialization then aborted; the benchmark
worker exited with code 1. Its 478.13-second lifetime includes imports, distributed
initialization, weight loading, compilation and failure handling; it is not an
inference duration.

## Cause and the limits of this conclusion

Installed DeepEP is `1.1.0+6845ffd`. Its `Buffer.dispatch` defaults
`num_worst_tokens=0`; SGLang's normal `_dispatch_core` does not supply this argument.
In the matching C++ source, dispatch launches `notify_dispatch`, then the CPU polls
mapped GPU counters until the total received token count and per-expert counts are
available. It allocates receive tensors from the resulting dynamic count.

During CUDA Graph capture, the communication kernel is recorded for later replay.
The CPU wait therefore cannot obtain the count it needs to finish constructing the
graph, and hits DeepEP's receive timeout. This is consistent with the observed
exception at `runtime.intranode_dispatch`. The baseline unified decode switch is
off, so the deterministic attention KV-length `.item()` guard is not involved.

This does not establish that A100 hardware or all DeepEP normal implementations
cannot use full CUDA Graph. The installed DeepEP API explicitly documents an
intranode graph-compatible option: `num_worst_tokens > 0` allocates a fixed worst-case
receive capacity without CPU synchronization. It returns an empty CPU per-expert
count list. That option is not wired into the current SGLang path. Supporting it
would require checking the capacity bound across participating ranks, padded route
handling, expert computation, combine and changing routing across actual graph
replays, followed by correctness validation. Merely changing the backend flag is
insufficient. No performance claim is made for this unimplemented alternative.

## Artifacts and cleanup

Run directory, relative to `slime/alignment_a_preparation`:
`gpu_runs/full-graph-baseline-hgx03-20261011-01/`.

- `baseline-full.log`: all-rank capture and exception stacks.
- `results.json`: 8 full-capture attempts, 8 primary timeouts, 0 formal batches,
  null inference time, installed-version provenance and source hashes.
- `baseline-full/experimental-server-args.json` and `post-normal-hook.json`:
  requested configuration and experiment-only restoration of `full`.
- `deepep-installed-version-deep_ep.cpp`: C++ source from installed-version commit
  `6845ffd`, retained in the ignored run directory for inspection.
- Driver PID/PGID was 56223; benchmark worker PID/PGID was 56236. Both exited.
  Cleanup was restricted to this worker process group. GPU inspection afterward
  found no compute processes; all eight GPUs showed 1 MiB used and 0% utilization.
  Slurm allocation 1031236 remains RUNNING. Per the user's previous instruction,
  no keepalive task was launched.

The existing baseline breakable result remains 5.081420 seconds for one formal
batch; it must not be relabeled as a full-graph measurement.
