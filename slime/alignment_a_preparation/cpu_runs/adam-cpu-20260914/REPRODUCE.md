# CPU-only Adam experiment

Run on ai_hgx_05, Intel Xeon Platinum 8358P, with the slime environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 24-31 \
/public/home/yingxy2024/.conda/envs/slime/bin/python \
/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/gpu_tools/benchmark_adam_cpu.py \
--pairs 16 --output /tmp/adam-cpu-UNIQUE
```

Use a new output path. experiment01 used `--pairs 4` (the default); experiment02 used `--pairs 16`. Each pair contains 6 and 12 MiB FP32 parameters with equally sized gradients. Each parameter has its own actual StatelessAdam instance, matching the CPU math portion of HybridDeviceOptimizer. No HybridDeviceOptimizer, H2D, forward/backward, or CUDA tensors are used.

The candidate class exists only in this script's memory. It changes `abs().mul(scale)` to `abs().mul_(scale)`, preserving the operation sequence while reusing the fresh abs output. The production optimizer is unchanged. Deterministic and fill switches affect only the benchmark process. A CPU profiler may query CUDA device metadata at export and emit `gpuGetDeviceCount failed with code 100` because devices are hidden; `torch.cuda.is_initialized()` remains false, and no GPU compute is submitted.

Timing: 3 rounds, alternating thread/config order, 3 warmup and 5 measured steps per configuration. Profiler is separate. Configuration initialization is excluded; allocations inside `step()` are included. All parameter bytes after 8 steps must agree across 48 configurations per experiment. Another 16 branch combinations test 5 updates and include signed zeros, NaN, infinities, tiny and large gradients. This is a comparison to the existing StatelessAdam, not an independent mathematical Adam reference.

Shared-host limitation: Wan21 keeper runs concurrently. CPU affinity limits the experiment to cores 24–31, but does not reserve these cores from other processes. Do not extrapolate timings to eight simultaneous optimizer ranks or the full 14.218 GiB/rank layout. Do not infer memory-bandwidth saturation or a production-optimal thread count.

Keeper restored after finding all GPUs idle and the old recorded process absent. PID/PGID 79791 on ai_hgx_05; worker PIDs 79898–79905. `keepalive.log` confirms `Trainer.train_one_step (first BP done)`. At 16:04 UTC all eight GPUs showed 93–95% utilization. Keep this keeper running; no formal GPU experiment was run for this task.
