# CPU optimizer 搬运路径快速排查（2026-09-12）

**已定位约 3 秒 pageable HtoD 的具体来源：CPU FP32 master parameter 在 StatelessAdam 更新后的回拷 hook。** 两种训练实现都走这条路径，优化它针对共同训练耗时，不能宣称降低了“一致性额外开销”。本轮没有改线上 optimizer，也没有重新启动整模型训练。

## 证据与调用链

原采样 `runtime_env.json` 指向 `/public/home/yingxy2024/slime-deps/Megatron-LM`，并非项目内 `Megatron-LM`。本次检查读取实际路径，另存 源码哈希（本地产物：`gpu_runs/p1-optimizer-inspected-source-hashes-20260912.json`）。这是本次读取快照，不能追溯宣称为 9 月 10 日原始源码哈希。

1. `slime/slime/backends/megatron_utils/model.py:250,309` 使用 `_patch_megatron_adam(StatelessAdam)` 同时替换 Megatron 的 Adam/CPUAdam。
2. 实际 Megatron `optimizer/__init__.py:306` 在 `optimizer_cpu_offload` 下创建 `HybridDeviceOptimizer(param_update_in_fp32=True)`；运行参数含 `--overlap-cpu-optimizer-d2h-h2d`、`--use-precision-aware-optimizer`、`--use-stateless-adam`。命令行 offload fraction 默认 1.0。
3. 实际 `cpu_offloading/hybrid_optimizer.py:274` 先 `param.detach().clone().cpu().pin_memory()`，随后第 277 行为 FP32 master 再 `param.detach().clone().float()`。**pinning 没有保留到最终 FP32 master。** 主任务在 ai_hgx_03 / torch 2.11.0+cu129 小内存实测：原 pinned BF16 `is_pinned=True`，`detach().clone()` 为 False，继续 `.float()` 仍为 False；原始记录见 pinning smoke（本地产物：`gpu_runs/p1-optimizer-pinning-smoke-20260912.json`）。这不是仅凭代码猜测。
4. 第 110–116 行分配 pinned FP32 gradient buffer，并从 GPU 异步 DtoH；第 171–175 行逐子 optimizer 等待其 D2H event 后执行 CPU StatelessAdam。
5. CPU Adam 的 post hook 第 121–128 行，在 `_h2d_stream` 上执行 `gpu_param.data.copy_(param.data, non_blocking=True)`。这里的源就是 pageable FP32 master。`non_blocking=True` 本身不会把源变成 pinned。

旧 trace 没有 Python stack/输入形状，不能直接从栈名断言张量对象；但所有相关 GPU memcpy 均能以 **correlation → CUDA API、External id → CPU copy** 关联，且 HtoD 的 CUDA API 全部位于 `Optimizer.step#StatelessAdam.step` 内。该 annotation 包含 post hook，不是纯 Adam 数学算子时长。关联证据与源码指向同一回拷路径。

## 四份 trace 的字节和耗时

机器可读归因结果（本地产物：`gpu_runs/p1-optimizer-transfer-attribution-20260912.json`） 包含四份原 trace SHA256、分析脚本哈希、correlation 示例、算子关联、拷贝大小直方图。与此前仅按时间窗口求交相比，这次以 **CPU optimizer 区间内的发起 API** 归属拷贝，即使 GPU 完成晚于 CPU 区间也能归属。四份 trace 没有窗口内无法匹配 optimizer API 的 memcpy。

| 采样 | optimizer CPU 秒 | pageable HtoD 次数 | HtoD GPU 累计秒 | HtoD CUDA API CPU 累计秒 | DtoH GPU 累计秒 | event synchronize CPU 秒 |
|---|---:|---:|---:|---:|---:|---:|
| native rank 0 | 15.402 | 1537 | 3.081 | 4.501 | 1.654 | 0.139 |
| aligned rank 0 | 15.411 | 1537 | 3.277 | 4.700 | 1.645 | 0.139 |
| native rank 4 | 15.433 | 1612 | 3.202 | 4.710 | 1.722 | 0.031 |
| aligned rank 4 | 15.512 | 1612 | 3.310 | 4.821 | 1.711 | 0.031 |

每份 trace 的 HtoD 都是 **15,266,061,312 bytes（14.218 GiB）**；DtoH 多 4 bytes，这额外一次由 `aten::_local_scalar_dense` 发起，其余为 1537/1612 次梯度 copy。rank 0 的 HtoD 分布：768 × 6 MiB、768 × 12 MiB、1 × 734.851 MiB。rank 4 相同总字节，但最后一部分分成更多小参数。只采样 rank 0、4，不能当成八 rank 全部实测。

因此问题不只是“有很多显式 synchronize”：rank 0 的 1537 次 `cudaEventSynchronize` 共约 0.139 秒；**4.5–4.7 秒落在 HtoD 的 `cudaMemcpyAsync` CPU 调用内**，pageable staging/驱动行为是需验证的方向。GPU copy 时长、CPU API 时长、optimizer 父区间存在包含与重叠，不能相加，也不能当作可全部回收的时间。

CPU 数学也有显著成本：optimizer 范围内 `aten::addcdiv_` native/aligned rank 0 约 2.301/2.430 秒，`aten::mul` 约 1.824/1.784 秒。分析 JSON 的 operator 时长是 inclusive，`abs` 等嵌套同名算子会重复累计，不能拿来做阶段分摊。这些不是本轮 staging 改动的对象。

## 有界 staging 实测与下一项实验

优先比较 **直接 pageable FP32 → GPU BF16** 与 **有界 pinned FP32 staging → GPU BF16**，保持 CPU FP32 master、Adam 数学、GPU cast 位置不变。用 2 × 32 MiB staging buffer，每 rank 额外 pinned 内存 64 MiB，八 rank 合计 512 MiB；超过 32 MiB 的参数分块，不能按最大的 735 MiB 参数给所有 buffer 扩容。

已经提供独立 [microbenchmark](gpu_tools/benchmark_optimizer_staging.py)：默认源 tensor 6/12/64 MiB，覆盖多 buffer 复用；CPU staging 和 GPU 完成都计入端到端时间；交替测试顺序；逐 bit 验证 BF16 结果，包含 NaN、正负零和舍入边界。它不导入训练 optimizer。运行形式：

```bash
/public/home/yingxy2024/.conda/envs/slime/bin/python slime/alignment_a_preparation/gpu_tools/benchmark_optimizer_staging.py --output slime/alignment_a_preparation/gpu_runs/p1-optimizer-staging-microbench-20260912.json
```

**已在 ai_hgx_03 的 A100-SXM4-80GB、torch 2.11.0+cu129 完成上述 microbenchmark。** 两次预热后交替顺序测量各 5 次，6/12/64 MiB 共 82 MiB 的 FP32 源，2 × 32 MiB pinned staging。直接 copy 中位 **7.120 ms**，staging 中位 **4.175 ms**，本微基准减少约 **41.4%**；所有轮次 BF16 输出逐 bit 相同。计时包含 CPU staging copy、GPU transfer/cast 与完成等待。见 原始五次计时及验证结果（本地产物：`gpu_runs/p1-optimizer-staging-microbench-20260912.json`） 和 脚本/结果哈希（本地产物：`gpu_runs/p1-optimizer-staging-microbench-hashes-20260912.json`）。

这是单 GPU 搬运可行性证据，**不是训练提速 41.4%**：不含 CPU Adam 更新重叠、1537/1612 个参数的完整序列、最大的约 735 MiB 参数、八 rank 资源竞争和分布式通信，也未验证线上流依赖修复。下一步可增加默认关闭的训练侧 staging 开关；仍须经非零学习率、小模型参数/梯度与 exact gate 验证后，用关闭 profiler 的原 training-phase 口径比较。当前 lr=0 旧采样不能替代真实更新验证；microbenchmark 的随机值和特殊值只验证搬运/cast，不证明 Adam 数学或整模型 correctness。

### 必须处理的流依赖

源码第 125 行在 `_h2d_stream` 发起 copy，第 128 行却从 `_d2h_stream` 记录事件让 current stream 等待。**此处局部代码没有建立“current stream 等待 H2D 完成”的直接依赖。** 这是一项明确的源码风险，尚未复现为旧运行错误；不能排除外层同步或当前 pageable 行为掩盖了它，不能宣称既有结果错误。

staging microbenchmark 显式做到两件事：每块 copy（含 GPU cast）后在 H2D stream 记录 event，CPU 复用对应 pinned buffer 前 `event.synchronize()`；参数消费者 current stream `wait_stream(h2d_stream)`。训练侧实现也必须建立这些依赖。**不能简单将全部 FP32 master 改成 pinned**，那既增大锁页内存，也可能暴露原有流依赖风险。

## 本轮验证与范围

- 4 份既有 GPU trace 全部离线处理，HtoD 全部关联成功；无重跑训练、无更改运行时数值代码。
- 新增离线分析脚本的测试通过：检验 CPU/GPU 同名 annotation 不混算、GPU copy 可晚于 CPU 窗口、无关窗口 copy 不被因时间重叠误归属。
- microbenchmark 语法检查及 ai_hgx_03 实测通过；5 次正式测量，BF16 bitwise equality 全部通过。该验证不等于线上 optimizer 或整模型性能/正确性验证。
- 本轮是共同训练耗时调查。推理确定性 matmul 和 RoutePack B1/B2 由并行任务分别处理。

## 后续 CPU 离线互斥归因

见 [CPU 离线验证](P1_OPTIMIZER_CPU_OFFLINE_20260912.md)。新增主线程互斥计时与 GPU/CPU 实际重叠，避免把嵌套算子、API 和 GPU 时间直接相加；另发现 fill_ 约 1.5–1.6 秒，原因待 CPU 对照验证。
