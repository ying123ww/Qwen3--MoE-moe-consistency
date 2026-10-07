# P1 瓶颈采样阶段记录（2026-09-10）

**最新状态：独立推理 baseline/aligned 对照采样已成功完成（见文末更新）；独立训练侧对照也已完成，见 [训练对照报告](P1_TRAIN_PROFILING_20260911.md)；在线权重同步问题仍未解决。** 当前先排查 CPU offload / optimizer / 权重搬运，以及 decode 的 CPU 发射与调度间隙。不能据此解释全部 alignment 增量，也不能发布加速比例。用户已选择跳过 P0 长时间重复测量，本次没有重做 P0。

## 实验与证据

- 节点 ai_hgx_03，Slurm 999484，8 × A100 80 GB，64 CPU，1 TiB 内存；独立 900 GiB cgroup 保护持续运行。
- online RL，3 轮，第 3 轮采样；沿用之前 lr=1e-6、StatelessAdam / reset optimizer states、CPU optimizer offload 配置。不是标准持续状态 Adam 收敛实验。
- SGLang：8 rank，各自 prefill（EXTEND）与前 8 个 decode step，CPU+GPU，不记录 stack/shapes/memory。
- Megatron：rank 0、4 的第三轮 train RPC，CPU+CUDA；新增 wake_up、数据获取、forward_backward、optimizer_step、CPU actor backup、sleep 标记。
- 权重同步也接入了 profiler 和实际 colocate Tensor updater 的打包/IPC 标记，但此次同步失败，**没有有效同步 trace**。
- 运行目录：[aligned-02](gpu_runs/a-p1-aligned-20260910-02/)。已解析 16 份推理和 2 份训练 gzip Chrome trace，全部包含 GPU 事件。
- 机器可读结果：[p1-summary.json](gpu_runs/a-p1-aligned-20260910-02/p1-summary.json)。原始文件在该目录 `profiles/`，日志 `job.log`，训练 operator 汇总和元数据位于 `profiles/train/`。
- 采样时源码哈希：[p1-source-hashes-20260910-02.json](gpu_runs/p1-source-hashes-20260910-02.json)。其后 launcher 增加了根盘空间检查，尚未 GPU 重跑该版本。

## 已观察到的候选瓶颈

**勘误（训练对照复核）：早期脚本把同名 `user_annotation` 与 `gpu_user_annotation` 累加，误报了 CPU 阶段时间。现仅统计 `user_annotation`，下表和原汇总 JSON 已修正；旧版本 49.720 秒 train RPC 等数字作废。原始 trace 与此前正常性能基准不受影响。**

下表是 CPU record_function 区间壁钟时间，单位秒。包含等待，不能当纯计算时间。父区间嵌套子区间；不同 rank 并行，不能相加。profiler 停止和文件导出另外扰动 driver 耗时。

| 区间 | rank 0 | rank 4 |
|---|---:|---:|
| train RPC 标记范围 | 28.508 | 28.475 |
| wake_up | 2.899 | 2.909 |
| forward_backward（累计） | 3.218 | 3.223 |
| optimizer_step（累计） | 15.749 | 15.903 |
| actor CPU 权重备份 | 1.804 | 1.946 |
| sleep | 2.784 | 2.611 |

1. **先细分 CPU offload optimizer 和搬运。** rank 0 GPU memcpy 事件累计：DtoH pinned 3.358 秒、HtoD pageable 3.353 秒、HtoD pinned 2.822 秒，共约 9.533 秒。它们可能与 kernel、CPU 区间重叠，不能把它们加到 optimizer 15.749 秒上，也不能认为全部都可消除。需要按时间戳把 copy 关联到 wake_up、optimizer 和 backup，再做单因素改动；已有禁用额外 CPU backup 的选项此前已启用，不重复当新优化。
2. **decode 需要同时看 host 调度，不能只盯 MoE kernel。** rank 0 的 8 decode 步 GPU 事件观察窗口 1.491 秒，已记录 GPU 事件区间并集仅 0.279 秒；8 rank 并集范围 0.279–0.472 秒，窗口 1.489–1.539 秒。这是单进程可见事件的覆盖率，不能直接称整卡利用率。空隙可能包括 CPU 发射、调度/等待和 profiler 扰动，尚未定因。
3. **decode kernel 优化候选。** rank 0 `bmm_kernel_persistent` 累计 79.54 ms（1160 次）、DeepEP `notify_dispatch` 40.05 ms（384 次）、`fused_moe_kernel` 26.80 ms（768 次）、`_fwd_kernel_unified` 12.41 ms（384 次）。这是短窗口事件求和，baseline 缺失，不能说这些就是 23.48% 推理增量的来源。
4. **prefill 通信等待异常暂不作为优化结论。** rank 0 `cached_notify_combine` 累计约 3.401 秒，但各 rank 分布非常不均，且正好处于分阶段 profiler 启停与导出流程；跨 DP 的采样启停可能扰动同步。需用连续短窗口或独立推理进程采样复核。

rank 0 train RPC 的 28.508 秒与上述子区间之和不相等，仍有未标记区间和等待；本次没有完成关键路径全覆盖。不能把本表直接换算成正常运行的节省秒数。

## 失败记录与当前阻塞

- `a-p1-aligned-20260910-01`：完成前两轮，第三轮入口错误地对 SGLang `start_profile` 纯文本成功响应调用 `.json()`。源码确认返回 `Start profiling.\n`；已改为检查 HTTP 状态，并添加对应回归测试通过。属于新增采样代码错误。
- `a-p1-aligned-20260910-02`：推理和训练采样导出完成，第三轮同步时 SGLang 出现 `free(): invalid next size (fast)` / `Fatal Python error: Aborted`，栈在 CUDA IPC 张量反序列化，其他 rank 随后 Gloo 连接断开。不能认定同步成功，也不能认定根因已经定位。与 profiler/CUDA IPC 生命周期的相互作用是待验证假设。
- 已通过 Ray stop API 停止该失效作业；顺序控制器因失败退出，**baseline 未提交**。最后 GPU compute query 为空，Slurm allocation 和内存保护保留。
- 同时发现根盘 `/dev/sda3` 普通用户可用空间为 **0，100%**；shell here-document 已报 `No space left on device`。`/var/crash` 约 133 GiB，为 root 所有、2025–2026 历史内核转储，不是本项目可直接删除的文件。没有修改这些系统文件。根盘满是否与 IPC 崩溃有关尚无证据。
- `/dev/shm` 尚有约 755 GiB，说明不是共享内存盘满。采样中保护读数约 798 GiB，未触发 900 GiB 阈值；该读数不是连续采样的全程最大值。

## 恢复后的执行顺序

1. 请管理员恢复节点根盘可写空间。launcher 现增加根盘可用空间至少 1 GiB 的硬检查；这是最低启动门槛，不是长期容量建议。
2. **先隔离 profiler 与 IPC 失败**：保留已有成功 trace，使用独立推理进程采集 baseline（及相同采样条件的 aligned 复核），避免在同一进程 profiler 停止后立刻执行在线 IPC 权重同步。训练采样也可用已有固定 rollout 路径单独采集；这些只能归因各阶段，不能冒充 online E2E。
3. 同步部分先使用 CPU 范围/轻量计时，隔离是否仅 GPU profiler 开启后触发崩溃；一次只改一个因素，不直接改动模型数值路径。复现前不自动重启原来的全链路采样控制器。
4. 两组证据齐全后，再给出候选优化优先级，先验证 CPU offload/copy 与 decode host 间隙，随后决定 RoutePack 的路由数据搬运/序列化优化是否有实际收益空间。

本次相关 CPU 检查：`test_p1_profile.py` 与 `test_launch.py` 共 2 项通过，改动文件语法检查通过。P1 尚未完成 baseline 对照和同步归因。现有正常运行的 training-phase、独立 inference、online pilot 数字保持原报告口径，不用本次 profiler 耗时替换。


## 更新：迁移临时目录后的独立推理对照已完成（22:30 UTC）

用户释放 GPU 后，运行 `run_p1_inference_pair.py`，两组均成功退出（returncode=0）。这证明此前“根盘满所以不能继续所有 GPU 实验”的判断过于绝对：独立推理可以通过迁移主要临时路径继续运行。根盘仍满，系统磁盘问题本身未修复；在线同步崩溃也未因此被认定修复。

### 路径与生命周期

- TMPDIR/TMP/TEMP：`/public/home/yingxy2024/moe-consistency/.a-tmp/<run-id>`，启动前实际写入探测通过。
- TVM FFI、Triton、TorchInductor、CUDA、XDG 缓存：`/dev/shm/p1-cache-<run-id>/` 各自子目录，启动前实际写入探测通过。
- 两组成功退出后，以上本次专用临时目录与缓存自动删除，已复查不存在。日志与 trace 保存在项目目录。
- core dump 上限设为 0；代理清空；仍沿用独立 900 GiB guard，未触发。导出时一次保护读数约 76 GiB，非全程峰值。
- `/tmp` 中已有 CUDA wrapper 和统一启动锁仍被引用；本次不是“完全不访问 /tmp”，而是将主要可写临时数据迁出。未改动正在运行的系统 Ray 服务和保护程序。
- 最终 GPU compute query 为空。此轮没有运行训练或权重同步。

### 对照协议和证据

baseline/aligned 各独立启动一次；每组 3 次预热 + 1 次完整 `Engine.generate` CPU/GPU 采样。128-token response，8 prompts，共 1024 输出 token；三批预热及采样批全部通过 token 数和有限 logprob 检查，cached tokens 全为 0。连续采样，不在 prefill/decode 边界切换 profiler。两组是同 prompt/长度，生成内容可不同；不是冻结相同 route 的 kernel 微基准。

- [baseline 原始目录](gpu_runs/a-p1-inference-baseline-20260910-01/)、[baseline 汇总](gpu_runs/a-p1-inference-baseline-20260910-01/p1-summary.json)
- [aligned 原始目录](gpu_runs/a-p1-inference-aligned-20260910-01/)、[aligned 汇总](gpu_runs/a-p1-inference-aligned-20260910-01/p1-summary.json)
- 每组 8 份 gzip trace，所有文件都包含 GPU 事件；两组采样结果不与前次在线分阶段采样直接计算差值。
- [运行源码哈希](gpu_runs/p1-inference-source-hashes-20260910.json)、[完成状态](gpu_runs/p1-inference-20260910-state.json)。

### 当前可以支持的归因候选

以下为 **rank 0 的一次完整生成采样**；累计 GPU 事件时间不是关键路径时间，不跨 rank 相加。

| 指标/事件 | baseline | aligned |
|---|---:|---:|
| 已记录 GPU 事件数量（含 memcpy/memset） | 190,028 | 363,838 |
| GPU 事件区间并集（秒） | 3.484 | 5.406 |
| GPU 事件观察窗口（秒） | 16.931 | 22.190 |
| DeepEP notify_dispatch 累计（秒，均 6,240 次） | 1.153 | 0.939 |
| fused_moe_kernel 累计（秒，均 12,480 次） | 0.354 | 0.530 |
| ReduceScatter BF16 累计（秒，均 6,240 次） | 0.340 | 0.374 |
| DeepEP cached_notify_combine 累计（秒，均 6,240 次） | 0.316 | 0.295 |
| AllGather 累计（秒，均 6,370 次） | 0.212 | 0.255 |

aligned 的首位 kernel 是 `matmul_kernel_persistent`，18,850 次、累计 1.280 秒。源码位于 `experiments/sglang-main-sm80/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py` 的 `_matmul_persistent_triton`，由 batch-invariant matmul 路径调用。**不能只凭 kernel 名把全部成本归到 Router**，还需按调用关联区分 dense projection、router、LM head 等。

优先顺序据此调整为：

1. 检查确定性矩阵乘路径的调用分布，以及增加的 GPU 事件/host 发射开销。两组观察窗口明显长于本进程记录的 GPU 事件并集；这只能提示 host/调度/等待值得查，不能直接将差额都认作 CPU 时间或整卡 idle。
2. 对 fused MoE 和 unified attention 做单因素复核，保持数值正确性 gate。此次同时改变多项推理配置，只能提出候选，不能声称因果已分离。
3. DeepEP dispatch/notify 仍有绝对成本，但本次 notify_dispatch 没有随 aligned 变慢，不能将整体推理增量统称为 DeepEP 通信开销。
4. 训练侧保持前文 CPU offload/optimizer/copy 候选，随后补训练对照；同步部分继续隔离 CUDA IPC 崩溃，不直接恢复原全链路 profiler。

profiler 批次 Engine.generate 时间为 baseline 16.944 秒、aligned 22.201 秒，**只作审计，不发布为新的性能开销比例**。正常性能结论仍引用之前独立推理重复测量。当前 CPU affinity 只覆盖一个 NUMA 节点，SGLang 对 GPU 4–7 跳过 NUMA 绑定，属于本次资源条件限制。
