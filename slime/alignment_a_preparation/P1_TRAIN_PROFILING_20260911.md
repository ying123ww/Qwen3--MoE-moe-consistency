# P1 训练侧对照采样结果

**两组已成功完成。共同的大项是 CPU offload optimizer 与权重搬运；本次采样中对齐增量主要出现在前后向区间。** 这是一次归因采样，不替代此前正常运行的 2%–3% training-phase 重复测量，也不构成收敛或端到端性能实验。

## 协议与验证

- ai_hgx_03，Slurm 999484，8 × A100 80 GB，CPU 64，内存 1 TiB，独立 900 GiB 保护。跨 9 月 10 日至 11 日执行。
- `a-p1-train-native-20260910-01`、`a-p1-train-aligned-20260910-01` 均 `SUCCEEDED`；各 3 次 train RPC，前两次不采样，第 3 次采样 rank 0、4。共 4 份 trace，全部含 GPU 事件。
- 同一 checkpoint、同一冻结 rollout（8 samples，1218 prompt tokens + 1024 response tokens），EP=4；`lr=0`，StatelessAdam/reset optimizer states、CPU optimizer offload。执行前后向和 optimizer 计算，但不是三次真实权重更新。
- 与原 training-phase 对照口径一致：native 保留 deterministic mode 和原软件栈，只去掉对齐实现。它不是原版上游或移除所有确定性限制的 baseline。
- 显式 `--debug-train-only`：不启动推理服务器，update_weights 直接返回。因此此次成功不代表此前 CUDA IPC 同步崩溃已修复。
- 8 rank 的 audit 均通过：aligned 每 rank 有 48 层 MoE 对齐包装，native 没有对应包装；已记录的第 2、3 轮梯度范数全部有限且非零。两种实现梯度数值不同，未把这次采样宣称为新的 exact correctness gate。
- 主要临时目录使用工作目录 `.a-tmp/<run-id>`，编译缓存使用 `/dev/shm/p1-cache-<run-id>`，启动前完成写入验证；成功后自动清理并复查。GPU 已释放，保护仍在运行且未触发，读取到的较高内存约 791 GiB（非全程峰值）。

## 正确的计时口径与勘误

早期汇总脚本仅按 `P1/` 名称筛选，把 CPU 的 `user_annotation` 和 GPU 的 `gpu_user_annotation` 加在一起，导致 train RPC、wake-up、forward/backward 等数字虚高。现仅汇总 CPU `user_annotation`，并增加回归测试，验证同名 GPU annotation 不会混入 CPU 时间、重叠 kernel 的并集不会重复计算。

之前在线 aligned 的 CPU train RPC 应为 rank 0 **28.508 秒**，不是 **49.720 秒**。原 [P1 报告](P1_PROFILING_20260910.md) 和对应汇总 JSON 已修正。原始 trace、正常 training-phase / inference / online pilot 基准未受此汇总错误影响。

## 第三轮 CPU 区间壁钟时间

单位秒。包含等待，父子区间嵌套；不同 rank 并行，不能横向相加。profiler 停止、导出与 operator 汇总发生在 train RPC 标记之外，不能用带导出的 driver 等待时间代替下表。

| CPU 区间 | native rank 0 | aligned rank 0 | native rank 4 | aligned rank 4 |
|---|---:|---:|---:|---:|
| train RPC | 27.612 | 29.686 | 27.513 | 29.472 |
| wake_up | 3.233 | 3.115 | 3.247 | 3.127 |
| forward_backward | 2.226 | 3.305 | 2.226 | 3.304 |
| optimizer_step | 15.402 | 15.411 | 15.433 | 15.512 |
| actor CPU 权重备份 | 1.770 | 1.781 | 1.911 | 1.927 |
| sleep | 2.711 | 3.728 | 2.468 | 3.369 |

- optimizer 是两组共同的大项，rank 0 的差异只有约 0.010 秒；此次没有证据把对齐增量归到 optimizer。
- forward_backward 在两个代表 rank 都增加约 1.079 秒，是值得继续细分的增量候选。它还包含 Python/kernel 发射、通信与等待，不等于纯 GPU 算术时间。
- sleep 增加约 0.9–1.0 秒；只有一次采样，应先查进程组销毁/offload 的波动与 profiler 扰动，不能视为确定性实现固有成本。
- CPU 区间尚未覆盖 RPC 的所有工作，不能将未覆盖差值都解释为某一种通信。

## 拷贝定位：共同成本大，但不是对齐独有

rank 0 完整 trace 中，三类主要拷贝累计时间：

| GPU memcpy 事件累计（秒） | native | aligned |
|---|---:|---:|
| DtoH → pinned | 3.482 | 3.472 |
| pageable → HtoD | 3.083 | 3.279 |
| pinned → HtoD | 2.812 | 2.824 |
| 上述三项合计 | 9.377 | 9.576 |

这些是 GPU 事件时长求和，不能加到 CPU optimizer 区间上；拷贝与 CPU/GPU 工作可能重叠。

按 GPU memcpy 时间戳与 CPU 范围求交集（描述时间重叠，不宣称逐操作因果关联）：

| rank 0 时间窗口内的拷贝（秒） | native | aligned |
|---|---:|---:|
| wake_up：pinned HtoD | 1.454 | 1.465 |
| optimizer：pageable HtoD | 3.081 | 3.277 |
| optimizer：pinned DtoH | 1.654 | 1.645 |
| actor backup：pinned DtoH | 1.722 | 1.733 |

CPU operator self time 也显示 optimizer 中的算子值得排查：`aten::addcdiv_` 约 2.30/2.43 秒，`aten::mul` 约 1.78/1.77 秒。此处是全 trace 汇总，进一步确定 CPU optimizer 内部归属仍需调用区间关联。

trace 同时含 `cat=overhead`、名为 `Command Buffer Full` 的事件，累计约 3.458/3.484 秒。说明采样扰动不可忽略；不把该时长当作模型正常运行成本，也不把此次前后向差值直接套回原 2%–3% 结果。

## 下一次实验优先级

1. **如果目标是整体 training-phase 提速：先验证 CPU optimizer 的 staging/copy 路径。** 检查为何仍有约 3 秒 pageable HtoD，以及是否能用有界 staging buffer 减少同步和搬运；先记录实际传输字节、同步点与 CPU optimizer 调用，再做单因素改动，保持 optimizer 数学语义和内存上限。不能简单将全部参数改成 pinned 或关闭已有 correctness 机制。
2. **如果目标是降低对齐额外成本：细分 forward_backward。** 结合推理侧已识别的 batch-invariant matmul，按 layer/算子关联 CPU 发射与 GPU kernel；不能只凭相同 kernel 名把训练、推理成本混在一起，也不能只凭此次 trace 声称消除了多少开销。
3. sleep/offload 波动先做轻量边界检查；CUDA IPC 同步失败单独隔离，暂不恢复全链路 GPU profiler。RoutePack 的实现优先级应由实际路由数据搬运成本支持，目前证据尚不足以称其为首要瓶颈。

任何后续改动都先过对应数值/正确性 gate，再用不带 profiler 的原口径验证性能；不补做用户已跳过的 P0 长时间端到端重复实验。

## 证据入口

- [机器可读对照](gpu_runs/p1-train-comparison-20260911.json)，含冻结数据 SHA256、4 份 trace 汇总和 copy window 结果。
- [native](gpu_runs/a-p1-train-native-20260910-01/) / [aligned](gpu_runs/a-p1-train-aligned-20260910-01/)：`job-status.json`、`argv.json`、`runtime_env.json`、`audit-rank*.json`、`measurements-rank*.jsonl`、`profiles/train/`、`p1-copy-windows.json`、`scratch-cleaned.json`。
- [采样源码哈希](gpu_runs/p1-train-source-hashes-20260910.json)。其中汇总器随后修正 CPU/GPU annotation 混算；报告使用修正版，另存分析源码哈希。
- 相关 CPU 测试 3 项通过，语法检查和 git diff 空白检查通过。
