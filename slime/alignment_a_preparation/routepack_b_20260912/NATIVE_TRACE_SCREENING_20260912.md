# Native scheduler 既有 GPU trace 离线筛查（2026-09-12）

**结论：当前 trace 没有建立 routing skew → slow rank → step time 的证据链，不支持据此优先扩展 RoutePack 调度实现。** 当前固定短 workload 的主要账面成本是 CPU-offloaded optimizer；native 采样 rank 的 DeepEP kernel 仅约 34–35 ms。大 GPU 活动空档位于 dense collective 的 CPU 调用区间，原因仍未区分。不能据此否定代表性生产 workload 中的 RoutePack 空间。

本轮只使用 CPU，未启动 GPU 或修改在线 scheduler / 训练数值路径。

## 数据和方法

打开并解析四份 9 月 10 日原始压缩 Chrome traces：`a-p1-train-native-20260910-01` 与 `a-p1-train-aligned-20260910-01`，各 rank 0/4、rollout3。一份文件为数十万至百万事件量级，包含 CPU ops、CUDA API、GPU kernel/memcpy/memset；无 input shapes、Python stacks、实际 top-k IDs。

- 以 CPU `P1/forward_backward` / `P1/optimizer_step` annotation 确定阶段窗口；GPU 事件与窗口取交集，活动时间用跨 stream 区间并集，避免累计重叠被误当 wall time。
- 通过 correlation 关联 GPU GEMM 和 CUDA launch；只有 launch 完全位于同线程 `_GroupedLinear` / `_GroupedLinearBackward` scope 时才标为专家 grouped GEMM。其它 GEMM 保留未归属，不把 kernel 名猜成具体层。
- DeepEP 区分 layout、notify、dispatch payload、combine payload。统计 kernel 时间不能将实际搬运和设备端等待分离。
- `measurements-rank*.jsonl` 的 train_one_step 包含 F/B、optimizer、梯度清零与 metric reduction；计时前有 barrier 和 synchronize，结束也 synchronize，排除外层 wake/offload/权重备份。它不是整轮 rollout-to-update 时间。
- 已验证区间合并/裁剪、不同线程不能误归因、GPU 异步尾部处理。分析脚本与原 traces 的 SHA256 保存在 [native_trace_screening.json](native_trace_screening.json)。

## 同一次 profiled step 的数量级

| 实现 / rank | train_one_step 秒 | F/B CPU 窗口秒 | optimizer CPU 窗口秒 | DeepEP kernel 累计 ms | 可关联 grouped GEMM 累计 ms | F/B 窗口 GPU 活动并集 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native / 0 | 18.1392 | 2.2260 | 15.4019 | 34.15 | 156.71 | 425.45 |
| native / 4 | 18.1392 | 2.2259 | 15.4333 | 34.69 | 152.43 | 428.60 |
| aligned / 0 | 19.2959 | 3.3048 | 15.4114 | 85.97 | 152.61 | 796.79 |
| aligned / 4 | 19.2957 | 3.3042 | 15.5124 | 95.44 | 152.08 | 810.03 |

**不能将表中 CPU 父区间、GPU 累计和 GPU 并集相加。** grouped GEMM 是保守可归属子集；DeepEP 包含 layout/notify。优化收益取决于关键路径和跨 stream 重叠，以上均不是可回收时间。

native rank0 optimizer CPU 窗口约占记录 step 的 84.9%，F/B CPU 窗口约占 12.3%。这是该固定输入/lr=0/offload 配置下的 profiled 观察，不是生产总体占比，也不是 RoutePack 加速上限。既有 optimizer 拷贝归因另见 `../P1_OPTIMIZER_TRANSFERS_20260912.md`。

native payload 延迟：rank0/4 dispatch 中位 92.70/94.75 μs，p95 104.77/113.44 μs，max 115.90/123.77 μs；combine 中位 104.83/103.20 μs，max 125.95/168.99 μs。没有在这两个 rank 的 native DeepEP kernel 中观察到毫秒级长尾；这不排除其它 rank 或其它 workload 的等待。

## GPU 活动空档不能直接归因为 routing skew

native F/B 窗口内，GPU 活动并集约 0.425/0.429 秒，未被所记录 GPU 事件覆盖约 1.801/1.797 秒。**不是“GPU 利用率仅 19%”或“可优化掉 1.8 秒”的证据**：它包含 CPU 发起、控制、通信协调以及 profiler 开销。

最大的三段空档分别约：rank0 275/176/173 ms，rank4 224/178/173 ms。事件交叠复核显示其落在 `c10d::reduce_scatter_tensor_coalesced_` 或 `_VocabParallelLogProbEntropy` 内的 `c10d::allreduce_`，同时有嵌套 `record_param_comms`；没有与相应空档交叠的 CUDA synchronize API。不能由这些父 annotation 判断究竟是主机提交/记录开销、进程协调还是未采样 rank 的上游延迟；也不能将 `record_param_comms` 的 inclusive 时长直接称为 profiler 纯开销。原始上下文见 [native_gap_context.json](native_gap_context.json)。

F/B CPU annotation 结束后仍有一个关联 GPU 事件尾部延伸约 84 ms（native），因此 CPU 范围不是完整 GPU phase 结束时间；分析已单列该尾部，没有把它当作无事件空档。

## 为什么不能用“rank 时间几乎一样”证明平衡

原始审计表明 rank0 和 rank4 均为 **EP local rank0、EP size4**，即两组的同一 owner 位置，**不是同一 EP collective 的所有四个 receiver**。它们的相近耗时不能证明组内负载均衡。

全部8 rank的 step记录：native未启用本次torch profiler的rollout2约17.9311秒；profiled rollout3约18.1393秒。后者rank间完成耗时差仅约71 μs，但计时包含同步和collectives，快rank可能已经等待慢rank；这个差不是独立服务时间或straggler证据。aligned对应约18.1859/19.2959秒。每种仅两个执行，且重复同一固定输入，不能做可靠回归或统计显著性结论。记录见 [native_step_screening.json](native_step_screening.json)。

aligned F/B明显更长，但这同时改变了数学/通信实现；其dispatch调用从native144增加到240、layout从96增加到192（每个采样rank）。故native-vs-aligned是实现开销对照，**不是相似FLOPs下只改变routing skew的配对实验**。后续B1的短response路由与这些旧profile不是同一运行，不能拼成因果证据。

## 决策与最小补证据

1. 暂停扩展 RoutePack 在线重排；本次筛查优先暴露 optimizer 公共成本，以及 dense collective CPU 侧大区间。保持推理确定性矩阵乘优化独立，不混入RoutePack收益。
2. CPU侧仍可用代表性实际routes/native schedule检验：合法交换是否改变EP group/slot输入集合、能否改善预测H/U，同时控制padding/长度组成。当前只有短fixture，不能假造多批数据。
3. 若离线负载空间值得补测，最小GPU采样必须覆盖**同一EP组的全部4rank**；要连到全局step关键路径，最好覆盖8rank。同时保存同一批次的sample/slot/layer、真实H/U与padding、dispatch/expert/combine时间和完整step边界。
4. 首轮只保持native scheduler，采集少量匹配token/计算量但skew不同的独立batch、重复无profiler计时。观察稳定关联后再扩大至10–20代表性batch并做合法重排配对。不要重跑一批很多次充当路由多样性。

目前结论是“未证实可行性，已有证据不足以支持优先投入”，不是RoutePack理论无效，也不是已经完成No-Go。

## 补查 MoE 主机侧 wrapper

为避免只看 GPU kernel 而遗漏 CPU 等待，另外读取原始 profiler key averages：rank0/4 的 FusedDispatch forward CPU inclusive 累计为39.43/39.98 ms，FusedCombine forward为14.51/14.87 ms，FusedCombineBackward为10.29/10.37 ms，FusedDispatchBackward为11.46/10.10 ms。它们同样没有显示采样rank上的秒级主机开销。`_GroupedLinear` CPU inclusive 为66.50/69.21 ms，`_GroupedLinearBackward`为121.67/128.47 ms。父子包含/异步重叠仍不可与GPU时间相加；这些wrapper之外的route处理未被概括为零。原始项及来源哈希见 [native_moe_cpu_scopes.json](native_moe_cpu_scopes.json)。
