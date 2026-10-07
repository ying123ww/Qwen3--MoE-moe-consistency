# CPU Optimizer 与 AutoPack 项目说明

本文描述已实现的方案。AutoPack 对应当前代码中的 RoutePack。所有效果字段留空，待实际测评填写；不将合成数据的代理指标改善写成训练加速。

## 项目一：CPU Optimizer Offload 搬运与同步修复

### 问题

在 Megatron CPU optimizer offload 路径中，GPU 梯度传回 CPU，由 CPU StatelessAdam 更新 FP32 master 参数，再将更新后的参数回拷 GPU。开启 overlap 后，独立 CUDA stream 并没有在已有采样中形成有效的 H2D 与 CPU 更新流水。

排查定位到当前依赖保留的两个 upstream 已知问题：

1. **最终 FP32 master 丢失 pinned memory 属性。** 参数先通过 `pin_memory()` 放入锁页内存，后续 `clone().float()` 重新分配张量，导致实际 H2D 源变为 pageable memory。回拷 API 与后续 hook 占用主线程，压缩了下一参数 CPU 更新与回拷重叠的机会。`non_blocking=True` 本身不保证主线程立即返回。
2. **H2D 完成事件记录在错误 stream 上。** 回拷提交到 H2D stream，但完成 event 记录在 D2H stream，不能正确表达后续 GPU 操作对参数回拷完成的依赖，存在竞态。

这是 native/aligned 共同使用的训练基础设施问题，不能将其全部归为一致性额外开销。已有 trace 能确认近似串行表现，但无法独立量化各个缺陷的因果贡献。

### 解决方法与个人工作

- 沿实际运行依赖检查 FP32 参数创建、梯度 D2H、CPU update、H2D post hook 和 stream/event 调度顺序。
- 对历史 trace 做主线程互斥墙钟归因，关联 CUDA API、GPU memcpy 与子 optimizer scope，避免将嵌套算子、API 和 GPU 时间重复累加。
- 核对 upstream 修复并进行最小移植：最终 offloaded FP32 master 在 `pin_cpu_params=True` 时重新 pin；将回拷完成事件改为在 H2D stream 上记录。
- 保留其他本地依赖修改，单独提交修复，并保存可复用补丁。

修复来源：NVIDIA/Megatron-LM [#4547](https://github.com/NVIDIA/Megatron-LM/pull/4547) 和 [#3140](https://github.com/NVIDIA/Megatron-LM/pull/3140)。个人贡献是本地诊断、证据关联、上游核对与适配移植，不将 upstream 修复描述为独立发明。

### 当前完成状态

实际依赖 `hybrid_optimizer.py` 已修复，本地提交 `73a68e7`。语法及差异检查通过。尚未完成修复前后的 GPU 对照、参数更新正确性回归与真实训练性能验证。

### 效果测评（待填）

| 测评字段 | 结果 |
|---|---|
| GPU / CPU / PCIe 拓扑 | |
| 模型、dtype、offload 配置、线程数 | |
| 补丁前后提交与运行配置 | |
| 参数规模、回拷字节数及拷贝次数 | |
| 预热次数 / 测量步数 / 重复次数 | |
| 参数更新正确性：比较范围、标准与结果 | |
| 修复前 / 后 H2D API 墙钟耗时 | |
| 修复前 / 后 H2D GPU 时间 | |
| 修复前 / 后 H2D 与 CPU 数学重叠时间及比例 | |
| 修复前 / 后 optimizer 耗时 p50 / p95 | |
| 修复前 / 后训练 step 耗时 p50 / p95 | |
| optimizer 耗时下降比例 | |
| 训练吞吐提升比例及配对区间 | |
| pinned 内存占用与额外资源成本 | |

### 可用于项目介绍的表述

定位并修复 Megatron CPU optimizer offload 中的锁页内存丢失与 H2D stream 同步问题。通过源码追踪和 CPU/GPU trace 关联分析识别回拷流水受阻路径，移植 upstream 修复并完成本地集成。经对照测评，optimizer 耗时由 ______ 降至 ______，训练 step 耗时降低 ______，参数更新正确性结果为 ______。

最后一句仅在测评完成后使用。

## 项目二：AutoPack——基于实际路由负载的跨 slot 样本调度

### 问题

原生 FLOPs balancing 使用样本或 microbatch 的标量估算 FLOPs 进行分区，无法区分相似长度、相似 FLOPs 下不同 expert routing 分布。

在 MoE expert parallelism 中，同一 EP group、同一 slot 的所有 source ranks 共同参与每一层的 dispatch。即使总 assignments 相同，不同样本组合也可能使某个接收 rank 集中承担更多 expert 计算。将所有层混成单个 histogram 会掩盖逐层 straggler。

项目要验证的是：在保持样本内容、optimizer-step 边界和布局约束的前提下，利用已采集的 routes 调整样本组合，是否能降低逐层 EP 偏斜，并进一步降低训练耗时。目前负载偏斜已被观察到，但尚未完成“routing skew → slow-rank time → step time”的因果与收益验证。

### 解决方法与个人工作

**1. 建立可信的数据链。** 采集 sample 的逐层 expert assignments，导出运行时 EP 拓扑、expert owner 与原生 schedule。独立校验实际 dispatch、Routing Replay 和负载重建，区分原始 routes、末端补齐和布局 padding。

**2. 按真实 collective 定义代理目标。** 对每个 group、slot、layer、destination rank 聚合 H：

```text
H[g,s,l,r] = Σ_source_rank q Σ_sample i∈B[g,s,q] H_i[l,r]
E[g,s] = Σ_layer l (max_rank H[g,s,l,r] − mean_rank H[g,s,l,r])
目标：降低 Σ_group,slot E[g,s]
```

实际评分包含按候选布局重新计算的 terminal/padding assignments。H 是计算负载代理；同时记录 U（token/destination 对数），不将 H 或 U 直接当作通信时间或训练耗时。

**3. 使用有预算的 sampled best-improvement。** 每轮以固定 seed 随机抽取候选 sample pairs，检查合法性后计算真实 excess 增量，选择其中收益最大的严格改善交换。只更新受影响的 collective；相同收益按固定 cell/sample 位置决定，保证可复现。

**4. 保留调度约束与回退。** 交换限制在同一 optimizer step 的不同 slot，保持完整 sample 身份、rank/slot 网格和每个 bin 的样本数；不改 tokens、routes、reward 或 loss mask。限制 padded token 容量，拒绝增加 padding 的交换。没有合法改善时保留原布局。当前交换版本未单独增加每个 bin 的 FLOPs 上限；真实训练接入前需检查 FLOPs 分布是否退化。

**5. 加入可比较的基线与成本核算。** 保留 Native 和相同合法约束、相同接受交换次数的 Random-Swap。提供候选数、迭代数、交换次数预算及协作式时间截止；分别测量准备、搜索、完整校验和总 CPU wall time。固定计数预算下验证 schedule 与 swap log 确定性。wall-time 截止受机器负载影响，不承诺相同停止点或整调用硬时限。

### 当前完成状态

保留算法为 sampled-best Swap，已删除旧 first-improvement 和构造式 Packing 实验代码。离线候选生成器、Random 对照、CPU benchmark 和确定性测试已实现，本地提交 `022e9b0`。

当前只生成离线候选 schedule；尚未接入实际训练调度，未验证交换后的 forward/backward/参数更新一致性、slow-rank 改善或 step time 收益。现有合成实验只用于筛选实现，不作为生产效果。此前 B1 前向一致性通过不等于交换后训练正确性通过。

### 效果测评（待填）

| 测评字段 | 结果 |
|---|---|
| 模型、硬件、EP/DP 拓扑、软件版本 | |
| 数据来源、独立 batch 数、sample 数及长度分布 | |
| 每步 group / slot / bin 数及 padding 比例 | |
| 原生 FLOPs/token balancing 配置 | |
| 搜索 seed 数、候选/迭代/交换预算 | |
| Native / Random-Swap / AutoPack excess | |
| Native / Random-Swap / AutoPack U 与通信偏斜指标 | |
| 随机对照交换次数匹配情况及多 seed 分布 | |
| 调度前后每 bin FLOPs/token 分布与 padding | |
| scheduler 搜索 CPU 时间 p50 / p95 | |
| scheduler 完整 CPU 时间 p50 / p95 | |
| 新增路由采集、元数据传输与存储成本 | |
| 交换后 forward / backward / 参数更新正确性 | |
| 相近 FLOPs/token 下 routing skew 与 slow-rank 时间关系 | |
| Native / Random-Swap / AutoPack 最慢 rank 耗时 | |
| Native / Random-Swap / AutoPack step time p50 / p95 | |
| 扣除新增调度与采集成本后的净收益及配对区间 | |
| 训练吞吐及在线 RL iteration time 变化 | |
| 峰值显存 / 主存变化及失败回退率 | |

### 可用于项目介绍的表述

设计并实现面向 MoE 训练的路由感知样本调度原型 AutoPack，在原生布局上进行跨 slot 的完整样本交换。基于实际 EP 拓扑构建逐 collective、逐层负载目标，采用固定 seed 的有预算候选择优搜索，并提供 Random-Swap 对照、调度成本核算与确定性验证。在 ______ 工作负载上，相比 Native 和 Random-Swap，逐层 EP excess 分别降低 ______ 和 ______；scheduler CPU 开销为 ______，扣除新增成本后的训练吞吐提升 ______。

最后一句仅在真实测评完成后使用；测评前应将项目阶段注明为“离线调度原型”。
