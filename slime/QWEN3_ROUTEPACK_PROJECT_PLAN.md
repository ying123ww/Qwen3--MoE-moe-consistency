# Qwen3 MoE 训推一致性与 RoutePack 后续项目规划

日期：2026-09-09  
状态：A 阶段已完成在线两轮正确性复现、训练/推理独立性能对照及一组短程在线端到端计时。RoutePack 尚未实现，未产生收敛结果。下文保留最初设计与历史进展，最新实测见文末。  
定位：优先服务于 RL Infra / 大模型训练系统岗位，形成正确性、性能代价、调度优化和可复现实验的完整证据链。

2026-09-09 A 阶段准备进展：已完成不使用 GPU 的基线盘点、历史 dump CPU 复算、源码开关审计和配置对照，见 [A 阶段准备说明](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/README.md)。两轮历史 logprob 均重新验证为零差异；当前模型尚未重跑，性能和稳定性实验未启动。审计确认 v89 使用 StatelessAdam 与 optimizer state reset，因此本文历史“Adam 更新”应理解为该特定配置下的更新，标准有状态 Adam 的长程验证仍属后续任务。

## 1. 项目路线与最终交付

建议保留已经完成的训推一致性项目，在它之上验证 RoutePack 的可行性。先测清一致性的工程代价，再判断路由信息能否帮助训练调度；有真实收益后，才投入完整优化与更大规模验证。RL 收敛实验作为独立的可选研究，不作为前两个项目成立的前提。

| 工作线 | 核心问题 | 最小交付 | 优先级 |
| --- | --- | --- | --- |
| A：一致性项目补全 | 精确一致性需要多少时间和显存？能否持续运行？ | 原生/对齐路径性能表、扩大范围的正确性结果 | 必做 |
| B：RoutePack 可行性 | 实际路由偏斜是否造成慢 rank？合法调度是否能改善？ | 拓扑清单、负载与耗时数据、Go/No-Go 报告 | 必做 |
| C：RoutePack 实现 | 在相同训练语义下，路由感知重排能否缩短耗时？ | 有界交换算法、回归测试、受控 benchmark | B 通过后做 |
| D：学习效果验证 | 一致性是否改善任务学习？调度是否维持质量？ | 多 seed、固定评测集的学习曲线 | 可选 |

最终最好能回答：**解决了什么问题、在哪些条件下成立、付出了多少代价、是否改善实际执行效率。** 不预设吞吐一定提高，也不预设收敛一定更好。

## 2. 已有成果与证据边界

依据 [Qwen3 交接报告](/public/home/yingxy2024/moe-consistency/slime/QWEN3_ALIGNMENT_HANDOFF_V66.md:937)。虽然文件名为 V66，正文最终状态已经到 v89。

- 在报告所列 BF16、确定性、EP=4 配置下，已完成 48 层输出和 final RMSNorm 的逐元素精确验证。
- 已验证 temperature=0 与 0.6 的 target-token logprob 对齐。
- v89 完成真实在线 rollout → train、非零 advantage、非零学习率、Adam 更新和完整权重同步；两轮分别有 8 samples、1,024 个 target token，logprob 零差异，第二轮覆盖更新后的模型。
- 报告记录相关单测 47 项通过；此为历史结果，本次没有重新运行。
- 未证明长时间训练稳定性、任务质量或吞吐改善，也未证明跨框架 backward 一致性。

**衔接关系：**已有项目提供数值诊断和正确性基础；RoutePack 尝试将路由信息用于执行调度。但是，v89 的 logprob 结果不等于已经验证了新调度布局下的 Routing Replay，也不证明历史 rollout trace 对任意后续训练均准确。两条路径的兼容性需要新增 gate。

冻结 v89 配置和结果引用，不覆盖其基线，不延续无边界的诊断版本扩张。新实验采用独立的 `align-perf-*`、`routepack-profile-*`、`routepack-bench-*` 标识。

## 3. 已核对的代码基础

本次只读检查的本地 slime HEAD：`fd5f0573405d7153168c430c51e94a249c700d78`。工作区还存在用户已有的未跟踪文件；本规划不改动它们。正式实验还需要冻结 Megatron、SGLang、DeepEP 的版本与实际加载路径。

| 本地证据 | 已确认内容 | 对规划的影响 |
| --- | --- | --- |
| [dp_schedule.py](/public/home/yingxy2024/moe-consistency/slime/slime/utils/dp_schedule.py:84) | 按 rollout ID 划分 optimizer step；先打包、对齐 microbatch 数，再分配到 DP rank | RoutePack 必须在原 step 边界内工作 |
| [flops_utils.py](/public/home/yingxy2024/moe-consistency/slime/slime/utils/flops_utils.py) | MoE FLOPs 使用 FFN 大小与 top-k 等结构参数，没有实际 expert ID 输入 | 路由负载可能提供额外的预测信息，收益尚待验证 |
| [rollout.py](/public/home/yingxy2024/moe-consistency/slime/slime/ray/rollout.py:862) | rollout routing tensor 被校验、按调度分区传给训练端 | 有复用数据的入口，但采集和传输成本需要测量 |
| [actor.py](/public/home/yingxy2024/moe-consistency/slime/slime/backends/megatron_utils/actor.py:106) | 调度配置的 dp_size 来自 dense data-parallel group，当前字典未提供完整 EP 拓扑 | 不能把它直接当成独立 EP 副本数 |
| [cp_utils.py](/public/home/yingxy2024/moe-consistency/slime/slime/backends/megatron_utils/cp_utils.py:362) | replay 断言 trace 行数为 token 数减一，并补末端及布局 padding，再进行 CP/TP 切分 | 原始 histogram 与实际训练 dispatch 计数需要核对 |
| [moe_layer.py](/public/home/yingxy2024/moe-consistency/Megatron-LM/megatron/core/transformer/moe/moe_layer.py:103) | 当前默认 local expert 索引为连续区间 | 可支持默认映射，但必须记录并校验实际 layout |
| [DeepEP buffer.py](/public/home/yingxy2024/moe-consistency/DeepEP/deep_ep/buffer.py:293) | API 分别提供 per-rank token、per-expert token 和 token-in-rank 信息 | 计算负载与通信计数应分开建模 |

上游 [Slime 调度源码](https://github.com/THUDM/slime/blob/main/slime/utils/dp_schedule.py)也提供了对应的调度背景。实现以冻结的本地代码为准，不能把可变的 main 链接当作实验版本。本文不主张 RoutePack 是“首次”使用路由 trace 调度；创新性需要另做相关工作检索。

## 4. 对原 RoutePack 方案的关键修订

### 4.1 调度单位必须覆盖同一次 EP 通信

原方案中单个 microbatch 的 histogram 可作为样本特征，但通常不足以代表一次 DeepEP collective 的完整负载。Megatron 将 dense DP、expert model parallel 与 expert data parallel 分别组织；上游 [parallel_state.py](https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/parallel_state.py)明确区分这些 group。

以“训练 world size=8、dense TP=PP=CP=1、expert TP=1、EP=4”为例，通常有 8 个 dense DP 数据分区与 2 组 expert 副本，不能把调度器的 8 个 DP rank 理解成 8 个完整 EP 副本。这个例子是拓扑推导，真实 rank 成员必须从运行时导出，不能按 GPU 编号猜测，也不能混用 SGLang 的 DP 定义。

V1 中令 `B[g,s,q]` 表示 EP group `g`、microbatch 执行槽位 `s`、源 rank `q` 的本地样本集合。成本应至少聚合这一组共同参与通信的输入：

```text
同一 EP group、同一执行槽位
  source rank q0: local microbatch B0 ─┐
  source rank q1: local microbatch B1 ─┤
  source rank q2: local microbatch B2 ─┼─> 一次 EP dispatch 的接收端负载
  source rank q3: local microbatch B3 ─┘
```

先固定 PP=CP=dense TP=expert TP=1、VPP=1，并核对运行时条件；这些条件未满足时不能直接使用简化槽位模型。暂不承诺 PP/VPP 支持。

**重要反例：**如果一个 step 对某个 EP group 只有一个槽位，且它包含的全体样本不变，那么只在该槽位的源 rank 之间交换样本，不会改变接收端 expert 计算总量。可能改变源端和链路流量，但不能宣称降低了接收端计算偏斜。

因此 No-Go 不能仅依据“每个本地 microbatch 都是 singleton”。即使是 singleton，跨槽位或跨独立 EP group 的组合仍可能有优化空间；要检查的是可改变的 collective 输入集合。

### 4.2 计算负载与通信负载分别统计

对于样本 `i`，路由为 `R_i[t,l,k]`，目标 expert 的 EP 本地 owner 为 `owner(e)`。

- `H_i[l,r]`：落到 rank `r` 的 **expert assignments** 数。同一 token 在该 rank 命中两个 expert，应计两次，用于计算负载代理。
- `U_i[l,r]`：该 token 是否至少有一个 expert 位于 rank `r`，每个 token 对同一目标 rank 最多计一次，用于 token 发送量代理。
- 诊断时另保留 per-expert 计数；相同 rank 总量下，各 expert 的 GEMM shape 仍可能不同。

`H` 和 `U` 均不直接等于实际通信字节。还要考虑源端位置、本地/远端路径、metadata、padding 和实现细节。以 DeepEP 实际 dispatch layout 和 profiler 校准，不能把 `H` 的下降直接写成通信量或通信耗时下降。

在 48 层、EP=4 下，单份 int32 `H` 为 768 bytes；`H+U` 为 1,536 bytes，不包含容器及其他 metadata。累加建议使用 int64。scheduler 消费摘要；replay 所需原始 trace 仍由现有链路持有，不能提前释放。

### 4.3 成本模型需要量纲一致与留出验证

定义槽位级接收负载，先在已支持的简单拓扑下使用：

```text
H[g,s,l,r] = Σ源rank q Σ样本 i∈B[g,s,q] H_i[l,r] + 实际布局补齐贡献
E[g,s] = Σ层 l (max_r H[g,s,l,r] - mean_r H[g,s,l,r])
```

`E` 是失衡代理，不是耗时定律。原方案 `F + λE` 中 `F` 是 FLOPs、`E` 是 assignment 数，不能随意使用 `λ=1`。另一个问题是：固定样本集合时，所有 microbatch 的可加 FLOPs 总和通常不因交换改变，因此简单最小化总 FLOPs 并不能平衡关键路径。

第一版采用少量可解释特征，分别拟合：

```text
baseline:  predicted_ms = a + b × max_source_FLOPs
route:     predicted_ms = a + b × max_source_FLOPs + c × E
optional:  route 模型 + 接收/发送 token 偏斜特征
```

系数在校准集上拟合并固定；按 rollout/运行批次划分校准集和留出集，禁止用测试耗时调参。同一 trace 的不同重排与重复测量不能跨集合泄漏。考虑非负系数和简单模型，避免第一版引入复杂学习器。

调度目标 `J(schedule)` 使用组与槽位级预测成本；先比较各 group 的预计总时间，并按实测同步关系决定是否采用跨组最大值或逐槽位最大值。它始终是代理目标，最终以端到端实测判定。

### 4.4 路由元数据有明确的覆盖范围

- 必须统计训练实际执行的 prompt 与 response token，不能只统计 loss mask 非零的 response。
- 原始 trace 是 `tokens-1` 行；缺少的末端一行和 replay padding 不能当成不存在。
- 对照训练端实际使用的路由，区分真实 trace、合成末端行、布局 padding，防止 CP/TP 重复计数。
- 要确认路由是逻辑 expert ID，且与当前训练权重版本、layer ID、expert layout 匹配。
- trace 缺失、越界、层不完整或 topology 不支持时，显式报错，或带原因回退为原生调度；不能填零后继续声称采用了 routing-aware scheduling。

### 4.5 保留 token 上限不等于保证不会 OOM

本地原生动态打包允许超长单样本独占并超过 token cap；`balance_by_flops` 路径也不保证每个 bin 的 token cap。RoutePack 必须明确超长样本策略：拒绝该实验输入，或保持其独占并单独标记。

即使满足 token cap，expert 接收缓存、padding、激活与通信 buffer 也可能改变显存。因此只能写“保留 token-cap 约束并测量显存”，不能直接承诺 memory-safe。

## 5. 工作线 A：补全一致性项目，先拿到可信的工程结果

### A1. 冻结并复现基线

交付 `manifest.json`：仓库 commit、dirty diff 摘要、模型与 tokenizer 标识、GPU 型号/数量、CUDA/PyTorch/NCCL、实际 Python 环境与模块路径、并行组、所有数值相关开关、数据 hash、seed、启动命令。

先复现现有 v89：两轮各 1,024 个 target token、非零梯度、更新后第二轮 logprob 零差异。检查 optimizer step 与参数变化证据；HTTP 200 或非零学习率本身不能单独替代参数更新验证。

### A2. 一致性开销比较

| 配置 | 数值路径 | Scheduler | 目的 |
| --- | --- | --- | --- |
| A-native | 冻结版本中可运行的原生路径 | 原生 | 性能与数值偏差基线 |
| A-aligned | 当前一致性路径 | 与 A-native 相同 | 测量整个对齐方案的代价 |

同一模型、硬件、样本和并行配置，关闭诊断 tensor dump。优先固定 rollout 做训练 replay；另用在线实验测 rollout 与完整 iteration。若原生路径在该拓扑不可运行，应明确报告并定义替代基线，不可把失败运行当作吞吐 0 参与加速比。

记录 rollout response tokens/s、train forward/backward 时间、optimizer 时间、weight sync 时间、端到端 iteration 时间、峰值显存，以及 train/rollout logprob 差异。由结果决定是否值得额外做算子开销消融；无需一开始逐个开关穷举。

### A3. 扩大验证范围

在 v89 之后一次只改变一个变量：先从 response length 128 扩至 512，再视显存与时长选择 1,024；之后测 batch、可变长度/EOS、更多在线更新。长度是候选档位，不承诺当前硬件均能运行。最小 8-sample 配置保留为 correctness gate，不把它当作吞吐代表。

预计先做 20 个真实 optimizer step 的稳定性 pilot，再按实际 GPU-hour 决定是否扩至 100 步。零差异仅对实际完成的样本与配置宣称；失败时记录首个分叉，不静默放宽原有 exact gate。

**A 的完成标准：**一份可复现的性能开销表，以及明确说明扩大了哪些范围的稳定性记录。即使吞吐下降，也可以形成“以 X% 开销实现严格一致”的工程结果。

## 6. 工作线 B：RoutePack Profiler 与 Go/No-Go

建议先投入 2–3 个工作日做 pilot；遇到 trace 或运行环境问题可延长一次排查，但须先记录新增预算。此阶段只增加观测和离线候选分析，不改在线调度决策。

### B1. 首先证明 trace 与拓扑可信

导出每个 global rank 的 dense DP、EP、expert DP、TP、CP、PP 成员关系，以及 expert owner layout。在一份固定 rollout 上启用 replay，校验送到训练 router/dispatch 的 route ID，按 `(sample_id, token_position, layer_id)` 对齐。

比较“原生 schedule + replay”与已知一致性基线。若 replay 与当前 alignment wrapper 不兼容，先独立修复和验证；不能把这个集成问题隐藏在 RoutePack benchmark 中。

### B2. Profiler 数据契约

建议每个运行产出以下文件，路径均位于独立实验目录：

| 文件 | 关键字段 |
| --- | --- |
| `manifest.json` | 版本、硬件、配置、数据 hash、seed、run ID |
| `topology.json` | 实际 process group 成员、expert layout、支持范围 |
| `samples.jsonl` | sample/rollout/group ID、长度、有效 loss token、trace hash、摘要引用 |
| `schedule.json` | optimizer step → group → slot → source rank → sample IDs |
| `route_loads.npz` | H、U、训练补齐统计、必要的 per-expert 诊断计数 |
| `timings.jsonl` | group/slot/layer/phase、测量方式、耗时、padding、峰值显存 |
| `go_nogo.md` | 数据质量、瓶颈证据、交换空间、预计收益、继续/停止决定 |

每个样本要有稳定 ID；`rollout_id` 与 GRPO reward group 不一定是同一个概念，都要保留。日志应能追溯到准确版本与 route trace，不能只记一个平均 imbalance。

### B3. 测量方法

1. 采集自然产生的真实 rollout，覆盖不同长度与 prompt；先检查原生调度拆分后有多少槽位、多样本 bin、可交换样本及互补路由。
2. 至少覆盖多份独立 rollout；初始目标为 10–20 个代表性 global batch，数据足够后再扩充。不是把同一小 batch 重放很多次就当作 workload 多样性。
3. 用低侵入训练 timers 采集整步耗时；选少量 step 用 Nsight Systems/CUDA events 检查 DeepEP、expert GEMM、等待和重叠。event 必须覆盖相关 stream 的实际依赖。
4. CUDA 是异步执行，禁止仅用 CPU 函数返回时间代替 GPU 完成时间。高侵入逐层同步 profile 和正常性能 benchmark 分开。
5. 同时观察组与槽位级 max/mean EP load、EP excess、每 expert shape、最慢 rank、FLOPs、prompt/response 长度、padding 和实际耗时。
6. 在相近 FLOPs/长度区间比较路由偏斜；在留出 rollout 上比较 FLOPs-only 与 route-aware 的 Spearman 和耗时预测误差。相关性只作为线索。
7. 对同一 batch 做少量合法离线重排，计算组与槽位级目标，再做小规模配对 replay，检查代理改善是否真的缩短耗时。预测相关性提高不能替代这个干预验证。

### B4. 继续或停止的条件

以下数值是建议的预先决策门槛，不是已有结果；可在正式 benchmark 前根据 pilot 噪声调整一次并记录。

| Gate | 继续条件 | 不通过时的动作 |
| --- | --- | --- |
| 数据可信 | trace 可对齐，padding 可解释，拓扑和 owner 已确认 | 暂停优化，先修数据契约 |
| 调度自由度 | 有足够的合法候选，能够改变 EP group/slot 的输入集合 | 限制为离线负结果；不为制造收益随意改任务 |
| 瓶颈相关 | 在控制长度/FLOPs 后仍有证据，实际时间线可看到 expert/通信等待 | 若瓶颈在别处，转向 A 的主要开销 |
| 预测有效 | 留出集预测误差有稳定改善；可参考相对降低 10% | 调整简单特征一次；仍无效则停止 scheduler 开发 |
| 可干预 | 同 batch 的合法重排在重复试验中有正向变化，且优于随机合法交换 | 只有 histogram 更好则不能 Go |
| 工程价值 | 预计训练耗时改善有望超过噪声与新增成本；可先以 3% 为参考 | 小于阈值可保留研究结果，但不投入大规模集成 |

若 DeepEP/expert 在总时间占比很小，先估计收益上限；不能将多个重叠 kernel 的持续时间简单相加成“可消除时间”。若路由普遍均匀，或所有样本都偏向相同 experts、缺少互补性，应明确记录重排空间有限。

**No-Go 也有交付：**负载观测工具、可信 trace、反例与瓶颈报告，回到一致性开销优化，不强行把 RoutePack 做成加速项目。

## 7. 工作线 C：最小 RoutePack 实现

### C1. 支持范围与接口

V1 固定 Qwen3-30B-A3B、BF16、normal DeepEP、当前已验证的确定性路径、Routing Replay，PP=CP=dense TP=expert TP=1、VPP=1、静态 expert placement。禁用尚未核对的 token dropping/capacity 政策；不支持的配置显式拒绝或有记录地回退。支持范围以运行时验证为准。

候选摘要结构：

```text
RouteSummary
  sample_id / rollout_id / reward_group_id
  model_version / trace_hash / expert_layout_version
  token_count / traced_token_count / valid_loss_token_count
  assignment_histogram[layer, ep_owner]
  unique_destination_histogram[layer, ep_owner]
```

V1 只消费 histogram，不新增一次 router forward。摘要构造、拷贝、序列化、传输和调度耗时全部计入开销；“已有 trace”不意味着免费。

### C2. 算法：在合法 schedule 上做有界交换

保留原生 pack、K 对齐和 rank 分配，先得到完整 `group/slot/source rank` 网格。然后只在同一 optimizer step 内尝试样本交换，增量更新受影响的 group/slot 成本。

接受交换前必须通过长度、padding、静态样本数（若适用）、step/group 语义和容量检查。只接受 `J_new < J_old - epsilon` 的候选，使用固定 tie-break 和 seed；设候选数、迭代数及 wall-time 上限，超预算返回最佳已验证 schedule。

实现顺序：

1. **RoutePack-A：**bin 保持原有 rank/slot 位置，交换 bin 内的样本，目标按真实 EP collective 聚合。
2. **RoutePack-AB：**A 通过后，增加完整 bin 在 rank/slot 网格中的重排，保持每个 rank 的槽位数不变。

这比直接把单个 bin 的成本塞给 DP partitioner 更符合 EP group 的耦合关系。跨源 rank、跨槽位或跨 expert 副本的变动都必须重算受影响的 collective 成本。

V1 不实现新 bin-packing、ILP、EPLB、expert placement 变化、router 改写或 DeepEP kernel 优化。小规模 oracle 只用于后续评估优化上限。

### C3. 建议代码落点

下表为拟议修改，不是已经完成的代码。接入前检查相应目录规范，按实际调用关系调整。

| 路径（相对 slime 仓库） | 责任 |
| --- | --- |
| `slime/utils/route_pack.py` | 纯 CPU 摘要、模型、约束、有界交换 |
| `slime/utils/dp_schedule.py` | 原 schedule 生成后接入 refinement，保持返回结构 |
| `slime/ray/rollout.py` | 构造摘要并按稳定 sample ID 交给 scheduler |
| `slime/backends/megatron_utils/actor.py` | 导出完整训练拓扑与 expert layout，不能只补 ep_size |
| `slime/utils/arguments.py` | 最少量模式、预算及固定成本模型配置 |
| `tests/test_route_pack.py`、`tests/test_dp_schedule.py` | 摘要、调度约束、语义与回退测试 |
| `tools/routepack/`（拟新增） | profile 汇总、成本校准、benchmark 分析 |

参数可收敛为模式 `off/profile/refine/place`、refinement 预算、成本模型文件。具体名称实现时决定；不要提前固定未校准的 `--route-pack-lambda 1.0`。

## 8. 正确性验收：分三层，不能只看 loss

### 第一层：数据与调度硬约束

- baseline 与 RoutePack 保留完全相同的样本多重集合，没有重复、遗漏或意外改变尾部 trimming。
- 每个样本的 tokens、positions、loss mask、reward、advantage、old logprob、route IDs 与 metadata 对应关系不变。
- 样本不跨 optimizer step，rollout grouping 与 reward grouping 不被重新计算或破坏。
- global batch、每 rank microbatch 数、K 对齐和原 loss reducer 的分母保持一致。
- token-cap、超长样本策略、padding 和不支持配置的行为均有显式检查。
- 固定输入、模型和 seed 时 schedule 可复现；trace 不完整和成本无改善时能返回明确结果。

测试必须包含：EP>1 的多源槽位聚合、H/U 不同、同槽位交换不改变接收 H 的反例、末端/padding token、全均匀路由、singleton、所有样本同向偏斜及非法 layout。

### 第二层：前向与 replay

从相同权重对同一 rollout 比较原生 schedule 与 RoutePack，按 sample ID 和 token position 重排后检查实际 route ID、token logprob、loss。已有 v89 gate 在 RoutePack 关闭时必须继续通过。

在 RoutePack 开启后优先尝试保持已有 forward exact gate；新 batching 若暴露非零误差，应先分类诊断，不能直接放宽原 exact 承诺。若最终只能满足预先定义的数值容差，应把“原配置 exact”和“新调度容差验证”分别报告。

### 第三层：梯度与参数更新

原生与 RoutePack 从同一模型、optimizer state、输入和 RNG state 开始执行同一个 optimizer step。比较 loss、非零梯度、梯度向量误差/方向以及参数更新差异，而不只比较 grad norm。

microbatch 和归约顺序可能改变浮点结果，因此不要求参数更新 bitwise exact。推荐记录：

```text
relative_gradient_error = ||g_route - g_base||₂ / max(||g_base||₂, epsilon)
relative_update_error = ||delta_route - delta_base||₂ / max(||delta_base||₂, epsilon)
```

提前记录容差依据：dtype、原生重复运行波动、已有合法顺序变化对照和任务敏感性；在确认集前冻结阈值。梯度或更新接近零时同时报告绝对误差。参数 delta 优先从 FP32 master weights/optimizer 可观测状态测量，避免 BF16 权重舍入掩盖小更新。

随后完成至少两轮在线更新与权重同步，再做更长稳定性测试。Routing Replay 保证的是采用的 expert assignment；跨 optimizer 更新时的策略 logprob 不应被要求仍等于旧 rollout logprob。

## 9. Benchmark：把一致性、replay 和调度收益分开归因

### 9.1 两套比较

**比较 I：一致性开销。** 使用 A-native 与 A-aligned，回答整个数值对齐方案的成本。

**比较 II：RoutePack 调度收益。** 所有主对照固定同一数值 kernel、deterministic mode、Routing Replay、trace 数据、checkpoint 与训练超参数，只改变 scheduler。不能将“原生、无 replay”与“对齐、开 replay、开 RoutePack”之间的全部差异归因给 RoutePack。

| Policy | 作用 |
| --- | --- |
| Length-FF | 原生 token-cap first-fit，关闭 FLOPs packing/rank balance；保留实际配置 |
| Native-FLOPs-DP | 与 Length-FF 相同 bins，启用原生 `balance_data` 的 FLOPs rank balance；主要对照 |
| FLOPs-Pack | 可选：原生 `balance_by_flops` 打包；单独标记 token cap、K 和 OOM 差异 |
| Random-Swap | 在主对照上进行合法随机交换，记录交换数与开销，排除任意重排的收益 |
| RoutePack-A | 主对照基础上做样本交换 |
| RoutePack-AB | 增加组与槽位感知的 bin 分配 |

不必一次跑满所有组合。先比较 Native-FLOPs-DP、Random-Swap、RoutePack-A；有收益再补其他对照。RoutePack 与其直接对照必须保持相同 K；不同 packing 导致 K 变化的结果单列，不能混作纯 routing 消融。

### 9.2 固定数据性能实验

- 使用同一批真实 rollout trace 和 checkpoint，每个配对实验恢复相同模型/optimizer 状态；预热采用独立状态或预热后恢复，避免测量起点不同。
- 固定 trace 重放可以用于测量，但若省略 optimizer、weight sync 或 rollout，必须明确标为训练 replay，不称为在线 RL iteration。
- 从 5 个预热 step、20 个测量 step、3 个独立运行开始；若编译/内存分配尚不稳定则延长预热。长 workload 按 pilot 时长缩小，并记录实际样本数。
- 不同策略交错执行，固定 GPU 与后台负载条件；同一份 batch 的结果做配对统计。
- 报告中位数、p95、运行级波动和配对置信区间；以独立运行/global batch 为重采样单位，不把一层中的多个 rank 当作独立实验。
- profiler 关闭后的正常运行用于 headline 性能数字；profiler 运行用于归因。

### 9.3 在线端到端实验

在线执行 rollout、数据传输、训练、optimizer 和权重同步，包含 trace 收集、摘要生成及 CPU 调度成本。分别报告：在本来就使用 replay 的系统中增加 RoutePack 的增量成本，以及从不采集 trace 的系统迁移过来的总成本。

若新调度使更新浮点路径或生成长度变化，在线 iteration 不再是相同工作量的严格配对。需同时报告生成 token、训练 token、样本数和有效 loss token，并以固定 trace 实验作因果归因的主要证据。

### 9.4 指标口径

| 指标 | 定义与限制 |
| --- | --- |
| EP peak ratio | 每组/槽位/层的 max(H)/mean(H)，空负载单独标记，不除零 |
| EP excess | 组与槽位级求和；同时保留分层分布，不只报平均 |
| DeepEP/expert 耗时 | 分开报告 dispatch、combine、GEMM、等待；不重复累加重叠区间 |
| train step time | 完整 optimizer step，包括所有累积 microbatch；明确是否包含搬运 |
| training tokens/s | 该 step 非 padding 的实际训练输入 token 总数 / train step time；另报有效 loss token/s |
| rollout tokens/s | 生成的 response token 数 / rollout wall time |
| RL iteration time | 明确 rollout 开始到更新/同步完成的边界，包含新增 metadata 和调度成本 |
| GPU-hour / M tokens | 分配 GPU 数 × wall time 秒 / 3600 / token 数 × 1e6；明确采用 train 还是 iteration 窗口 |
| 内存/CPU 开销 | 峰值 GPU memory、raw trace/摘要字节、CPU schedule p50/p95、序列化/传输时间 |

收益公式分别为 `time_reduction = 1 - T_new/T_base` 与 `throughput_gain = Q_new/Q_base - 1`。相同 token 数下二者相关，但百分比不相等；例如耗时下降 8% 对应吞吐约提升 8.70%。

建议预先把“主 workload 的训练 step 中位数下降至少 3%、配对区间支持正向收益、在线新增成本未抵消收益”设为实用性参考门槛。若只达到微基准改善或区间跨零，则报告为局部/尚不确定结果。

## 10. 工作线 D：收敛实验如何做，以及何时不做

在 A/B/C 之外独立立项；开始前估算 GPU-hour。使用 random reward 的在线 gate 无法支持学习质量结论。

分清两个不同问题：

| 实验 | 对照 | 能支持的结论 |
| --- | --- | --- |
| 一致性的学习收益 | 原生数值路径 vs 对齐数值路径，scheduler 固定 | 是否在该任务和预算下改善评测表现/稳定性 |
| RoutePack 的质量守恒 | 数值/replay 固定，原生 scheduler vs RoutePack | 加速是否伴随可接受的质量变化 |

优先复用仓库已跑通的可验证奖励任务；确定具体数据集前另行核对许可、数据划分、评测方式和运行成本。使用独立 held-out 集、冻结 evaluator、相同起始 checkpoint、相同训练/评测预算，至少 3 个 seed 起步，报告波动与不确定性。

同时画评测表现随训练 token 与 wall time 的曲线。固定 token 预算下效果更好与固定时间下效果更好是不同结论。只有预先定义了目标分数并在多个 seed 达到，才能比较到达目标所需时间；未达到的运行不可删去。

“差异没有统计显著性”不能证明质量等价。若要宣称 RoutePack 不降低质量，提前定义可接受的质量下降界限并给出对应区间证据。预算不足时，只宣称短程数值和训练稳定性通过。

## 11. 执行里程碑与投入控制

以下为单人开发、环境可用时的工作日估计，不是 GPU 运行时保证。不并行扩大模型、拓扑、长度、算法和 reward。

| 阶段 | 估计投入 | 交付与退出标准 |
| --- | --- | --- |
| P0：冻结与复现 | 1–2 天 | manifest、拓扑、v89 回归；失败则先恢复可信基线 |
| P1：一致性代价 | 2–3 天 | A-native/A-aligned 性能表与初步稳定性记录 |
| P2：RoutePack feasibility | 2–3 天 | profile、trace 校验、留出预测、合法重排 pilot；明确 Go/No-Go |
| P3：RoutePack-A | 3–5 天，仅 Go 后 | 纯 CPU 核心、约束测试、同 batch 数值验收 |
| P4：集成和 benchmark | 3–5 天 | 原生/随机/A 对照、开销拆解、在线两步 gate |
| P5：AB 与消融 | 2–4 天，可选 | 仅在 profile 支持新分配收益时投入 |
| P6：整理交付 | 1–2 天 | README、复现入口、图表、适用范围、可审阅 diff |
| P7：学习效果 | 单独预算 | 完成短 pilot 后决定是否启动多 seed 正式训练 |

GPU 预算用 pilot 实际时长计算：`GPU-hours = GPU 数 × 独立运行数 × 单次秒数 / 3600`，再加启动、在线 rollout、诊断和失败重试成本。冻结资源预算后再确定扩展矩阵，不承诺“三周必然有加速结果”。

## 12. 文档、图表与简历交付

推荐新增的实验目录结构（尚未创建）：

```text
experiments/routepack/<run-id>/
  manifest.json
  topology.json
  config/
  schedule.json
  summary.json
  logs/
  profiles/       # 仅少量代表性 trace
  comparisons/
```

保留原始性能记录、统计脚本与小型摘要，避免持续保存全层大 tensor。复现入口应覆盖 correctness gate、固定 trace benchmark 和在线 gate；明确每条命令预计使用的 GPU 与时间。

最终报告建议包含四张图：

1. 原生与对齐路径的 logprob 偏差、耗时和显存对照。
2. 留出 workload 中 FLOPs-only / route-aware 预测误差，以及相近 FLOPs 下偏斜与耗时关系。
3. 原生/随机/RoutePack 的组与槽位负载分布，以及关键路径前后时间线。
4. 包含新增开销的训练 step 和在线 iteration 性能对照，附波动范围。

简历是否拆为两个项目由结果决定：RoutePack 有独立实测收益和完整代码时可单列；若主要交付是 profiler 与负结果，将它作为现有一致性项目的性能分析扩展。

结果成立后可以写：

> 基于 SGLang routing trace，在保持 replay 路由与全局 batch 语义的条件下，对 Slime/Megatron 的 EP group 与 microbatch 槽位进行负载感知重排；在 [已验证配置与工作负载] 上，使训练 step 耗时降低 X%，端到端 iteration 耗时降低 Y%，新增调度开销为 Z ms，并通过数据、梯度与更新后回归验证。

X/Y/Z 必须来自正式结果。仅 H/E 改善时不能填写训练加速；仅训练 replay 改善时不能填写在线端到端加速；短程 loss 稳定不能填写“提升 RL 收敛”。

## 13. 下一次实施从这里开始

1. 冻结运行环境、代码与 v89 基线，读取实际训练 process group 并导出 topology。
2. 在原 scheduler 下确认 alignment 与 Routing Replay 兼容，逐 sample/token 校验 trace，解释末端与 padding。
3. 采集自然 workload 的合法 schedule、H/U 和低侵入耗时，测清一致性路径本身的开销。
4. 离线检查能否改变 EP group/slot 输入集合，比较少量合法交换与随机交换；通过配对 pilot 后给出 Go/No-Go。
5. 只有 Go 后才实现在线 RoutePack-A。若 No-Go，保留 profiler 报告，优先优化 profile 已证明的主要开销。

本规划的完成标志是文档可执行、假设可验证。后续项目的完成标志则是得到可信的正确性与性能结论，无论结论是否为正向加速。


## 2026-09-10 实测进展

A阶段首组实验已完成：两轮在线StatelessAdam更新的一致性复现通过；固定rollout中native344/1024 mismatch、aligned0；每路径一次独立运行、预热后20步，training-phase step中位耗时17.4537s→17.9502s（+2.84%）。这是小批量固定输入受控对比，不是端到端RL吞吐或收敛实验。详细证据和下一步见 [结果报告](alignment_a_preparation/A_STAGE_RESULTS_20260910.md)。多次独立重复、扩大负载和RoutePack仍待推进。

### 三组独立重复验收

三组配对开销2.84%、2.19%、2.00%，平均2.34%。每种路径3次独立运行、60个预热后step。当前受控配置下可表述为约2%–3%training-phase step开销实现精确一致性，不外推生产吞吐。详细见 [重复结果](alignment_a_preparation/A_STAGE_REPEATED_RESULTS_20260910.md)。独立重复阶段已完成。

### 2026-09-10：补测推理侧并修正性能口径

既有 2%–3% 仅为 Megatron training-phase step 增量，不含 SGLang rollout 或权重同步；baseline 仍保留 deterministic mode。PR #2262 作者所述 25% 为 e2e step time，两者不能直接比较。推理侧三组独立配置对照已完成，平均生成耗时增幅 23.48%，协议及结果见 [推理补测与 PR 对比](alignment_a_preparation/INFERENCE_AND_PR2262_20260910.md)。

### 2026-09-10：真实在线端到端短程对照完成

每种配置独立运行一次，首轮预热、随后五轮真实在线迭代。总时间从 rollout 发起到训练更新、权重同步及下一轮推理就绪：baseline 中位数46.360s，aligned52.399s，本轮增幅13.03%。均值比增幅20.75%，早期轮次仍偏慢，因此不作为稳态开销结论。两组正式部分均5120 response tokens/10590 training input tokens，对应prompt哈希一致。下一步是延长预热/测量并独立重复，不需再次实现计时。详见 [在线实测](alignment_a_preparation/ONLINE_RL_TIMING_20260910.md)。

### 2026-09-12：CPU 搬运与确定性 matmul 快速排查

三条支线并行推进，详见 [CPU optimizer 调查](alignment_a_preparation/P1_OPTIMIZER_TRANSFERS_20260912.md)、[推理 matmul 归因](alignment_a_preparation/P1_MATMUL_ATTRIBUTION_20260912.md)、[RoutePack B1/B2 状态](alignment_a_preparation/routepack_b_20260912/REPORT.md)。

- CPU optimizer 的 pageable HtoD 已定位到 FP32 master 参数回拷：每个已采样 rank 约 15.27 GB，pinning 在 clone/float 过程中丢失。2×32 MiB staging 的独立微基准通过 BF16 bitwise 对比，耗时 7.12→4.18 ms；尚未集成训练，不代表 training-phase 加速。
- 新增一次 aligned 短窗口 shape capture。代表 rank 中，persistent matmul 累计时间约 42% 来自 router、51% 来自 QKV/输出投影、7% 来自 LM head。decode 小 tile pilot 出现可行候选，但尚未通过模型级 exact gate，也未部署到线上数值路径。
- RoutePack 已新增拓扑/owner 导出、原生 schedule 转换、H/U/末端/padding 校验及独立原始 route 采集入口。发现 aligned Qwen3 expert 选择分支绕过原生 capture 回调；已补回原生捕获回调，修复前 GPU 采集因重复 expert ID 被拒绝，修复后独立采集 8/8 条合法原始 routes（48 层、top-k=8，约 2 MB）。详见该支线报告。训练端 topology、actual dispatch 与 alignment+Routing Replay exact gate 仍待验证，未作调度 Go 决定。

下一步优先级：对 router 小 tile 做真实激活与模型级 exact 回归；CPU staging 进入受控训练验证前补齐 H2D stream/event 依赖；RoutePack 先完成训练端数据可信 gate。继续保留已跳过的长时间端到端重复，不将微基准比例外推为在线 RL 提速。

### 2026-09-12 后续并行执行状态

- Router 原实现完成 4 组模型级采集，batch8/3/1 对应 tokens、routes、response logprobs 全部逐 bit 相同；候选小 tile 仍默认关闭。
- 已生成包含真实 routes/logprobs 的 B1 固定输入；B1 实际 actor/dispatch/replay gate、B2 真实 topology/schedule/H/U 汇总入口及 CPU 约束验证完成。
- 两次启动被另一 Qwen3-4B GPU 实验占用打断，均发生在模型加载前。当前需连续 8 卡窗口执行 shadow、candidate、训练 B1，并据真实产物运行 B2。不能将初始化失败判作数值 gate 失败，也不能宣称 RoutePack Go 或整模型提速。
- 证据与续跑入口见 [router 模型 gate](alignment_a_preparation/P1_ROUTER_TILE_EXACT_20260912.md)、[B1 gate](alignment_a_preparation/routepack_b_20260912/B1_GATE_20260912.md) 和 [B1/B2 报告](alignment_a_preparation/routepack_b_20260912/REPORT.md)。

### 2026-09-12 本轮最终验收

- **推理 router 小 tile 模型 gate 通过**：off/shadow/candidate 的四组输入输出全部逐 bit 一致；shadow 对 3,538,944 个真实 FP32 输出元素零差异，覆盖 8 ranks × 48 层。实际 M=1/2、greedy、graphs 关闭；默认仍 off，尚无整模型计时结论。
- **B1 前向 gate 通过**：真实 Megatron/DeepEP 8 rank × 48 层，router/dispatch 路由差异 0，aligned/R3/Engine 的 128 response logprobs exact，max diff=0，optimizer steps=0。修复了与 Megatron `tools` 包的冲突，入口为独立 `routepack.*`。
- **B2 真实数据链通过**：独立重读 1,536 份 route 数组，真实 H/U/per-expert 与重建结果一致。实际 EP groups 为 `[0,1,2,3]`、`[4,5,6,7]`。当前单批 fixture 布局 padding 占 67.14%，不能用于 Go 或收益判断。
- 下一步优先无审计开销的 router 成对计时；RoutePack 扩至 10–20 个代表性独立 global batch，结合耗时再评估合法重排。完整证据见 [B1/B2 最终报告](alignment_a_preparation/routepack_b_20260912/REPORT.md) 和 [router exact gate](alignment_a_preparation/P1_ROUTER_TILE_EXACT_20260912.md)。

### 2026-09-12 native trace 离线可行性筛查

已离线分析 native/aligned 各 rank0/4 的真实训练trace，未使用GPU。native profiled step约18.139秒，F/B CPU窗口约2.226秒，optimizer约15.4秒；DeepEP GPU累计约34–35ms，采样rank上的MoE CPU wrapper也无秒级开销。大GPU活动空档落在dense collective CPU调用区间，尚不能区分主机/profiler/跨rank等待。rank0/4均为不同EP组的local rank0，无法证明组内平衡或定位straggler。

当前不支持据此优先扩展RoutePack在线实现；先保持native scheduler，补代表性匹配workload、同一EP组全rank路由与耗时关联，再决定Go/No-Go。不能把本次负面筛查外推为RoutePack普遍无效。完整结果见 [离线报告](alignment_a_preparation/routepack_b_20260912/NATIVE_TRACE_SCREENING_20260912.md)。

### Cross-slot exchange CPU implementation

Implemented `tools/routepack/exchange.py`: opt-in offline first-improvement sample swaps within an optimizer step and across different slots, using per-collective/per-layer EP excess. Incremental H/U includes recomputed padding; enforce fixed rank/slot cells, complete sample identity, padded bin capacity and no increased padding. Independent full B2 reconstruction validates output loads. This is a candidate generator, not live training integration. The original single-slot B1 fixture has no cross-slot candidates; no performance claim follows. See `tools/routepack/README.md` for CLI and limits.
