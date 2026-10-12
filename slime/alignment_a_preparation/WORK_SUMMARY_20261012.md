# alignment_a_preparation 工作总览与实验总结

整理日期：2026-10-12，Asia/Shanghai。目录实际名称为 `alignment_a_preparation`。
覆盖 2026-09-09 至 P10–P14 实验的准备、训练一致性、
性能测量、瓶颈调查、CPU optimizer、RoutePack/AutoPack 与推理优化工作。

本文是总览，原始阶段报告和失败目录保留。旧报告中的“未运行”“进行中”和
交接 PID 是对应日期的状态，不能直接作为今天的状态。
第1–16节记录 P9 及以前的历史总结。P10–P14 的最新结果与未完成范围见第17节。
历史baseline完整decode Graph已通过捕获/回放和有限模型门禁，20正式批中位数
1.890481秒，比同轮最佳baseline Breakable的3.093301秒下降38.885%。Prefill保持eager。
完整记录见第13节；目录入口已更新，所有历史证据保留。

## 1. 项目目的与已经建立的事实

本项目首先验证 Qwen3-30B-A3B 在 SGLang 推理与 Megatron 训练之间的 token
logprob 精确一致性，再分开评估实现成本，最后尝试保持数值语义的优化。
AutoPack 是另一条调度研究线，目标是改善实际专家负载及最终训练耗时。

已经完成的核心工作：

1. 恢复两轮在线 StatelessAdam 更新及全量权重同步后的一致性；保存数据独立复算
   均为 1024 个目标 token 零 logprob 差异。
2. 在受控训练对照中，目标 token 不一致从 344/1024 降为 0；三组 training-phase
   配对增幅平均 2.34%。
3. 独立推理原始配置对照约有 23%–24% 增量；这个指标与训练 2.34% 使用不同
   baseline 和计时边界。
4. 完成一次短程在线 RL 计时 pilot；总时间中位数比 +13.03%，尚无稳态独立重复证明。
5. 定位并验证 CPU offload optimizer 的 FP32 master pinning 与 H2D event 问题；
   隔离 optimizer 测试明显改善，完整模型训练收益未重新测量。
6. 完成真实 routes 采集、原生/交换 schedule 的前向检查、负载重建与 AutoPack
   离线搜索；已测完整训练 step 尚未证明稳定净加速。
7. 推理侧依次完成 router 小 tile、残差 RMS、Q/K 布局融合、Breakable Graph、
   KV 每批检查、专家 Graph 和 DeepEP Normal 单节点固定容量 full decode。
8. 可迁移优化已接入非确定性 baseline；baseline 使用自身原生 norm 融合。

近期推理融合与 Graph 的有限模型门禁，不等于训练端、参数更新和权重 onload/
offload 生命周期已重新通过；这些范围单独标明，不从早期训练结果自动推导。

## 2. 目录结构、数据与实际依赖

| 路径 | 内容与作用 | 阅读方式 |
|---|---|---|
| 顶层阶段报告 `*.md` | 每项实验的方法、数据、失败和局限 | 本文按主题归纳，原报告作证据入口 |
| `README.md` | 9 月 9 日 CPU 准备方案和原始对照定义 | 保留历史正文，入口指向本总览 |
| `gpu_tools/` | 启动、有限门禁、基准、profile 分析、资源保护及 CPU 测试 | 使用对应阶段入口，不能任意执行旧启动器 |
| `gpu_runs/` | 成功/失败运行、JSON/JSONL、日志、trace、源码快照 | 本地产物，Git 忽略；相同 run ID 不覆盖 |
| `cpu_runs/` | CPU Adam 数学/填充/线程诊断 | 不是 GPU 或整模型计时 |
| `cpu_validation/` | 历史 step1/2 logprob 的 CPU 复算 | 保存数据证据，不是当前 GPU forward |
| `routepack_b_20260912/` | B1/B2 集成、拓扑、实际 schedule、负载重建与早期交换 | 与后续 AutoPack 长样本实验衔接 |
| `source_snapshots/` | 早期启动器和依赖差异快照 | 不当作当前可执行 launcher |
| `artifact_inventory.json`、`source_manifest.json` | 早期文件来源、包 metadata、实际路径和 SHA | 记录日期为 9 月 9 日，不能代替后续 worker manifest |
| `comparison_recipe.json`、`v89_cli_static.json` | 初始声明式对照和历史 CLI 审计 | 不是 GPU 验收结果 |
| `results_template.csv` | 初始待填结果模板 | 空值是未测，不能理解为零 |

整理时扫描了目录层级、顶层全部阶段报告、B 阶段报告及工具职责；`gpu_runs`
包含数十个运行目录和大量数组/trace。总览不重新执行所有历史实验，也不逐个
重新解压所有大型 trace。关键结论引用已完成的汇总与对应原始证据路径。

实际推理源码为项目 `experiments/sglang-main-sm80/python`，实际训练 Megatron
依赖为 `/public/home/yingxy2024/slime-deps/Megatron-LM`，不是根目录下另一份副本。
近期解释器为 `/public/home/yingxy2024/.conda/envs/slime/bin/python`；PyTorch
2.11.0+cu129、Triton 3.6.0、安装 DeepEP 1.1.0+6845ffd。SGLang 安装 metadata
的 0.0.0 不能识别实际源码版本，应看 worker import、manifest 和源码快照。
本地 DeepEP 仓库 HEAD 也不能替代已安装扩展的 commit；通信分析使用匹配的
`6845ffd` 源码。

冻结推理输入是 `/public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt`，
SHA256 为 `ed72c22c9d16e13d455f83af09bda05420ad5e81d678c39a800bd721a550204d`。
8 个 prompt 共 1218 tokens；标准推理测量每请求固定生成 128 tokens，共 1024。
早期训练使用完整 prompt+response，共 2242 个未填充输入 tokens。
这份历史 rollout 的 routes 为空，不能作为 Routing Replay 数据；后续另采真实 routes。

## 3. 比较对象与计时口径

### 3.1 四种容易混淆的比较对象

| 名称 | 含义 | 不应推导的结论 |
|---|---|---|
| native-controlled 训练 | 同一补丁软件栈，关闭 Qwen alignment，保留训练确定性、offload 与 StatelessAdam | 不是最优非确定性训练配置 |
| aligned-controlled 训练 | 相同条件，启用 Qwen 对齐 hooks、attention/MoE/logprob 路径 | 其增量不是整个在线 RL 的增量 |
| configured baseline 推理 | 关闭 deterministic inference/unified decode，采用 baseline norm、sampling、通信配置选择 | 不是未经修改上游或调优生产峰值 |
| deterministic 推理 | batch-invariant 数学、unified decode、确定性采样/通信约束及显式 DeepEP config | 与 baseline 的差值不能只归因于 RMSNorm |

确定性开关还会改变 matmul、sampling、custom all-reduce 等行为。
normal/breakable/full 是执行方式；它们与“是否确定性”是两个维度。
比较确定性额外开销时，应让两边使用相同 Graph 方式和测量协议。

### 3.2 指标定义

| 指标 | 计时范围 | 排除或限制 |
|---|---|---|
| training-phase step | 同步边界内 F/B、optimizer、清零、指标归约；取最慢 rank | 不含外层 rollout、模型搬运和同步 |
| blocking `Engine.generate` | 调度、prefill、decode、sampling、logprob、输出/IPC | 不含加载、Graph 捕获、预热、cache flush、训练和同步 |
| 在线 RL iteration | rollout 发起到训练/同步、KV onload 阻塞返回 | 短程有早期慢轮，生成/更新轨迹可不同 |
| 单算子 wall 微基准 | 分配、Python/Triton 发射、GPU 完成 | 不是纯 kernel，也不是整模型收益 |
| profiler GPU duration sum | 所采活动时长累计 | 含跨 stream 重叠/等待，不是关键路径 |
| GPU interval union | 所记录活动在时间轴上的并集 | 不是整卡利用率；看不到的间隙不全是 CPU 成本 |
| H/U/E 负载代理 | 专家 assignments、目的 rank token 数、峰值减均值 | 不是通信字节，也不是训练时间 |

相对 baseline 开销为 `candidate/baseline - 1`；优化延迟下降为
`1 - optimized/reference`。分母不同，10.8% 剩余开销与约 10.9% 优化收益不是同一指标。
配对增幅均值、配对增幅中位数、先汇总中位数再相除也不同，历史报告采用各自明确口径。

## 4. A 阶段：训练一致性和性能

9 月 9 日先做 CPU 准备，9 月 10 日完成实际 GPU 验收和三组独立重复。

| 检查 | 结果 | 证据范围 |
|---|---|---|
| 历史 step1/2 CPU 复算 | 各 8 样本、1024 targets，最大差异 0 | 保存数据，CPU CUDA 未初始化 |
| 在线两轮恢复 | 两轮非零梯度，lr=1e-6，更新/全量同步后零 mismatch | StatelessAdam，每步重置动量 |
| 同权重 correctness 对照 | native 344/1024 mismatch、max_abs=0.79167509；aligned 0 | 小样本、既定 TP1/EP4 拓扑 |
| 第一组显存 | peak allocated 最大 rank 50.831→50.963 GiB | PyTorch 统计，不含完整通信/驱动内存 |

| 训练重复 | native 中位数 s | aligned 中位数 s | 配对增幅 |
|---|---:|---:|---:|
| 1 | 17.4537 | 17.9502 | 2.84% |
| 2 | 17.4999 | 17.8823 | 2.19% |
| 3 | 17.7675 | 18.1225 | 2.00% |

配对增幅平均 2.34%。每次 26 steps：安装测量的一步排除、5 步预热、20 步正式；
每路径 3 次独立启动、60 个正式 steps。性能 pass 为 lr=0，正确性与在线非零
更新分别验收，不把计时步骤描述为每步重新做了 logprob gate。
标准有状态 Adam、长期收敛和大 batch 尚不能由这个实验证明。

报告：[A 首组](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/A_STAGE_RESULTS_20260910.md)、
[三组重复](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/A_STAGE_REPEATED_RESULTS_20260910.md)。
工具：`launch.py`、`alignment_a_probe.py`、`continue_a.py`、`repeat_a.py`、
`check_run.py`、`summarize.py`、`summarize_repeats.py`。

## 5. 原始推理和在线 RL 对照

9 月 10 日独立推理三组 baseline/aligned：

| 组 | baseline s | aligned s | 增幅 |
|---|---:|---:|---:|
| 1 | 12.9616 | 16.0159 | 23.56% |
| 2 | 12.9220 | 16.0050 | 23.86% |
| 3 | 13.0203 | 16.0178 | 23.02% |

每组 3 预热+20 正式，每模式 60 正式批；平均配对增幅 23.48%。两边都因
DeepEP Normal 配置而关闭 Graph。该结论涵盖整体确定性配置，并非单个 norm 的消融。

在线 pilot 每模式一次独立启动、1 预热+5 正式迭代：总时间中位数
46.360→52.399 s（+13.03%），均值 47.197→56.988 s（+20.75%）。正式部分
两边均有 5120 response tokens、10590 training input tokens，但生成内容和权重
轨迹不要求相同。使用 random reward 和 StatelessAdam；aligned 有一轮零
advantage/零梯度，不能称六轮全部非零更新。

独立推理 23.48%、训练 2.34%、在线中位数 13.03% 不能相加，也不能以其中
最小的数字描述整个方案。此前与 slime PR #2262 的讨论属于不同模型/精度与
e2e 口径比较，本项目未复现该 PR 的生产配置；这里引用本地既有审计报告，
没有重新核实外部 PR 的当前状态。

报告：[推理口径](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/INFERENCE_AND_PR2262_20260910.md)、
[在线 pilot](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/ONLINE_RL_TIMING_20260910.md)。
工具：`inference_benchmark.py`、`run_inference_pairs.py`、`summarize_inference.py`、
`online_timing_helpers.py`、`continue_online_pair.py`、`summarize_online.py`。

## 6. Profiling：发现了什么，哪些结论已更正

### 6.1 初期训练与同步

独立训练采样的 native/aligned optimizer CPU 范围都约 15.4 s，不能将共同
optimizer 成本当作 alignment 独有成本。F/B 范围约 2.226→3.305 s，提示应
细分前后向，但一次带 profiler 的差异不能替代无 profiler 重复测量。

早期汇总按名称混加了 CPU `user_annotation` 和 `gpu_user_annotation`，已修正；
在线 rank0 train RPC 正确值为 28.508 s，旧 49.720 s 作废。原始 trace 和正常
性能结果没有因此失效，汇总器补充了对应回归检查。

一次在线 GPU profiler 后的 CUDA IPC 权重反序列化崩溃尚未独立定位或修复；
后续推理/训练独立采样成功不能证明在线同步问题解决。
根盘满与 NFS JIT 锁通过本轮专用 TMPDIR、节点 `/dev/shm` 缓存绕开；
没有删除系统 crash dump，也没有把历史系统状态当作今天仍存在的事实。

### 6.2 推理矩阵乘及发射

shape/correlation 关联把 `matmul_kernel_persistent` 分成 router、QKV、O 与
LM head。在短 aligned 窗口中，router 约占该类 kernel 累计时间 42%，QKV/O
合计约 51%，LM head 约 7%；这不是整个 generate 的耗时分摊。
decode 实际形状为 router M2/K2048/N128、QKV M4/K2048/N2560、O M4/K2048/N2048。

RMS 融合后 rank0 的 GPU 事件仍为 190028→242238，多 52210：kernel +46360、
memcpy +6241、memset −391。增加分布在 attention/norm、matmul、router 等，
不能概括成 DeepEP 内新增 52210 次 CPU launch。GPU 事件数包含 copy/memset，
也不等于 CUDA launch API 次数。

Q/K 每次原路径包含两个非连续布局复制和两个 norm kernel，baseline 有一次
fused inplace QK norm。另有确定性 unified attention 的每层 `.item()`，见 KV 专节。
这些证据说明慢并非只由 `rms_norm_batch_invariant` 一个函数造成。

本地 SGLang 的 `batch_invariant_ops.py` 首行明确写着改编自
Thinking Machines Lab 的 batch_invariant_ops；Q/K 调用其中的
`rms_norm_batch_invariant`。本项目新增的是联合布局/norm 候选和接入工具，
不应把上游原函数归为本项目新写。

报告：[初始 profile](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_PROFILING_20260910.md)、
[训练](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_TRAIN_PROFILING_20260911.md)、
[矩阵乘](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_MATMUL_ATTRIBUTION_20260912.md)、
[剩余发射](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_REMAINING_INFERENCE_PROFILE_20261009.md)、
[hgx03 Q/K profile](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_QK_PROFILE_HGX03_20261011.md)。

## 7. CPU optimizer：公共成本修复和未部署候选

### 7.1 搬运归因与 upstream backport

旧 trace 每个代表 rank 回拷 14.218 GiB；rank0 有 1537 次、rank4 有 1612 次。
GPU H2D 约 3.1–3.3 s，CPU H2D API 约 4.5–4.8 s，互相包含/重叠，不能相加。
定位到 CPU StatelessAdam 更新后的 FP32 master→GPU BF16 回拷 hook：

- 初始化先 pin BF16，随后 `clone().float()` 丢失最终 FP32 master 的 pinning。
- H2D stream 提交回拷后，却从 D2H stream 记录等待 event，未表达正确 H2D 完成依赖。
- overlap stream 存在不代表 CPU 线程能形成流水；旧 trace 99.79% 有后继的回拷
  在下一 CPU 子 optimizer 开始前已完成。

已移植 upstream #3140/#4547 对应两项修复，可复用 patch 为
`gpu_tools/optimizer_upstream_3140_4547.patch`；实际 Megatron 依赖修复记录为
`73a68e7`。这是局部 backport，没有整体升级 Megatron。旧训练 trace 是修复前数据。

单 A100 的实际 HybridDeviceOptimizer/StatelessAdam 隔离测试，128 参数、
1.125 GiB FP32 masters、CPU 4 threads、正常异步环境：

| 轮 | old ms | both 修复 ms | 延迟下降 |
|---|---:|---:|---:|
| 0 | 375.237 | 215.540 | 42.56% |
| 1 | 372.765 | 220.663 | 40.80% |
| 2 | 371.734 | 221.213 | 40.49% |

15 次更新后四种路径 GPU 参数哈希一致。先前测试继承 `CUDA_LAUNCH_BLOCKING=1`，
未形成重叠；有效实验仅在测试进程显式设 0，没有修改用户全局 shell 配置。
未在完整训练 step 上复测，不能报告训练加速 41%。

### 7.2 CPU 数学与有界 staging

确定性算法+未初始化内存填充会在 StatelessAdam 的 abs 输出分配产生 fill_；
CPU 开关干预确认该机制，但不能推导历史约 1.5 s 全部都是它。
`abs().mul_()` 候选保持原运算顺序，减少临时输出；有限位模式/分支检查通过，
尚未部署。4 threads 的 288 MiB 测试汇总 63.589→33.646 ms，存在较大轮间波动；
8 threads 原实现有秒级异常，不能据此推荐生产线程配置或宣称几十倍加速。

2×32 MiB pinned staging 微基准 7.120→4.175 ms（约 −41.4%）且 BF16 bits
一致；它只证明搬运可行性，不等于已接入线上 optimizer 的方案。

报告：[搬运](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_OPTIMIZER_TRANSFERS_20260912.md)、
[互斥归因/backport](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_OPTIMIZER_CPU_OFFLINE_20260912.md)、
[CPU 数学](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_OPTIMIZER_CPU_MATH_20260914.md)、
[GPU 隔离验证](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_OPTIMIZER_GPU_VALIDATION_20260914.md)。

## 8. RoutePack/AutoPack：数据链已建立，净训练收益未证明

### 8.1 B1/B2 和原始 route capture

首次 deterministic Qwen3 route capture 失败，因为手工 top-k 返回绕过 capture
helper；默认全零 cache 被唯一性检查拒绝。最小修复增加 helper 调用，保持原
IDs/weights/dispatch 数学。后续取得合法真实 routes。

B1 观测真实进程组、local expert owners、router IDs 与传给实际 Buffer.dispatch
的 primary payload，并比较 Engine capture、aligned fallthrough、R3 replay 的
response logprob bits。B2 独立读取实际 router/dispatch 和 schedule，再重建 H/U。
末端 synthetic route 与 layout padding 分开计算；不能按 dense DP 推断 EP topology。
真实 EP groups 为 `[0,1,2,3]` 与 `[4,5,6,7]`，各 rank 持有 32 experts。

| workload | 前向验收与规模 | 能支持的结论 |
|---|---|---|
| 8 请求短 fixture | 128 response tokens 三方 bits 一致；两 EP collectives | 原生 schedule 集成/计数链 |
| 32 独立 prompts×128 responses | 4096 response tokens，6 collectives，8 rank gate 全过 | 首个真实 multi-slot 验证 |
| 32 prompts×8 长回答 | 256 distinct 序列、1599412 response tokens、24 collectives | 长样本原生 schedule 前向和负载重建 |

长回答 cap8192：145 自然结束、111 达到上限。沿用已验证 TP1/EP4，不是 recipe
TP4/EP8；显式 sibling seeds42..49 的 capture02 有效，重复 sibling 的 capture01 排除。
这些 forward gate 不自动证明 backward、参数更新或调度加速。

### 8.2 搜索代理和 CPU 成本

H 是 token×expert assignments，U 是去重的 token→目的 rank 数；
`E=sum(group,slot,layer)[max H - mean H]`。
同一 collective 内只交换源 rank 通常不改变接收 H，真正改变输入集合的合法
跨 group/slot 交换才可能改变该目标。

早期合成 fixture：Random E=58846 优于 first-improvement RoutePack E=64858
（Native68574），没有删掉这个反例。后续 sampled-best 搜索预算明确限制为
500 candidates/20 iterations/20 swaps，并用相同合法性/匹配交换次数的 Random 对照。

短样本真实 E 仅改善约 0.32%–0.36%；padding 为28.67%。换长样本后 padding
降至1.3625%，原始路由峰均比 p50=1.192、p95=1.580、max=1.894；这个 workload
差异不是 AutoPack 降 padding 的收益。
长样本五 seeds 的 E 改善2.9491%–4.0581%，均优于匹配 Random；seed0 为3.1257%。
同一个 batch 的五个搜索 seeds 不是五个独立训练 batch。
seed0 搜索 CPU 中位数2.913 s、完整离线调用81.600 s；准备和复核各约39 s。
在线成本尚未接入，不能忽略它们报告净吞吐。

### 8.3 实际训练计时和归因

256 长样本 TP1/EP4 尝试多次 OOM，涉及 FP32 logits/loss/完整词表反向梯度。
仅缩小 loss chunk 不能消除 SplitBackward 拼接的完整 `[T,V]` 缓冲；CPU 保存张量
实验是 opt-in benchmark helper，不能称生产 PPO loss 已修好。

随后使用128完整样本、max_tokens_per_gpu10240：

| 指标 | Native | RoutePack | 结论 |
|---|---:|---:|---|
| 正式5轮完整 step 中位数 s | 141.05494 | 140.67554 | −0.269%，未证明稳定收益 |
| 正式均值 s | 141.17832 | 141.48607 | 反而 +0.218% |
| E | 20597635 | 19381518 | −5.904%，是偏斜部分的比例 |
| 最忙 rank H 之和 | 103566211 | 102251790 | 仅 −1.269% |

局部专家 forward 下降2.410%，重计算专家下降1.197%，但完整 MoE 重计算区间
增加3.905%；完整执行还包含 dispatch/combine、等待、排列、梯度和 optimizer。
对齐 wrapper 在 grad-enabled 情况运行 SGLang no_grad 前向和原生 Megatron
专家前向桥接梯度，局部 wrapper 不等于单套 GEMM。
后续首 microbatch 分段 profile 中重计算 dispatch 1.84845→0.77748 s，但
对应未采样单次完整 step 140.06787→140.07860 s，仍基本相同。
不能把局部 union 或 dispatch 降幅填入整体训练加速。

报告：[B 集成](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/routepack_b_20260912/REPORT.md)、
[真实长 workload](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/AUTOPACK_REAL_RESULTS_20260914.md)、
[局部/整体](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/AUTOPACK_LOCAL_TIMING_ANALYSIS_20260916.md)、
[分段 profile](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/AUTOPACK_COMPONENT_BREAKDOWN_20260916.md)。
核心调度/数据代码在项目 `slime/tools/routepack/`，本目录主要保存准备、实验和分析入口。

## 9. Plan 1：保持确定性数学的 eager 优化

### 9.1 Router 小 tile

保持 K32、原 persistent kernel 数学，只把 M/N tile 改16×64、4 warps。
真实模型适用范围收窄为 FP32 M1/2、K2048/N128，默认 off。
有限模型 tokens/routes/logprob bits 和 batch gate 全过；shadow 3538944 个元素零差异。
单算子约30% wall 改善，三轮整轮 generate 延迟下降中位数仅0.380%。
CUDA Graph 仍拒绝，不能自动叠加到今天 full Graph。

### 9.2 残差 RMS 融合与“10.8%”

确定性 `forward_native` 残差相加和 RMS 的多算子实现融合到一个 kernel。
初版真实 layer39 有1 bit 差异，原因是 load 布局改变 FP32 归约树；修复固定
每线程连续4元素、保留原归约/算术顺序后，通过真实模型224133120个BF16元素检查。

10 月 8 日三轮：baseline 中位数约12.694 s、未融合约15.768 s、RMS约14.042 s；
配对融合延迟下降中位数10.926%，剩余比 baseline 慢10.614%–10.962%。
所以历史“修完额外开销10.8%”是该环境测得的剩余比例，不是固定理论常数。

10 月 9 日 hgx02/srun 测量剩余21.517%–23.313%，融合仍比未融合快13.892%–15.734%。
清除 Slurm/PMI 子进程变量未消除差异，不能认定融合失效，也不能断言根因已定位。
10 月 11 日回到 hgx03/SSH extern 的180正式批复现：未融合额外23.877%，
RMS剩余10.027%，融合收益11.125%（配对比例中位数）。
节点与启动方式同时变化；尚无同节点 SSH/srun 因果实验，NUMA/频率/CPU 调度
仅是未隔离的候选。

### 9.3 Q/K norm 与布局复制联合

packed QKV 行 stride2560，Q `[4,2048]`、K `[4,256]`；原 reshape 各复制一次，
各执行一次 batch-invariant norm。联合候选直接读 stride，一次写连续 Q/K 输出。
初版 rsqrt 与上游 sqrt+倒数顺序不同产生1 bit 差异，已修正。
384真实张量+48合成案例7335936 BF16元素、完整模型127401984元素均零差异。
decode每 rank6144次调用，4→1 kernel，减少18432个kernel；copy类事件未因此消失。

| 10月11日 Q/K 三轮 | baseline s | RMS s | RMS+Q/K s |
|---|---:|---:|---:|
| 1 | 12.750854 | 14.097863 | 13.607312 |
| 2 | 12.748009 | 14.076789 | 13.571872 |
| 3 | 12.753549 | 13.999291 | 13.532129 |

Q/K 配对延迟下降中位数3.480%，最终剩余比 baseline慢6.463%。
这组180正式批全部 eager，不能直接拿这个收益比例预测 Graph 下的收益。

### 9.4 QKV/O 候选

真实投影张量的小 tile 保持 BK64和原累加方式，有限bit/batch检查通过：
QKV单算子47.225→34.285µs（−27.503%），O44.266→34.295µs（−22.470%）。
尚未接入完整模型或通过完整训练/Graph门禁，不应列为已部署加速项。

报告：[router](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_ROUTER_TILE_PERFORMANCE_20261007.md)、
[RMS](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_RESIDUAL_RMS_PERFORMANCE_20261008.md)、
[hgx03复现](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_HGX03_THREE_MODE_REPRO_20261011.md)、
[Q/K与dense](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P1_QK_LAYOUT_AND_DENSE_INVESTIGATION_20261011.md)。

## 10. Plan 2：Breakable、KV 检查与内部融合

### 10.1 最初的切分

显式 decode Breakable 下每 rank/bucket49段、48个break；可捕获的attention、
norm、gate/top-k等在图里，每层整个MoE（dispatch→expert→combine）为eager。
不只是通信eager，也不是整模型full Graph。prefill保持eager。
适配了静态 `LogitsProcessorOutput` 输出桥和normal配置层限制，原默认不自动启用。

三轮240正式批汇总中位数：baseline eager12.708907 s、原确定性eager15.761854 s、
Plan1 eager13.563693 s、Plan2 Breakable6.450765 s。
相对确定性eager的配对延迟下降中位数59.073%；与eager baseline不同执行方式，
不能解释为确定性数学比baseline更快。
早期60个Graph正式批中2批首次写了bucket2证据，敏感性分析中位数最多变化0.0225%；
后续工具增加计时前小bucket预热，避免正式区间的一次性证据I/O。

### 10.2 为什么每 rank6144次 `.item()`

本地确定性 unified decode 每层检查 `any(kv_lens<=0).item()`，用途是拒绝空KV。
48层×128次decode=6144次GPU布尔标量读回，包含1字节D2H和CPU流等待；
原baseline关闭unified分支，因此此处为0次。这是本地KV检查逻辑，不是DeepEP
或batch-invariant RMSNorm要求。

最初捕获内改设备异步断言可运行，但非法长度导致CUDA上下文失效；独立异常
进程验证确认这一点。最终改为metadata准备阶段每batch检查一次live KV缓冲，
各层只读Python已验证标记；非法长度在replay前抛ValueError，最小GPU检查后
上下文仍可用。没有把检查全删掉，也不是绝对零同步。
按用户要求未单独计时KV优化，因此没有这一项的单独加速百分比。

### 10.3 RMS/QK 与 Breakable 叠加

候选必须在捕获前安装，新增显式 `before_cuda_graph_capture`，否则图中仍是旧算子。
RMS只允许无审计candidate显式graph opt-in；Q/K hook也要求显式允许graph。
tensor变化输入/权重回放以及有限模型/batch门禁通过。

| 同一单轮，确定性+KV-once | 中位数 s | 相对原Breakable |
|---|---:|---:|
| 原Breakable | 6.460628 | — |
| +RMS | 6.122183 | 延迟下降5.239% |
| +RMS+Q/K | 6.145010 | 延迟下降4.885% |

Q/K联合版比RMS单项慢0.373%，一次顺序启动不足以证明稳定回归；也没有证明
叠加后的额外收益。当前选择保留RMS，Q/K在Graph下为可选实验。

报告：[Plan2实现](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P2_BREAKABLE_CUDA_GRAPH_IMPLEMENTATION_20261011.md)、
[Plan2计时](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P2_BREAKABLE_CUDA_GRAPH_PERFORMANCE_20261011.md)、
[KV审计](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P2_KV_ITEM_AUDIT_20261011.md)、
[KV最终修正](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P3_KV_ONCE_AND_BASELINE_GRAPH_20261011.md)、
[融合叠加](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P5_BREAKABLE_GRAPH_FUSIONS_20261011.md)。

## 11. 专家 Graph 与 DeepEP Normal 固定容量 full decode

### 11.1 专家计算单独捕获

保留outer49段/48MoE回调、dispatch/combine eager，在回调内部捕获专家core，
真实路由/hidden/weight每次拷入静态buffer。缓存的是计算形状，不是路由。
为每rank48层×接收M1..16建768个图，M0走原兼容路径；权重地址替换要求重建。
24246真实shadow调用229115904元素与原core bit相同，四case模型及batch门禁通过。

| 确定性+RMS | expert eager s | expert Graph s | 延迟下降 |
|---|---:|---:|---:|
| 原顺序 | 6.171170 | 4.307625 | 30.198% |
| 反序复测 | 6.123793 | 4.333908 | 29.228% |

各组3预热+20正式。观察到新增驻留显存约1678MiB/rank，不是峰值；allocator
active tensor统计约27.8MB不能代替总Graph/保留池显存。

### 11.2 普通 Normal full 捕获为什么失败

仅绕过配置禁用而仍使用普通动态dispatch：8rank均在full捕获中
`DeepEP error: CPU recv timeout`，没有正式generate时间。
普通dispatch提交notify后CPU等待GPU接收数量，再分配输出；捕获阶段kernel只
记录而不执行，CPU得不到数量。后续stream捕获结束错误是清理失败的次生异常。
这证明当前集成方式不兼容，不能说A100不支持CUDA Graph。

### 11.3 已安装 DeepEP 自带的 Normal 单节点分支

传 `Buffer.dispatch(num_worst_tokens=16)`，接收容量固定，不等CPU数量；
仍是Normal/intranode通信，不是切换low-latency，也没有改底层通信源码。
当前EP4、每rank最多4发送行，16为保守上界。prefill不走此改动。
实际路由仍动态、每次handle重新产生；padding IDs为−1，设备侧将无效X/weights
清零，原Tritoncore使用固定M16。捕获后Graph复用静态显存地址。

限制“单节点”来自已安装版本的Normal实现：跨节点有RDMA和NVLink两级接收，
仍等待最终/RDMA接收数量，缺少对应固定容量分支。不是理论上跨节点不能做。
low-latency另一套接口原本有固定每专家容量，不能直接替代Normal分支。

确定性full gate：21504 shadow调用206491648有效BF16元素零差异，包含343零接收；
8rank×bucket2/4实际捕获48dispatch/48expert层并成功原生full回放。
full3.268245 s、P95 3.385360 s、313.32输出tokens/s，20正式批。
相对随后比较的expert Graph4.333908 s延迟下降24.589%，属于一次顺序试验。

full含decode forward的attention、norm、router、dispatch、padding清理、expert、
combine、LM head；metadata准备、调度、sampling及输出处理仍在图外。
不代表整个请求或在线训练流程全部进入Graph。

报告：[专家Graph](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P6_EXPERT_COMPUTE_CUDA_GRAPH_20261011.md)、
[固定容量full](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P7_FIXED_CAPACITY_FULL_DECODE_20261011.md)。
普通动态Normal捕获失败的独立P4报告和探针代码仅保留在历史实验分支，按用户
要求不纳入main；此处保留原因说明，便于理解成功方案为何需要固定容量。

## 12. 非确定性 baseline 的可迁移优化

baseline原本用native fused add/RMSNorm以及符合条件的fused inplace QK norm。
旧确定性RMS/QK候选不是直接可迁移的新优化；KV每批检查也属于baseline关闭的
unified路径。专家Graph与固定容量不依赖batch-invariant数学，可以迁移。

| baseline Breakable，同轮各20正式 | 中位数 s | P95 s | 输出tokens/s |
|---|---:|---:|---:|
| 原版 | 5.179104 | 5.244751 | 197.72 |
| +expert Graph | 3.271475 | 3.359833 | 313.01 |
| +fixed dispatch、expert eager | 5.701649 | 5.830163 | 179.60 |
| +fixed dispatch+expert Graph | 3.110243 | 3.169973 | 329.23 |

组合相对原Breakable延迟下降39.946%，相对expert单项再下降4.928%；fixed独立
启用反而慢10.089%，不能把固定容量本身描述为必然加速。
组合每rank只缓存48个M16专家Graph；outer仍49段/48break，尚不是full。

第一次fixed模型比较出现route/logprob差异，起点在未改动的prefill；日志显示
两版本prefill请求分组不同。baseline本身batch-sensitive，因此第二次只在
正确性gate用 `prefill_max_requests=1` 固定分组，正式计时仍采用默认自然调度。
四case有限tokens/routes/logprob bits和组件shadow通过。
自然调度23批采样哈希与原版匹配数为expert18、fixed18、组合17，不能称所有
正式输出bit相同。固定输入/长度不等于固定token身份和MoE路由工作量。

报告：[baseline迁移](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P8_BASELINE_BREAKABLE_OPTIMIZATIONS_20261012.md)。
本轮结果不是与确定性full的同轮同方式对照，不能拿3.110和3.268直接算确定性开销。

## 13. 本次 baseline 整个 decode forward full Graph

状态：完成；捕获、回放、固定容量shadow和有限完整模型比较通过，40正式批
及source锁/缓存/配置/退出状态已核验。
运行ID `baseline-full-hgx03-20261012-01`。
入口 `gpu_tools/run_baseline_full_pipeline.py`，独立adapter保留baseline原生norm、
非确定性开关、unified关闭、baseline通信config选择，只启用Normal固定容量。
correctness gate控制prefill分组；timing保持自然调度，与最佳baseline Breakable
固定容量+expert Graph重新同轮测量，各3预热+20正式。

| 本次baseline，同轮独立Engine | 中位数s | P95s | 输出tokens/s |
|---|---:|---:|---:|
| Breakable+fixed dispatch+expert Graph | 3.093301 | 3.215569 | 331.04 |
| fixed dispatch+full decode Graph | 1.890481 | 1.984975 | 541.66 |

延迟下降38.885%，吞吐增加63.625%；每配置仅一个独立启动，不宣称高精度稳定比例。
16份捕获和16份实际full回放记录覆盖8rank×bucket2/4，每次含48dispatch和48expert层；
不是Breakable fallback。全decode forward包括attention、norm、gate/top-k、dispatch、
padding、expert、combine和LM head；prefill、metadata准备、采样/调度/输出仍图外。

Fixed eager shadow共21504次、171687936个有效BF16元素，包含1085本地零接收，
零bit差异；eager固定容量和full各与原baseline的四组受控greedy输入tokens/routes/
logprob bits一致。自然计时两模式23对输出哈希17对一致，baseline的batch敏感性
仍存在，不能写成所有正式输出逐bit等价。两组resolved参数仅图配置和hooks不同。

共46标准生成样本（含6预热），全部零缓存、无profiler/数值shadow/逐调用文件I/O，
计时source SHA不变、worker退出0，证据写入在正式测量前完成。Breakable部分
bucket4证据在标准预热第一批写入，full记录在额外小bucket预热结束前完成。
配置/计时/证据/哈希核验在 `measurement-validation.json`，raw数据留本地。
完整阶段说明：[P9 baseline full](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P9_BASELINE_FULL_DECODE_20261012.md)。
不与旧确定性full3.268245秒直接算已确认的确定性开销。

## 14. 实现状态与复跑入口

| 项目 | 代码位置/入口 | 当前适用范围 |
|---|---|---|
| router tile | SGLang `models/qwen3_router_tile.py`；`router_tile_performance.py` | 默认off、M1/2确定性、Graph仍拒绝 |
| residual RMS | SGLang `batch_invariant_ops/fused_rmsnorm_candidate.py`；`residual_rms_performance.py` | 默认off；显式无审计graph opt-in |
| Q/K联合 | `qk_norm_prototype.py`、`qk_runtime_hooks.py`、`run_qk_model_and_performance.py` | 小M/BF16/head128，显式hook |
| dense tile调查 | `dense_projection_capture.py`、`dense_tile_probe.py` | 未集成模型 |
| Breakable | SGLang config/output桥；`run_breakable_model_and_performance.py` | 显式backend，Normal MoE eager |
| KV once | SGLang `layers/attention/triton_backend.py`；`kv_once_gpu_check.py` | 当前受支持unified decode metadata |
| Graph融合 | SGLang捕获前hook支持；`run_breakable_fusions.py` | 无审计RMS/QK显式开关 |
| 专家Graph | `expert_graph_hooks.py`、`run_expert_graph_investigation.py` | BF16固定权重地址、EP4、decode≤4 |
| fixed/full确定性 | `fixed_capacity_dispatch_hooks.py`、`fixed_capacity_experiment.py`、`run_fixed_capacity_pipeline.py` | 单节点EP4/H2048/alignment1、cap16 |
| baseline Breakable | `baseline_graph_*.py`、`run_baseline_graph_optimizations.py` | 原生norm/非确定性，组合cache_rows=[16] |
| baseline full | `baseline_full_*.py`、`run_baseline_full_pipeline.py` | 本次实验专用显式full bypass |
| optimizer backport | `optimizer_upstream_3140_4547.patch`、`benchmark_optimizer_backport.py` | 实际依赖局部修复，整训练未复测 |
| Adam数学/staging | `benchmark_adam_cpu.py`、`benchmark_optimizer_staging.py` | 隔离候选，未部署 |
| AutoPack | `slime/tools/routepack/`，本目录 `autopack_*` 和分析工具 | 离线/受控实验，净训练收益未证实 |

本次baseline full可从已进入ai_hgx_03 allocation的SSH shell复跑，使用未占用ID：

```bash
cd /public/home/yingxy2024/moe-consistency
/public/home/yingxy2024/.conda/envs/slime/bin/python \
  slime/alignment_a_preparation/gpu_tools/run_baseline_full_pipeline.py \
  --run-id baseline-full-hgx03-新的编号
```

工具依赖本地模型、冻结rollout和已有环境manifest，不是可在任意机器无数据运行
的通用benchmark。部分旧入口写死日期/节点/参考目录，复跑前需核对，不能直接
使用已存在run ID或假定旧Slurm/PID仍有效。

资源控制共用项目锁、GPU空闲/P2P检查、实际cgroup检查、内存和超时守护；仅
停止确认属于当前worker的进程组，保留Slurm资源，不停止无关任务。
正确性/shape capture/audit和正式计时分进程，source SHA锁防止计时期间变更。

## 15. 失败与结论边界索引

| 现象 | 已确认事实或修复 | 仍不能宣称 |
|---|---|---|
| 根盘满/NFS JIT锁 | 迁移专用TMPDIR/节点缓存成功运行 | 系统根盘问题已修复 |
| 在线profile后CUDA IPC崩溃 | 保存失败证据，改独立采样 | 在线profiler+权重同步生命周期已修复 |
| hgx04 peer access unsupported | GPU2 NVLink/P2P检查失败，换节点 | router数值失败 |
| RMS初版1 bit差异 | load布局/归约顺序修正后模型gate通过 | 微基准相同足以保证模型相同 |
| Q/K初版1 bit差异 | rsqrt改回sqrt+倒数后通过 | 可任意替换等价代数式 |
| 图捕获后安装hook | 增加明确捕获前安装与计数证据 | 设置candidate开关就代表图中生效 |
| 每层KV `.item()` | 最终每批metadata验证一次 | 单独节省多少秒已测 |
| 普通Normal full timeout | 固定容量分支绕过CPU计数等待 | 所有DeepEP/跨节点都兼容full |
| baseline gate分组差异 | gate固定prefill grouping，timing保持默认 | baseline跨batch不变或正式全部bit相同 |
| trace gzip截断 | 完整原JSON、结束标记、唯一前缀、全量SHA恢复 | 任意损坏trace都可推测恢复 |
| AutoPack full-step OOM | 保留独立内存实验及TP1/EP4容量限制 | 前向gate成功就能完整训练 |
| profiler dynamic-toggle无GPU事件 | 标invalid，独立窗口smoke后重新采集 | CPU-only trace可用于GPU时间结论 |
| SSH丢失memory cgroup | 资源预检拦下，重新SSH归属原allocation | 被拦下运行已有GPU数值/性能结果 |

## 16. Git、结果保存与下一步

验证通过的六个阶段已直接选择合入main并保留原始实验分支的提交记录。
失败探针 `d5441e77` 的代码及独立P4报告未合入main；原始失败运行目录保留本地。
main整合核验确认保留的运行时代码与GPU测量source SHA完全一致；Breakable输出/
配置和KV-once的13项CPU检查通过，仓库规则及文档链接检查通过。CPU检查期间
将FlashInfer缓存重定向到项目可写临时目录，解决导入阶段默认用户缓存只读的问题；
没有为此修改实现，也未重新计时GPU。

| 成功阶段 | 原分支提交 | main提交 |
|---|---|---|
| Breakable Plan2 | a94d2423 | 77c8f7ec |
| KV每batch验证 | 962a79d4 | de696d5d |
| Graph融合及专家Graph | fb87cd28 | 6fbdd4a3 |
| 固定容量确定性full | 92558305 | 0e9e2430 |
| baseline Breakable优化 | 0a3a91cd | 3788f6fe |
| baseline full和总览 | 195fb228 | adafc97a |

以下为历史分支记录，失败探针一项仅供来源审计，不表示它已进入main：

| 分支/提交 | 内容 |
|---|---|
| main `f139801f` | hgx03三模式复现 |
| main `f99fd917` | Q/K布局联合候选和dense调查 |
| `codex/breakable-cuda-graph-plan2` / `a94d2423` | Plan2及KV审计 |
| `codex/kv-validation-once-per-batch` / `962a79d4` | KV每批一次 |
| 同分支 `d5441e77` | 普通baseline full失败探针，仅历史分支，未合入main |
| 同分支 `fb87cd28` | Graph融合及expert Graph |
| `codex/fixed-capacity-full-decode` / `92558305` | 固定容量确定性full decode |
| `codex/baseline-breakable-optimizations` / `0a3a91cd` | 非确定性baseline四模式、专家缓存配置及P8报告；已推送 |
| `codex/baseline-full-decode` / `195fb228` | baseline完整decode实现、P9、本总览及README；成功内容已选择合入main |

本次开始时baseline迁移工具/P8报告为工作区改动，随后保存并推送 `0a3a91cd`；
full工具、P9、总览和README另保存并推送 `195fb228`。按用户最新要求，现直接
在main整合成功内容，后续工作也直接使用main；失败独立探针不合入。
`gpu_runs`、`cpu_runs`、routepack数组等按仓库约定保留本地，不能force-add
大数据、权重或trace。这里只新增入口和文档，不移动或删除历史证据。

仍值得开展、但不冒充已完成的工作：

1. baseline与确定性full在相同后端/配置协议下重新配对重复，估计当前确定性开销。
   本次baselinefull若成功与旧确定性3.268 s跨轮比较，只能作线索。
2. 在所有参与rank发送行上界有证明和运行时guard后，评估cap16→8；先处理
   接收padding和正确性，不能依据一次观测1/2行直接缩容量。
3. 减少padding清理GPU工作，或验证QKV/O小tile与full叠加；先真实模型gate。
4. 参数更新、权重重绑、onload/offload与Graph重建生命周期门禁，扩大batch/模型拓扑。
5. 完整训练验证optimizer backport/数学候选的净收益；扩大在线RL独立重复，保持
   reward/optimizer/同步口径明确。
6. AutoPack以多独立workload、同group全rank和关键路径验证净收益，把在线routes/
   搜索CPU成本计入；当前不建议仅增加swap预算来宣称训练提速。

最终实际driver PID/PGID127310及全部worker已退出；GPU compute列表为空，
8卡1MiB/0%，Slurm1031236保留RUNNING。用户此前明确取消保号，本轮未恢复；
历史报告中保号PID仅是历史记录。
Driver日志为项目 `.a-tmp/baseline-full-hgx03-20261012-01-driver.log`。
本次未移动/删除历史运行目录，完成工作以阶段报告、总览和Git提交保存。

## 17. P10–P14：公平 Full Graph 对照、归因与更新生命周期

P10、P12 已完成 GPU 数值和性能验收；P11 按用户取消剩余 NCU 采集后的范围完成。
P13 真实在线生命周期回归也已完成；P14 尚未验收。以下结果不追认第16节的历史进程或资源状态仍有效。

### 17.1 P10：比较执行方案的实际剩余差距

三模式均采用同一提交、checkpoint、8×A100、TP8/DP2/CP2/EP4/MoE DP2、
decode full、buckets2/4、接收容量16和 prefill eager。A 非确定性默认通信配置，
B 非确定性显式20SM配置，C 确定性且使用B的通信配置与已验证残差RMS融合。
每Engine3预热+20正式，三轮ABC/BCA/CAB，共9次独立启动、180正式批。
实际token/routes被保存，route回传成本计入generate，文件保存和差异分析在计时外。

| 模式 | 三个Engine中位数再取中位数 s | 与对应对照的三轮配对变化中位数 |
|---|---:|---|
| A：非确定性默认通信 | 2.019609 | 基线 |
| B：非确定性固定通信 | 1.994497 | B/A −1.243% |
| C：确定性固定通信 | 3.389224 | C/B +69.939% |

默认A实际也使用20SM，区别主要是NVL/RDMA分块参数，不能说A→B仅改变SM数量。
各rank正式区间2560次原生full replay，纯全局decode eager回退0；全局prefill期间
DP局部decode允许eager，并单独记录。C改变GEMM、attention、sampling、all-reduce
等多条数值路径，C/B不是RMS或纯GEMM开销。自然生成的A/B/C实际token/routes
存在差异，因此这是一组完整执行方案的性能比较，不是固定每个GPU操作数的消融。
P9没有相同route回传开销，1.890481秒不能与此表直接作同条件比较。

证据：[P10测量](P10_FULL_GRAPH_DETERMINISTIC_AB_20261012.md)、
[P10模型与通信门禁](P10_FULL_GRAPH_AB_GATE_20261012.md)。
成功提交 `68b8fef3`、`ff82e68d`；独立验证器重算9个Engine、180正式批、
NPZ实际工作负载和配对比例，原始数据留在 `gpu_runs/p10-full-ab-timing-hgx03-20261012-01`。

### 17.2 P11：实际GPU时间线与来源，而非按名字猜测

A/C 各选相同32个decode steps，保存rank0/4原始Nsight Systems报告和SQLite。
GraphNodeId按进程区分，沿clone父链关联到同线程最内层来源NVTX，所有采样图内
kernel都有来源；rank0/4均从A的57088个图内kernel增加到C的64768个，多7680个。
这对应当前Full窗口，不能与历史eager的52210个GPU事件混作同一个计数口径。

| rank0 / 32 steps | A ms | C ms | 解释 |
|---|---:|---:|---|
| GPU活动并集 | 456.066 | 814.080 | 同时考虑跨stream重叠，仍不是整轮generate |
| GPU kernel duration sum | 457.880 | 815.903 | 不能相加解释关键路径 |
| 所选GPU活动首尾窗口 | 765.127 | 1141.176 | 包含间隙，不能把每个间隙都归因CPU |

来源归因显示 Router、QKV/O、attention 与专家计算均有实际GPU差异，通信差异相对小。
Router只有1个CTA、QKV20、O16，相对108SM的A100是很小的launch规模；这属于
launch metadata证据，未测achieved occupancy，不能把它称为硬件占用率测量。
CPU `cudaGraphLaunch` 在profiler下明显膨胀，无profiler性能结论采用P10/P12。
图内D2H为0；图外sampling/metadata/prefill仍有拷贝，不称整个请求无CPU同步。

选定NCU指标实际返回 `ERR_NVGPUCTRPERM`。用户明确取消剩余采集后，P11按
调整范围验收；occupancy、SM利用率和memory throughput保持未测。
O模块输出被后续inplace改写的问题已通过独立、逐bucket立即保存的raw GEMM
快照解决：24份真实矩阵，112128输出元素和适用批次前缀逐bit门禁通过。
失败的原始O快照没有被新证据追认为正确。

证据：[P11完整归因](P11_FULL_GRAPH_PROFILE_20261012.md)、
[原始投影门禁](P11_RAW_PROJECTION_GATE_20261012.md)。成功提交 `89cb2f2e`、
`52196c0b`，用户范围调整提交 `3545a52d`。失败NCU实现没有进入main。

### 17.3 P12：先保持归约顺序，再验完整模型和实际性能

QKV/O使用BM16/BN64/BK64，保留原K归约与累加顺序；Router沿用已验证的小tile。
残差RMS和联合Q/K norm/layout均在捕获前安装，逐rank/bucket记录实际图中候选调用。
七种候选完整模型有限门禁通过；RMS on捕获95次、off为0，Q/K on为48次，
Router on覆盖48层，QKV/O两类各48层。正式运行还证明各rank实际full回放，无纯decode回退。

| 同条件Full方案，首轮pilot | P50 s | P95 s |
|---|---:|---:|
| Reference，原QKV/O、RMS on | 3.390534 | 3.402511 |
| QKV/O小tile、RMS on | 3.075585 | 3.124919 |
| 原QKV/O、RMS off | 3.666562 | 3.809668 |
| 小tile、RMS off | 3.342247 | 3.382743 |
| 小tile + Q/K、RMS on | 3.050546 | 3.193955 |
| 小tile + Router、RMS on | 2.751210 | 2.889395 |
| 小tile + Q/K + Router、RMS on | 2.715794 | 2.852725 |

最终选择小tile + RMS + Q/K + Router。与Reference的三轮配对变化分别为
−19.9007%、−20.2348%、−19.9317%，中位−19.9317%；不是在线RL端到端提速。
RMS关闭在两种投影路径均变慢，不支持“剩余慢主要来自RMSNorm”的归因。
共11次独立Engine、220正式批；160个候选/参考批对的tokens/routes相同，
163840个target logprob元素逐bit一致。所有GPU快照、测量和计数由独立验证器重算。

条件性Split-K2/4对16份实际BF16矩阵、110592输出元素分别出现155/159个bit
mismatch，每种均16/16矩阵有差异。尽管微基准可能更快，严格alignment拒绝它们。
失败Split-K源码只在忽略目录，不进入main。成功实现和报告提交 `9419498a`。
证据：[P12完整报告](P12_FULL_GRAPH_TILES_FUSIONS_20261012.md)。

### 17.4 P13：真实更新生命周期完成；P14仍待GPU验收

P13的推理隔离生命周期补充测试已经实际执行：435个模型参数和96个KV张量的
地址被记录，synthetic原地norm更新后旧图与重新捕获相同，sleep/wake恢复相同，
地址替换负例拒绝旧图并可重建恢复。它不替代两轮真实Slime/Megatron更新。
真实在线测试保留v89的StatelessAdam/reset states和全量权重同步；用真实自然生成
的8×128 response，每个版本与训练端1024 target logprob bits比较。
原random reward曾使组内两条奖励相同、GRPO advantage和梯度全为0，该轮被拒绝。
当前诊断奖励固定每两样本0/1，仍执行真实前后向、优化器和权重同步；不代表任务
学习效果，也不冒称原随机reward的结果。路由器503失败也不计作成功训练回归。

同步路径必须按实际源码和运行检查：CLI `--update-weight-transport nccl` 在
`--colocate` 条件下选择 `UpdateWeightFromTensor`，含训练侧NCCL收集及Gloo/IPC
向SGLang投递，不能只凭CLI字符串称最后投递为纯NCCL广播。验收重点是完整
权重同步、实际更新后旧图读取新值、地址/graph pool生命周期，以及版本0/1/2
严格logprob门禁。成功运行 `p13-full-online-hgx03-20261012-06` 已完成该回归：
三版本各1024个target logprob bits零差异，8rank前两轮梯度均非零，抽样master
参数变化分别47250–69715、44490–65662个元素。所有rank全量同步后旧graph
object/pool ID保持一致，旧图与重新捕获输出相同，v1/v2实际greedy输出与前一
版本不同；每rank记录435个weight parameters、96个KV张量和request指针。
地址替换负例实际被拒绝，重建后恢复，sleep/wake与50bucket完整恢复同步均经
独立验收。成功runtime发布到 `gpu_tools/p13_online/`，不是任意partial update
或标准有状态Adam的通用保证。见 [P13完整报告](P13_ONLINE_FULL_GRAPH_LIFECYCLE_20261012.md)。

P14准备cap16/cap8全rank物理发送上界与guard、多batch/变长请求、零接收/idle
路由、完整模型与eager shadow、PyTorch allocated/reserved及NVML驻留采样、
捕获成本和允许范围外拒绝。CPU上界检查或准备好入口均不等于实际GPU验收，
P14当前尚未执行GPU测试。P14完成后会补充独立报告和此处结果，再逐项
审计 [实验计划](P10_P14_EXPERIMENT_PLAN_20261012.md)，才宣布五阶段全部完成。
