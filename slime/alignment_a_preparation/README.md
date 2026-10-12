# alignment_a_preparation：实验入口与历史准备说明

最新整理：2026-10-12。整个目录的工作、结果、实现状态、失败原因和复跑入口见
[详尽总览 WORK_SUMMARY_20261012.md](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/WORK_SUMMARY_20261012.md)。
历史 P9 baseline 完整 decode Graph 测量为1.890481秒，较同轮最佳Breakable的3.093301秒
降低38.885%；prefill仍eager。配置和验证边界见
[P9_BASELINE_FULL_DECODE_20261012.md](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P9_BASELINE_FULL_DECODE_20261012.md)。

P10–P14 的当前验收状态如下；未完成项不会由历史结果代替。

| 阶段 | 当前结论 | 报告或验收要求 |
|---|---|---|
| P10 | 同轮 Full/Full 三模式完成；C/B 配对额外开销中位 +69.939% | [P10报告](P10_FULL_GRAPH_DETERMINISTIC_AB_20261012.md)，包含真实 route 回传成本 |
| P11 | A/C 32 decode steps 时间线归因完成；用户取消剩余 NCU 硬件计数器采集 | [P11报告](P11_FULL_GRAPH_PROFILE_20261012.md)、[原始投影门禁](P11_RAW_PROJECTION_GATE_20261012.md) |
| P12 | 小tile、残差 RMS、Q/K、Router 叠加，完整 generate 配对延迟下降中位 19.93% | [P12报告](P12_FULL_GRAPH_TILES_FUSIONS_20261012.md)；Split-K 有 bit mismatch，未发布失败实现 |
| P13 | 真实两轮非零更新后回归通过；v0/v1/v2 各1024 target logprob bits零差异 | [P13报告](P13_ONLINE_FULL_GRAPH_LIFECYCLE_20261012.md)，含旧图/重捕获、sleep/wake、地址替换恢复及诊断奖励范围 |
| P14 | 工具已准备，GPU 验收尚未运行 | cap8/16 全rank上界、模型多batch、零接收/idle、显存与捕获成本、边界拒绝 |

P10/P12 与 P9 的 route 观测及配置不同，不能直接相除发布新的额外开销。
范围和最终验收清单见 [P10–P14计划](P10_P14_EXPERIMENT_PLAN_20261012.md)。

以下正文保留2026-09-09 CPU准备阶段的原始实验设计，其中“未完成”“下一步”等
均为当时状态，不代表后续GPU实验仍未完成。早期启动记录保留于 `gpu_runs/PROGRESS.md`、
`gpu_runs/NODE_STARTUP_DIAGNOSIS.md` 和 `gpu_runs/LAUNCH_FIXES.md`，当前结论以总览
及对应已完成阶段报告为准。

## 先看这部分：我们接下来究竟比较什么

我们要让两个版本处理同一份数据：一个关闭你的 Qwen3 对齐处理，一个开启。先问“精确一致性需要多少训练时间和显存”，之后再问“更多轮更新、更长响应是否仍然一致”。

第一轮不比较模型答题能力，不做 RoutePack，也不追求某个预设加速百分比。

准备好的第一份输入来自 v89 的第一轮：8 条样本、每条响应 128 tokens。总训练输入为 **2,242 tokens**，其中 **1,024 tokens** 是 response/target。两边必须使用同一份文件和同一个起始 checkpoint。

比较对象称为：

- **native-controlled：受控原生路径。** 在当前带有项目补丁的同一套环境里关闭 Qwen3 alignment；保留确定性模式、相同硬件拓扑和优化器。
- **aligned-controlled：受控对齐路径。** 使用已有完整 Qwen3 alignment。

这回答的是“在固定软件栈和确定性条件下，开启对齐方案的增量代价”。它不是未经修改的上游版本比较，也不能代表最快的非确定性生产配置。未来如需对比后者，另立实验。

## 已完成的准备与新发现

### 1. 保存的数据确实还能复核

已用现有 `compare_qwen3_logprobs.py` 的比较函数在 CPU 上重新计算，设置 `CUDA_VISIBLE_DEVICES=''`，加载使用 `map_location='cpu'`，结束时 `torch.cuda.is_initialized()` 为 false。

| 保存数据 | 样本数 | 实际输入 token | 目标 token | logprob 最大差异 | 不同 token 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v89 step 1 | 8 | 2,242 | 1,024 | 0 | 0 |
| v89 step 2 | 8 | 2,166 | 1,024 | 0 | 0 |

这是对历史 tensor dump 的独立复算，不是当前代码再次完成 GPU forward 的证明，也不能重新证明历史参数更新或同步过程。

证据：CPU 汇总（本地产物：`/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/cpu_validation/summary.json`）、step 1（本地产物：`/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/cpu_validation/step1.json`）、step 2（本地产物：`/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/cpu_validation/step2.json`）。

这两份 rollout 的 `rollout_routed_experts` 均为空。因此本次固定数据 replay 不等于 Routing Replay；A 不开启 R3，也无需为 RoutePack 新采 trace。

### 2. 历史“Adam 更新”需要更精确的描述

v89 启动器同时使用 `--use-stateless-adam` 和 `--reset-optimizer-states`。本地 [stateless_adam.py](/public/home/yingxy2024/moe-consistency/slime/slime/backends/megatron_utils/stateless_adam.py:7)明确说明，这相当于每步从零一阶/二阶矩开始的 Adam 更新，不保留跨步动量。

所以历史成果可描述为“在 StatelessAdam/每步重置动量配置下，参数更新后的下一轮仍对齐”。不能据此宣称已经验证了标准有状态 Adam 的长程稳定性。

第一轮性能对照两边都保留这个优化器，避免一起改变数值路径和优化器。标准 Adam 单独放到 A 的扩展阶段，需要去除 stateless/reset 开关，并重新检查状态内存与 offload 行为。

### 3. 真正使用的依赖路径和想象的可能不同

v89 环境声明的 PYTHONPATH 依次为：

1. `/public/home/yingxy2024/moe-consistency/experiments/sglang-main-sm80/python`
2. `/public/home/yingxy2024/slime-deps/Megatron-LM`
3. `/public/home/yingxy2024/moe-consistency/slime`

不能用工作区根目录下另一份 Megatron/SGLang 的 commit 代替。已保存依赖来源、关键文件 SHA-256 与 tracked diff；Megatron 存在用户已有补丁，必须原样保留，因此对照名称使用“受控原生”。

当前系统 Git 较旧，直接进入 SGLang worktree 不能识别；准备时使用显式 shared gitdir、worktree、index 和 commit 只读提取差异，没有修复或改写仓库。见 [source_manifest.json](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/source_manifest.json)。

这里只核对了文件系统和已安装包 metadata。GPU 作业启动时还要打印 worker 实际加载的 `__file__` 与版本，确认没有被 site-packages 或远端环境覆盖。

### 4. 历史脚本不能直接作为新 benchmark 启动器

原 `run.sh` 会轮询 GPU、启动或复用 Ray，并使用固定 submission ID 和固定输出目录。本次仅读取，没有执行。

已发现：

- `/tmp/slime-cuda-runtime-qwen-align-v41` 当前不存在；历史脚本可创建它，但新实验应使用独立路径。
- 历史脚本重复设置 `--moe-token-dispatcher-type`，先 `alltoall` 后 `flex`；新配置只保留明确的最终 `flex`。
- `--num-rollout` 是训练循环的排他上界。历史配置 `3` 覆盖 step 1、2；新单步 recipe 使用 `2` 的前提是有效 `start_rollout_id=1`。启动后必须核对，不凭作业 succeeded 判断执行步数。
- 保留目录只有小型 rollout/train dump、JSON 和极短 launcher 日志，没有更新后模型 checkpoint 或完整 optimizer snapshot。step 2 数据不能直接配初始 checkpoint 做 exact 重放。
- HF config、DCP checkpoint iteration 标记、数据文件和 Python 可执行文件路径存在；没有扫描完整大权重或验证其 hash，所以暂不声称模型完全就绪。

## 原生与对齐配置差异表

配置以新进程分别启动；已 monkey-patch 的模型不能靠进程内改环境变量恢复原生。

| 位置/开关 | native-controlled | aligned-controlled | 原因 |
| --- | --- | --- | --- |
| before-log-prob Qwen hook | 移除 | 保留 | hook 安装 Router/MoE/attention 等处理 |
| before-train-step Qwen hook | 移除 | 保留 | 避免训练前重新安装 |
| `--qwen3-alignment-layers` | 完全移除，解析为 `None` | 0–47 | loss 以 `is not None` 启用特定路径，空列表不等于关闭 |
| 五个 Qwen attention/norm 环境开关 | 全设 `0` | 全设 `1` | 控制输入 norm/QKV、QK norm、pre-MLP norm、attention core、output projection |
| `MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS` | `0` | `1` | final RMSNorm 等路径的 guard |
| Router GEMM、Triton MoE、DeepEP bridge | 不安装 Qwen hook | hook 安装 | 不能仅关闭 attention 就声称原生 |
| LogSoftmax | 原生 PPO 实现 | SGLang batch-invariant 实现 | 由 alignment layer 参数控制 |
| temperature=0.6 的除法 | Python scalar | device FP32 tensor | 同样由 alignment layer 参数控制 |
| deterministic mode、EP=4、其余并行参数 | 相同 | 相同 | 第一轮先测对齐方案增量代价 |
| checkpoint、样本、mask、reward、batch、padding、recompute | 相同 | 相同 | 保证工作量和训练语义可比 |
| StatelessAdam、state reset、CPU offload | 相同 | 相同 | 不同时引入优化器变量 |
| Routing Replay / int4 QAT | 关闭 | 关闭 | v89 数据无 route trace；排除继承环境污染 |

完整开关见 [comparison_recipe.json](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/comparison_recipe.json)。这是**声明式方案，不是已验证的启动配置**。还应排除其他 inherited hooks、legacy DeepGEMM layer 参数和数值环境开关。

代码依据：[Qwen hook](/public/home/yingxy2024/moe-consistency/slime/slime/backends/megatron_utils/alignment/qwen3.py:640)、[temperature](/public/home/yingxy2024/moe-consistency/slime/slime/backends/megatron_utils/loss.py:103)、[LogSoftmax 选择](/public/home/yingxy2024/moe-consistency/slime/slime/backends/megatron_utils/loss.py:597)。

**DeepEP 配置仍需特别核实：**Qwen hook 校验声明的固定配置，但仅存在 `SLIME_QWEN3_FIXED_DEEPEP_CONFIG` 环境值不证明实际 dispatch 用到了它。原生路径会调用 `Buffer.get_dispatch_config` 等 API。GPU 阶段要记录两边实际选择的配置；若有差异，应列为整个方案的一部分，不能把耗时差全部归因于精度计算。

## 第一轮实验步骤：小范围证明比较成立

以下是未来流程，当前均未启动。

### 步骤 1：恢复当前环境下的 v89

使用独立 run ID、输出路径和临时 CUDA 构建目录，复现原有两轮配置，先检查原有 exact gate。保留 `lr=1e-6`、StatelessAdam 和其余原设置。

验收：真实执行两轮、非零 gradient、更新/同步证据、两轮 logprob gate 通过。若失败，先定位环境或行为漂移，暂停性能比较。

### 步骤 2：对同一份第一轮数据运行两条路径

固定使用 v89 `baseline/rollout/1.pt` 和初始 checkpoint。使用 `--load-debug-rollout-data`；代码会进入 train-only 模式，不创建 SGLang servers，但对齐训练仍需导入 SGLang 的算子。

先将两边学习率都设为 0，让权重不变，便于排除参数漂移。这仍可执行非零 advantage 的 backward/optimizer 路径，但只能叫固定权重重放，不能当成真实更新实验。

在独立进程中先运行 aligned，再运行 native，保存小型 train dump：aligned 应复现同权重 logprob 一致；native 记录真实偏差，不要求为零。两边都应检查 token/mask 语义、有限 loss/gradient、执行步数和实际生效开关。

若 native 在确定性配置下不可运行，明确记录原因，另行设计替代基线；不能偷偷移除 deterministic 开关后继续按同一比较解释。

### 步骤 3：正确性通过后才计时

关闭 `save-debug-train-data`、`save-debug-rollout-data` 与所有 tensor dump，另运行计时 pass。正确性 pass 的 dump 会触发 logprob capture，不能拿其耗时当作正常训练吞吐。

先测单次 pilot 的资源与时长，再决定预热/重复数量。建议起点为 5 次预热、20 次测量、3 个独立运行；当前单步 recipe 尚不具备重复计时能力，需要下一轮补受控 harness，不能直接把 `num-rollout` 改大就宣称完成 benchmark。

记录完整训练 step 耗时、输入 token/s、峰值 allocated/reserved 显存及波动。学习率仍为 0，其他配置保持一致。后续再另测在线 `lr=1e-6` 更新的完整开销。

**计时注意：**现有 `perf/actor_train_time` 来自 CPU wall-clock timer，包含训练函数范围内的 backward/optimizer 等工作；本地 `config.timers=None`，不能假设已有完整 GPU 阶段计时。现有 `perf/step_time` 是 train wait 与 train 时间之和，也不是天然严格的 rollout→更新同步端到端窗口。

正式计时需要校验 GPU 完成边界与所有 rank 的最慢完成时间，分别测 CPU wall time 和 GPU 阶段，且不要把每层强制同步用于正常吞吐测量。峰值显存应在定义好的窗口 reset/read，并跨 rank 汇总。这部分本次只设计，没有修改生产计时代码。

### 步骤 4：看结果再扩大

这份 8-sample 小数据首先用于“比较方法是否正确”，不是饱和吞吐。可能受 512-token padding 和 CPU offload 显著影响，结果必须带上这些条件。

之后再商量代表性 batch 和长度，一次只改一个维度。响应从 128 扩至 512 时，不能只改 response cap：旧 `rollout-max-context-len=512` 还包含 prompt，必须一起明确新的总上下文预算并重做正确性 gate。

标准有状态 Adam、更多在线轮数、长响应分别作为后续小实验。新增优化器状态可能改变 CPU/GPU 内存与耗时，不能将其结果混进第一轮 StatelessAdam 对照。

## 当前交付文件

| 文件 | 用途 |
| --- | --- |
| 本 README | 面向讨论的实验说明与事实边界 |
| [artifact_inventory.json](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/artifact_inventory.json) | 历史数据、启动文件 SHA-256 与包版本 metadata |
| [source_manifest.json](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/source_manifest.json) | 真正引用的仓库路径、版本和审计文件指纹 |
| [v89_cli_static.json](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/v89_cli_static.json) | 未执行 shell 的静态参数提取，含重复参数提醒 |
| [comparison_recipe.json](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/comparison_recipe.json) | 两条路径的声明式差异、正确性和计时 pass |
| [results_template.csv](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/results_template.csv) | 待测结果表；空值表示未测，不是 0 |
| `cpu_validation/` | 本次 CPU 重新计算的结果 |
| `source_snapshots/` | 历史启动文件文本副本与 tracked diff；不是可执行 launcher |

**准备已完成；A 的 GPU 复现、性能表和稳定性实验还未完成。** 下一次先讨论这组受控比较是否符合你的目标，再安排有预算的 GPU smoke。无需你现在理解所有开关，也无需立即投入长时间训练。
