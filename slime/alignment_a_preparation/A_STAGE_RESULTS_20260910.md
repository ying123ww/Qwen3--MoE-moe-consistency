# A 阶段首组实验结果 — 2026-09-10

已完成在线两轮更新一致性复现、固定 rollout 正确性对照，以及两种路径各一次独立启动、预热后20步的Megatron training-phase step测量。

在当前8×A100、EP=4、确定性模式、固定8样本的配置下，开启对齐把1024个target tokens中的344个不一致降为0；训练step中位数增加 **2.84%**。这支持“以可测的小幅training-phase step开销实现精确一致性”，不支持“显著提高训练吞吐”或“改善RL收敛”的说法。

## 实测结果

| 指标 | native-controlled | aligned-controlled |
|---|---:|---:|
| 正确性 pass mismatch / 1024 targets | 344 | 0 |
| 正确性 pass max_abs(logprob) | 0.79167509 | 0 |
| 正确性 pass mean_abs(logprob) | 0.01412920 | 0 |
| training-phase step中位数（秒） | 17.4537 | 17.9502 |
| training-phase step P95（秒） | 17.8285 | 18.2958 |
| 输入 tokens/s | 128.45 | 124.90 |
| PyTorch peak allocated 最大rank（GiB） | 50.831 | 50.963 |
| PyTorch peak reserved 最大rank（GiB） | 52.537 | 52.896 |

开启对齐的吞吐变化为 **-2.77%**，allocated峰值差值约 **0.132 GiB/rank**。显存口径不含完整NVSHMEM/驱动开销，不能当作nvidia-smi总占用。

## 在线复现

`a-restore-20260909-06` 正常结束（SUCCEEDED）。CPU独立复算step1/2保存数据：每轮8样本、1024 targets、max_abs=0、0 mismatch、无非有限值。日志记录两轮非零梯度范数8.39038449、8.00749237，lr=1e-6，并完成权重同步。

此处优化器为StatelessAdam，每步重置动量；不能描述为标准有状态Adam长期稳定性实验。

## 测量方法与边界

- 两种路径使用同一软件栈、初始checkpoint、历史rollout1；输入文件SHA256为 `ed72c22c9d16e13d455f83af09bda05420ad5e81d678c39a800bd721a550204d`。native是关闭Qwen对齐的受控路径，不是未经修改的上游最优配置。
- 8×A100-SXM4-80GB、TP=1、EP=4；8样本，2242个未填充输入tokens、1024个target tokens。输入吞吐按2242/step中位数计算，不是填充后的实际算子token数。
- 两种路径均lr=0、StatelessAdam、CPU optimizer offload，使用相同确定性设置；八rank审计确认aligned有48层MoE包装、native无对齐包装。
- correctness pass保存logprob供比较；timing pass关闭debug dump。二者分开运行。timing没有逐步重新保存并比较logprob；不能宣称20个计时step都做了独立数值gate。
- 每组26个训练step：第1步用于安装测量包装且不计时，第2–6步预热，第7–26步统计。所有rank记录完整，单步取八rank耗时最大值。
- 同步CUDA后计时完整train_one_step，含前后向、优化器、梯度清零和指标归约；不含外层数据准备、rollout、offload/onload和dump。因此不是在线RL端到端吞吐。
- 每组仅一次独立启动，且aligned先于native。当前差异是描述性实测，尚未排除跨运行波动或顺序效应；不宣称统计显著。
- 固定小批量、CPU offload的开销会掩盖部分算子差异，不能外推到大batch或不同并行拓扑。
- 900 GiB独立内存保护持续运行且未触发。采样期间观察到作业内存约782–784 GiB，但没有连续峰值历史，不能把它当精确峰值。

## 原始证据

- 在线复现：`gpu_runs/a-restore-20260909-06/`，含correctness-summary.json与job.log。
- 正确性对照：`gpu_runs/a-{aligned,native}-correctness-20260910-01/`。
- 计时：`gpu_runs/a-{aligned,native}-timing-20260910-02/`，含八rank审计、原始JSONL和measurement-summary.json。
- 对照JSON/CSV：`gpu_runs/comparison-20260910.{json,csv}`。
- 当前工具和SGLang临时路径补丁哈希：`gpu_runs/measurement-source-hashes-20260910.json`。基础依赖修订和原有补丁见source_manifest.json；此次额外修改numa_utils使临时启动脚本尊重TMPDIR，不改数值计算。

## 简历表述建议

在8×A100、EP=4的受控小批量实验中，将跨runtime目标token logprob不一致由344/1024降至0；完成两轮StatelessAdam在线更新及权重同步后的精确一致性复现。首组性能测量中，开启对齐的Megatron training-phase step中位耗时增加约2.84%（每种路径预热后20步）。

## 下一阶段

首组性能比较已完成；整个项目计划尚未全部完成。下一步优先进行至少3组独立重复、交替执行顺序，确认约3%差异的稳定性；之后再扩展batch/响应长度和在线更新轮数。RoutePack和RL收敛改善尚未实施或验证。

## 推理侧补测（2026-09-10）

三组 SGLang 独立推理对照已完成：固定 8×128 输出 token 的生成耗时增加 23.56%、23.86%、23.02%，平均 23.48%。本报告的 2%–3% 仅为 training-phase step 增量；两者不能相加，也不是端到端 RL 开销。推理协议、baseline 区别及 PR #2262 对比见 [推理补测报告](INFERENCE_AND_PR2262_20260910.md)。
