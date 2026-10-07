# AutoPack 真实长度工作负载：原生调度验证与离线交换结果

日期：2026-09-14。GPU 节点：`ai_hgx_05`；离线计时主机：`debug02`。

## 当前结论

把之前固定生成 128 tokens 的短样本换成标准训练 recipe 的生成配置后，原生布局的 padding 比例从 **28.67% 降至 1.3625%**。去掉 padding 和 synthetic terminal 后，真实路由的逐 collective、逐层 EP 峰均比仍为 **p50 1.192、p95 1.580、最大 1.894**。因此这批数据的 EP assignment 偏斜不是短样本 padding 造成的假象。

在同一份真实路由、原生布局和固定搜索预算下，seed 0 的 AutoPack 将逐层 collective excess 从 **36,957,810 降至 35,802,633（下降 3.1257%）**；相同 18 次交换的 Random-Swap 为 **36,684,722（下降 0.7389%）**。五个 search seeds 的降幅为 **2.9491%–4.0581%**，全部优于匹配交换次数的 Random 基线。这支持离线负载代理指标改善，**不证明 slow-rank time 或训练 step time 改善**。

当前实现的 CPU 成本很高：串行测量的搜索中位数约 **2.913 秒**，完整离线调用约 **81.600 秒**，其中准备和重新校验分别约 39 秒。它仍是离线原型，不能据此宣称可直接放进在线训练。

## 数据来源与实际运行口径

配置来源为仓库 `scripts/run-qwen3-30B-A3B.sh`：32 个 prompts × 每 prompt 8 个回答，共 256 个 samples；response cap 8192、temperature 1、top-p 1、top-k -1、global batch 256、动态 microbatch、max_tokens_per_gpu 20480。真实数据集为 DAPO-Math-17k，通过实际 `Dataset.shuffle(epoch=0)` 和默认 rollout seed 42 取前 32 个 prompt groups，不按长度筛选。

这是**参照当前标准 recipe 进行的一次 workload 实跑**，不是已验证的历史 production run。实际保留此前数值对齐通过的 **TP1/EP4** 拓扑，而非 recipe 的 TP4/EP8。不能把这份结果外推为所有 batch、其他拓扑或稳态训练表现。

最终采用 capture02：每个 prompt 内的 8 个回答显式使用 seeds 42..49，得到 **256 条不同完整 token 序列**。不显式传 seed、可能出现重复 sibling responses 的 capture01 已排除。总 prompt tokens 39,896，response tokens **1,599,412**，完整 tokens **1,639,308**；145 个回答确认 EOS，111/256（43.36%）达到 8192 长度上限。长度上限截断是真实测得的 finish reason，不是依据 token 数推断。该生成分布仍受 8192 上限截断。

| Tokens / sample | p10 | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|
| Prompt | 103 | 152.5 | 197 | 243 | 261 |
| Response | 3,197 | 6,652 | 8,192 | 8,192 | 8,192 |
| 完整序列 | 3,355.5 | 6,791.5 | 8,386 | 8,391 | 8,435 |

## 原生调度与 B1/B2 验证

B1-02 在 8 ranks 全部通过：每 rank 12 slots，共 96 bins、2 个 EP groups、24 个 collectives、48 层。1,599,412 个 response token 的 logprob 在 Engine capture、aligned Megatron fallthrough、R3 replay_forward 间 bitwise 一致；实际 router/dispatch mismatch 为 0。

B2 独立重读 18,432 个 route arrays、316,896,384 个 route rows 和 4,798,236 个保存的 logprob 值，确认实际 dispatch H/U/per-expert 计数与重建结果一致。这里验证的是**原生 schedule 的 routes 和前向数值**，没有执行 backward 或 optimizer，也没有验证交换后的 schedule。

B1-01 在 full logits 转 FP32 时 OOM。B1-02 只添加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后完成，精度、token shapes、batch、max_tokens 和拓扑不变；这不是单独的性能优化效果测评。详见 [B1 验证记录](AUTOPACK_REAL_B1_20260914.md)。

## Padding 与 EP 负载

训练 tokens 1,639,308，layout padding tokens 22,644，padding 占 padded total 的 **1.362494%**。之前短样本运行的 28.67% 仅作工作负载差异对照，不是 AutoPack 已降低 padding 的收益。

所有峰均比先在同一个 `(EP group, slot, layer)` 内计算 `max_rank H / mean_rank H`，再对 24 × 48 = 1,152 个值取分位数，未将不同层混成 histogram。

| H 口径 | p50 | p95 | max | Excess E |
|---|---:|---:|---:|---:|
| 含 layout padding、synthetic terminal | 1.1904 | 1.5788 | 1.9197 | 36,957,810 |
| 扣 layout padding、保留 terminal | 1.1918 | 1.5801 | 1.8947 | 37,245,999 |
| 原始 routes：同时扣 padding 和 terminal | 1.1917 | 1.5804 | 1.8945 | 37,254,095 |

本批数据中 padding 甚至略微降低总 excess，说明不能用“减去 padding 一定降低 E”解释结果。H 仍是计算负载代理；没有在这里测通信时间、send/recv skew 的时间影响或 slow-rank 因果链。

## Native / Random-Swap / AutoPack

目标为 `E = sum_(group,slot,layer) [max_rank H - mean_rank H]`，包含候选布局对应的 terminal 和 layout padding。预算为最多 500 candidates、20 iterations、20 swaps，每轮随机抽取最多 25 candidates 后选 best-improvement。只接受满足完整 sample、step/slot 和容量约束的改善交换。Random 使用匹配 AutoPack 的交换次数，不要求 E 下降。

| Search seed | Native E | Random E | AutoPack E | AutoPack 降幅 | Random / AutoPack swaps |
|---|---:|---:|---:|---:|---:|
| 0 | 36,957,810 | 36,684,722 | 35,802,633 | 3.1257% | 18 / 18 |
| 1 | 36,957,810 | 36,538,206 | 35,661,221 | 3.5083% | 20 / 20 |
| 2 | 36,957,810 | 35,971,096 | 35,458,023 | 4.0581% | 19 / 19 |
| 3 | 36,957,810 | 36,354,276 | 35,720,421 | 3.3481% | 20 / 20 |
| 4 | 36,957,810 | 36,630,154 | 35,867,869 | 2.9491% | 20 / 20 |

五个 search seeds 均低于对应的相同交换次数 Random 基线；AutoPack 相对 Native 降幅为 **2.9491%–4.0581%**。这些 seeds 使用同一个 batch，并不等价于五个独立训练 batches。Seeds 1..4 各运行一次且并行执行，其 CPU 时间不用于性能比较。

## CPU 开销与确定性

下表仅采用 **debug02 上 seed 0 的串行测量**，一次预热后 3 次正式重复。3 个样本的 p95 仅是本组观测分位数，不是稳定尾延迟估计。

| 阶段 | p50 (s) | p95 (s) |
|---|---:|---:|
| Prepare：原始 routes histogram 准备等 | 39.416 | 39.530 |
| Swap search | 2.913 | 3.039 |
| 完整重新校验与报告 | 39.276 | 39.871 |
| 完整离线调用 | 81.600 | 82.187 |

上述时间是当前离线工具的工作量，不包含 GPU 生成/routes 采集，也不是在线 scheduler 的已测时延。搜索约 2.9 秒本身仍需要优化或摊销；不能仅去掉离线校验的 39 秒就声称已满足在线预算。各阶段分位数不能直接相加作为总分位数。

Seed 0 三次正式重复最终 schedule hash 一致：`9ccbfc65d6ebe4259d664dbc4b9aeee07d1f4eddde1ab67a51e6da5aecfef82b`。这里的确定性证据限于这个 seed/输入/固定候选数预算的三次重复，不代表跨平台保证。其他 seeds 的单次运行不新增重复确定性证据；wall-clock 截止触发时也可能改变最终 schedule。

## 下一步需要补充的效果

- 接入交换后的 schedule，验证前向、梯度和参数更新一致性。
- 同样真实 batch 下比较 Native、相同交换次数的 Random 与 AutoPack 的 GPU slow-rank、collective、step time。
- 测量并计入在线 routes 获取、histogram 构造和 scheduler CPU 成本，给出实际净收益。
- 扩展独立 batches 和重复 GPU runs，避免把单个 batch 或 seed 当作稳定效果。

在这些结果完成前，本报告可以证明真实长回答工作负载存在 EP assignment 偏斜，以及 AutoPack 在该样本上的离线 E 改善；不能证明训练加速。

## 证据文件

根目录：`alignment_a_preparation/gpu_runs/autopack-real-hgx05-20260914/`。

- [长度与终止原因](gpu_runs/autopack-real-hgx05-20260914/capture-summary.json)
- [原生 H/padding/collective 统计](gpu_runs/autopack-real-hgx05-20260914/native-summary.json)
- [Seed 0 三次串行重复与交换轨迹](gpu_runs/autopack-real-hgx05-20260914/exchange-seed0.json)
- [Seed 1](gpu_runs/autopack-real-hgx05-20260914/exchange-seed1.json)、[Seed 2](gpu_runs/autopack-real-hgx05-20260914/exchange-seed2.json)、[Seed 3](gpu_runs/autopack-real-hgx05-20260914/exchange-seed3.json)、[Seed 4](gpu_runs/autopack-real-hgx05-20260914/exchange-seed4.json)：同 batch 离线质量对照
- [原生调度 B1/B2 验证](AUTOPACK_REAL_B1_20260914.md)
- [输入与来源配置](gpu_runs/autopack-real-hgx05-20260914-input/prompts.json)

重型原始数组保存在 ai_hgx_05 节点临时目录，路径见 B1 记录；清理这些数组前需要单独安排归档。GPU 正式实验已由 root 完成并恢复授权保号任务，当前 PID/日志以 root 的最新交接为准；本离线报告未切换 GPU 任务。
