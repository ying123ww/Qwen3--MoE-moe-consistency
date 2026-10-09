# Qwen3 residual RMSNorm 融合验证（2026-10-08）

## 范围和状态

三轮、9个独立Engine、180批正式完整generate计时已完成。配对延迟下降中位数10.926%，各轮为10.796%–11.339%；候选仍比同轮baseline慢10.614%–10.962%。真实模型逐bit门禁通过。

候选默认关闭。仅在 CUDA deterministic/native 分支、Qwen3 decoder 的 input/pre-MLP RMSNorm、连续 BF16 input/residual、H2048、M1..4、BF16/FP32 weight、cast-after-weight 下启用。额外 residual、FP32 residual、variance override 等情况保留原路径。CUDA graph 本轮关闭；未做跨框架训练验证。

本轮采用逐 bit 门禁，shadow 比较归一化输出和可见 residual 两个张量；candidate 真实计时路径关闭逐调用审计。任何差异直接失败，不允许进入性能计时。

## 初版失败与修复

单卡初版在688,128个BF16输出/残差元素上零差异，wall约130→24µs。真实模型仍在 rank0/layer39/input、shape[2,2048] 的第三次调用出现1bit差异。因此初版不能使用，未启动完整性能计时。

证据分别在首次模型门禁（本地产物：`gpu_runs/rms-perf-hgx05-20261008-01/`）与带失败张量的重现（本地产物：`gpu_runs/rms-repro-hgx05-20261008-01/`）。重现 rank4/6 的失败张量保留在 exact-shadow/audit/*.pt。具体输出坐标[1,446]：reference=-0.0927734375，candidate=-0.09228515625；residual输出仍相同。

分析编译后的 Triton IR：原 FP32 mean load 使用每线程4个连续元素；融合BF16 load默认选择8个，使归约树改变。方差出现FP32 1ULP差异，经过倒数、乘weight与BF16舍入后落在不同值。

修复用 `tl.max_contiguous(tl.arange(0, BLOCK), 4)` 约束原始load布局，并保留原版两个1024元素sum及有序累加、FP32算术、禁用FP融合。对上述失败张量，修复后方差、倒数和最终输出bits都一致。该重现修复结果不替代完整模型门禁。

## 重跑

ai_hgx_05原分配结束后，按用户授权的命令重新申请，排除ai_hgx_04。新Slurm作业1028519分配到ai_hgx_03，8×A100-SXM4-80GB、64 CPU、1TiB内存；P2P检查通过，无无关GPU任务。

修复后单卡目录：residual-rms-hgx03-20261008-02（本地产物：`gpu_runs/residual-rms-hgx03-20261008-02/`）。完整门禁和计时目录：rms-perf-hgx03-20261008-02（本地产物：`gpu_runs/rms-perf-hgx03-20261008-02/`）。数值门禁和完整性能对照均已完成；state.json记录complete，所有正式worker和driver109631均已退出，GPU空闲。Slurm作业1028519仍在ai_hgx_03，用户本轮取消保号。

计时协议与router对照相同：8冻结prompts、1218输入tokens，每请求128输出；每Engine预热3批、正式20批，三轮交错 baseline/aligned/candidate，共9个独立Engine。所有模式router候选均off，candidate仅启用本项RMSNorm融合。测量完整blocking generate；模型加载、预热、cache-flush重试均在计时区间外。

运行环境为Python3.12.13、PyTorch2.11.0+cu129、Triton3.6.0；SGLang使用本项目源码快照（安装包版本标记0.0.0）。版本记录见runtime-versions.json（本地产物：`gpu_runs/rms-perf-hgx03-20261008-02/runtime-versions.json`）。

CPU分派、审计失败快照、数值覆盖和测量控制测试36项通过。失败快照测试使用真实CPU张量并检查signed-zero bit差异被报告并保存。

## 修复后数值结果

单卡688,128个BF16输出/残差元素零bit差异：

| M | 原版 wall µs | 融合 wall µs | 延迟下降 |
|---|---:|---:|---:|
| 1 | 124.204 | 23.201 | 81.320% |
| 2 | 124.656 | 23.103 | 81.467% |
| 4 | 124.899 | 23.039 | 81.554% |

10轮交错顺序，每轮每种方式100次调用；wall包含分配、Python/Triton发射与GPU完成。上述数据是单算子有限合成试验。

真实模型比较（本地产物：`gpu_runs/rms-perf-hgx03-20261008-02/exact-comparison.json`）：model_exact=true、batch_exact=true，failures为空。shadow完成45,980次调用，覆盖760个rank/层/norm类型组合，比较224,133,120个归一化输出及residual的BF16元素，零bit差异、零非有限值；实际eligible形状为M1/2。无审计candidate的完整tokens/routes/logprob bits与off和shadow相同，三个模式自身batch8/3/1同行结果也一致。范围为四组有限greedy输入、每请求16输出tokens、关闭CUDA graph；未据此宣称任意输入或跨框架一致。

诊断证据（本地产物：`gpu_runs/rms-repro-hgx05-20261008-01/analysis/`）保留初版1bit失败与4元素布局修复的方差/倒数对照。

## 完整性能结果

| 轮次 | baseline s | aligned s | RMS candidate s | 配对延迟下降 | candidate比baseline慢 |
|---|---:|---:|---:|---:|---:|
| 1 | 12.654734 | 15.764362 | 14.041924 | 10.926% | 10.962% |
| 2 | 12.711930 | 15.767615 | 14.065290 | 10.796% | 10.646% |
| 3 | 12.693644 | 15.836728 | 14.040993 | 11.339% | 10.614% |

三轮配对延迟下降中位数为10.926%。各模式三轮中位数为baseline 12.694秒、aligned 15.768秒、candidate 14.042秒；该汇总不替代上表各轮配对计算。相对baseline的差距由24.038%–24.761%缩到10.614%–10.962%。

完整结果（本地产物：`gpu_runs/rms-perf-hgx03-20261008-02/comparison.json`）与最终样本核查（本地产物：`gpu_runs/rms-perf-hgx03-20261008-02/measurement-validation.json`）。9组均完整预热3批、正式20批，全部输出固定1024tokens，计时无profiler和逐调用审计。baseline第二/三轮各1次cache flush短暂重试，全部成功且在计时区间外；没有失败样本或续跑拼接。

全部数值/控制源码SHA256保持不变，已保存在source-snapshots（本地产物：`gpu_runs/rms-perf-hgx03-20261008-02/source-snapshots/`）供核查。三轮aligned/candidate各23批输出哈希全部一致，共69/69批；这是采样输出哈希检查，不替代greedy gate的routes/logprob逐bit比较。

图状态以resolved-server-args.json中的`cuda_graph_config`为准：9组prefill和decode backend均disabled，日志明确记录DeepEP normal禁用CUDA graph。旧兼容字段`disable_cuda_graph=False`不能单独代表实际图后端。

## 使用与下一步

默认仍off。启动进程前设置以下变量，并沿用本次已验证的deterministic/aligned配置，才能使用候选：

```bash
export SLIME_QWEN3_RESIDUAL_RMS_MODE=candidate
export SLIME_QWEN3_RESIDUAL_RMS_AUDIT=0
export SLIME_QWEN3_ROUTER_TILE_MODE=off
```

[运行时实现](../../experiments/sglang-main-sm80/python/sglang/srt/batch_invariant_ops/fused_rmsnorm_candidate.py)、[门禁](gpu_tools/router_tile_exact.py)、[完整对照启动器](gpu_tools/residual_rms_performance.py)。在新的8卡分配上复跑，用一个未使用的rms-前缀run-id；换节点不能复用本节点的部分配对性能样本。

这项融合的整轮收益约10.93%，高于[router小tile的约0.38%](P1_ROUTER_TILE_PERFORMANCE_20261007.md)。优先完成该融合的跨框架训练/新输入门禁，再对融合后的版本重新profiling，检查剩余发射热点及QKV/O的小M分块收益。两项候选的组合本轮未测试，不能将各自收益直接相加。

本轮未测RL整轮、训练吞吐或CUDA graph，也未宣称剩余差值都是CPU开销。
