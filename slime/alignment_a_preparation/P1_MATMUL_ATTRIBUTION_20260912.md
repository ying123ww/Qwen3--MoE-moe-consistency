# 推理确定性矩阵乘快速归因（2026-09-12）

**已将同名 `matmul_kernel_persistent` 分成 router、QKV、attention 输出投影与 LM head。** 两个代表 rank 的结果一致：router 约占这类 kernel 累计 GPU 时间的 42%，两个 attention 投影合计约 51%，LM head 约 7%。这不是整轮耗时占比，也没有证明这些占比就是原先 +23.48% 推理增量的分解。

## 本轮实际运行

- ai_hgx_03，Slurm allocation 1000742，8 × A100 80 GB / 64 CPU / 1 TiB。沿用冻结 aligned Engine 配置和实际 SGLang 路径；未修改模型/kernel 数值实现。
- 固定 8 prompts，共 1218 prompt tokens；1 批预热 + 1 批采样，每条生成 16 tokens，共 128 response tokens。采用 CPU/GPU profiler、record_shapes=True，不采 Python stack。该短窗口不与历史 128-token 性能比例直接比较。
- 8 份 GPU trace 导出，进程成功退出。rank 0、4 各 2610 个 persistent matmul kernel，全部通过 CUDA correlation、External id 与 CPU `aten::mm` 输入形状关联，无未归因 kernel。
- 分 prefill/decode 使用 CPU `step[...]` annotation。GPU kernel 的完成时间可以晚于发起 CPU 区间，关联不依赖二者时间窗口重合。CPU/GPU 同名 annotation 不混算。
- 根盘仍满；主要临时目录迁移至项目 `.a-tmp`，编译缓存迁移至 `/dev/shm`。本轮专用临时目录成功后清理，900 GiB guard 未触发。

运行证据：实验目录（本地产物：`gpu_runs/a-matmul-attribution-20260912-01/`），含 manifest、实际 server args、源文件 hashes、资源检查、退出状态、原始 trace、分 rank 汇总与 小型对照 JSON（本地产物：`gpu_runs/a-matmul-attribution-20260912-01/matmul-comparison.json`）。

## 矩阵乘构成

每份 trace 覆盖 2 次 prefill 和 16 次 decode。时间为 GPU 事件累计毫秒；不同 rank 并行，不能横向相加。

| 调用 | 每 rank 次数 | rank 0 GPU ms | rank 4 GPU ms | decode 实际矩阵形状 M×K @ K×N |
|---|---:|---:|---:|---|
| router | 864 | 76.891 | 76.885 | 2×2048 @ 2048×128，FP32 |
| QKV | 864 | 48.856 | 48.108 | 4×2048 @ 2048×2560，BF16 |
| attention 输出投影 | 864 | 44.575 | 44.612 | 4×2048 @ 2048×2048，BF16 |
| LM head | 18 | 11.873 | 12.993 | 4×2048 @ 2048×75968，BF16 |

形状分类同时核对了本地模型 config 与 Qwen3 调用代码。router 的 M=2 与 attention 的 M=4 来自实际记录，不能按请求 batch 大小或 GPU 序号猜测。prefill 形状另保留在分 rank JSON。

rank 0 的 CPU `aten::mm` 区间累计：QKV 92.725 ms、输出投影 84.098 ms、router 85.479 ms、LM head 1.815 ms。这些包括分配、Python/Triton 调度和发射，且与 GPU 执行重叠；不能加到上表或直接称为纯 CPU 开销。LM head 的 GPU 时间大于 CPU 发起区间也符合异步执行。

## 明确的下一项优化候选

1. **先针对 decode 小 M 分块，而不是移除确定性。** 当前通用 kernel 对上述 M=2/4 仍采用 BLOCK_M=128、BLOCK_N=128。router 的 FP32 BLOCK_K=32，BF16 投影 BLOCK_K=64。小 M 下需要检查冗余计算和 CTA 分布；仅观察 tile 形状还不能断言收益。
2. 保持 K 维分块与累加方式，独立比较较小 M/N tile。候选必须先逐 bit 比对原 kernel、检查改变 batch 后同一行结果，再做模型级 exact gate。不能因微基准无误差就直接替换线上 kernel。
3. router 还有 hidden/weight FP32 cast 及 top-k/softmax；本表只统计 matmul，不能用它解释整个 router 或全部推理增量。
4. 本轮只有 aligned shape capture，不能将绝对热点直接说成相对 baseline 的额外开销。完成候选模型回归后，再用不带 profiler 的原三组推理协议验收实际收益。

## 代码和验证

- `gpu_tools/inference_benchmark.py` 新增可选 `--profile-shapes`、`--response-tokens`，原默认 128-token 实验保持一致。
- `gpu_tools/run_matmul_attribution.py` 为独立短窗口入口；`gpu_tools/summarize_matmul.py` 为可重跑归因汇总。
- 汇总测试验证 kernel 异步完成、未匹配 correlation、CPU/GPU annotation 分类；1 项通过。
- 本轮没有执行新的跨框架 exact gate：没有修改数值路径，但有限 logprob 检查不能替代该 gate。

## 补充：小 M tile 已做 GPU pilot

在同一 A100 上独立调用原 Triton kernel，仅比较 launch tile/warps；保持原 K 分块（FP32=32，BF16=64）和数值实现。三类实际 decode 形状、每类 3 个随机 seed，另检查加入一行后原有行结果；全部候选与原 kernel 逐 bit 无差异。这里只覆盖有限合成输入，不是模型级 exact gate。

| 调用 | 原 tile 128×128 / 8 warps | 候选 16×64 / 4 warps | 本次 wall time 变化 |
|---|---:|---:|---:|
| router | 93.35 µs | 60.28 µs | -35.4% |
| QKV | 61.65 µs | 52.16 µs | -15.4% |
| 输出投影 | 57.57 µs | 42.67 µs | -25.9% |

表中为每次调用的中位 wall time，包含输出分配、Python/Triton 发射和 GPU 完成；CUDA event 区间也包含发射间隙，不能称为纯 kernel duration。每配置 5 次测量、每次 20 次调用，配置顺序固定，未做独立交错重复；只用于筛选下一项实验。只缩小 M、保持 N=128 的候选在 QKV 上反而变慢（106.38 µs），因此不能统一假定“小 tile 总是更快”。

微基准结果（本地产物：`gpu_runs/p1-matmul-tiles-20260912.json`）保存所有候选、精确比较和源码 SHA256；[脚本](gpu_tools/benchmark_matmul_tiles.py)可重跑。**微基准完成时建议：把 router 的 16×64 tile 做成默认关闭的候选，使用真实权重/激活与变长 batch 过 exact gate，再测完整推理。模型 gate 的后续结果见下节。** 本轮没有部署该候选，也不把上表写成整模型提速。


## 后续模型级 gate（2026-09-12）

router 16×64/K32/4-warps 候选现已完成独立 off/shadow/candidate Engine 的 4-case exact gate：tokens、routes、response logprob bits 与跨 batch8/3/1 同行结果全部一致；shadow 覆盖 8 ranks × 48 层，比较 3,538,944 个 FP32 输出元素，零 bit 差异。**真实候选输入仅覆盖 M=1/2，CUDA graphs 关闭；未验证 M=3..16、跨框架或完整推理性能，默认仍 off。** 详见[模型级 gate 报告](P1_ROUTER_TILE_EXACT_20260912.md)及三模式比较（本地产物：`gpu_runs/router-exact-candidate-20260912-01/comparison.json`）。
