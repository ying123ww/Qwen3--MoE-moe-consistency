# 推理补测与 slime PR #2262 对比（2026-09-10）

推理补测已完成。三组独立配置对照下，SGLang 生成耗时增加 **23.02%–23.86%，配对增幅平均 23.48%**，生成吞吐约 **79 → 64 tokens/s**。此前 **2%–3% 仅指 Megatron training-phase step time 的边际增量**，不能作为整个训推对齐方案的开销。

## 实测结果

| 配对 | baseline 中位数（秒） | aligned 中位数（秒） | 耗时增幅 | 输出 tokens/s |
|---|---:|---:|---:|---:|
| 1 | 12.9616 | 16.0159 | +23.56% | 79.00 → 63.94 |
| 2 | 12.9220 | 16.0050 | +23.86% | 79.24 → 63.98 |
| 3 | 13.0203 | 16.0178 | +23.02% | 78.65 → 63.93 |

每种配置独立启动 3 次，每次排除 3 批预热、统计 20 批。每种配置共 60 批正式数据，统计上的独立运行数为 3。配对增幅按 `(aligned 中位数 / baseline 中位数 - 1) × 100%` 计算，再对三组求算术平均。吞吐下降约 19%，与耗时增加约 23.5% 的分母不同，不能混用。

六次进程均正常退出。138 批（包含预热）全部通过 8×128 输出长度和有限 output logprob 校验；每批 cached_tokens 均为零。CPU 汇总器核验逐批记录完整、输入 SHA256 一致、参数匹配及中位数计算。该校验不是新的训推 bitwise exact correctness gate；已有精确一致性来自此前独立正确性实验。

## 为什么不是此前的 2%–3%

| 数字 | 测量对象 | 主要限制 |
|---|---|---|
| 训练平均 +2.34% | Megatron train_one_step：前后向、优化器更新、梯度清零、指标归约 | 固定历史 rollout；baseline 保留 deterministic mode、CPU offload、StatelessAdam；不含 rollout 和权重同步 |
| 推理平均 +23.48% | 阻塞 SGLang Engine.generate：调度、prefill、decode、sampling、output logprob 和结果返回 | 独立推理进程；不含 reward、training、weight sync、HTTP/router 网络层 |
| PR #2262 的约 +25% | 作者称为 e2e step time | GLM-5/FP8/DSA 等不同模型和执行路径，不能与上述任一数字直接等同 |

[PR #2262 原文](https://github.com/THUDM/slime/pull/2262)中，作者明确回复 25% 按 e2e step time 计算。PR 涉及 GLM-5、block-FP8、FP8 KV、DSA 和 DeepEP normal/low-latency 桥接；本项目为 Qwen3-30B-A3B、BF16、8×A100、EP=4。可访问的文字不足以独立核对该生产运行的全部计时边界、batch 和各阶段占比。

因此不能说“我们把他们的 25% 优化到了 2%”。我们此前只测了训练包装的边际成本，且 baseline 已保留部分确定性约束；本次补测显示推理侧也有明显成本。共享的 CPU 搬运等成本可能稀释训练百分比，但尚无分段 profiling 证明其贡献。

**后续已补齐一组短程在线 RL 测量，中位数比增幅约 13.03%；稳态开销仍未确认。** 详见 [在线实测报告](ONLINE_RL_TIMING_20260910.md)。 两侧百分比不能直接相加；两个独立 benchmark 的时间也不能相加冒充真实在线 RL iteration。推理 23.48% 与 PR 的 25% 数值接近，不代表同一指标或复现了该 PR。

## 补测协议与实际开关

- 历史冻结文件中 8 条 prompt，共 1218 prompt tokens；每条强制生成 128 tokens，共 1024 output tokens，temperature=0.6、ignore_eos=True，返回 output logprob。
- 同一 HF checkpoint、软件栈、8×A100、TP=8、DP=2、attention CP=2、EP=4、MoE DP=2、Triton attention/MoE、DeepEP normal、BF16 KV、memory saver。
- aligned 开启 deterministic inference、unified attention decode 和历史固定 DeepEP 配置；configured baseline 关闭前两项，DeepEP 使用默认配置。它不是 pristine upstream 或经过生产调优的最快配置。
- 解析后配置核验：baseline 使用 FlashInfer sampling、允许 custom all-reduce；aligned 使用 PyTorch sampling、禁用 custom all-reduce，并强制禁用 FlashInfer all-reduce fusion。deterministic 模式还在初始化中设置 NCCL allreduce tree。
- 当前 Qwen3 源码在 deterministic inference 下用 FP32 torch.mm Router、unordered top-k 与 FP32 selected-logit softmax。历史 CLI 的 `--sglang-enable-fp32-moe-router` 已不属于当前 ServerArgs，不将其列为有效独立开关。
- 两组均因 DeepEP normal 而禁用 CUDA graph。相同 backend/topology 是控制变量，但也限制了对生产最优配置的外推。
- 每批计时前 flush cache；计时不含加载、预热、cache flush。批内重复 prompt 的复用按引擎行为保留，实际记录均无 cache hit。
- 固定输入及输出长度，不强制两配置生成相同内容；生成内容和 MoE 路由可不同，结果属于该生成负载下的配置开销，不是纯 kernel 微基准。
- 配对顺序 baseline→aligned、aligned→baseline、baseline→aligned；仅 3 个独立运行，报告描述性结果，不宣称统计显著或跨负载普适性。

冻结输入：`/public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt`，SHA256：`ed72c22c9d16e13d455f83af09bda05420ad5e81d678c39a800bd721a550204d`。

## 运行修复与资源

首次启动因旧 Router 参数不再受支持，在创建 Engine 时失败，未加载 GPU。移除无效参数后，baseline 的 custom-all-reduce JIT 在 NFS 上的 TVM-FFI 缓存锁等待；`/proc/PID/stack` 显示 nfs_flock/lockd/RPC，打开的 fd 指向 `.cache/tvm-ffi/.../lock`。只终止该次补测的进程树，GPU 正常释放，无需重启。

随后六次成功运行均设置 `TVM_FFI_CACHE_DIR=/dev/shm/alignment-a-tvm-ffi-999484`，避开 NFS 锁。失败目录保留、不纳入统计。本次未修改对齐数值实现。

全程沿用 900 GiB 作业级内存守护，未触发；结束检查 GPU 无计算进程。Slurm 999484 分配与内存守护仍保留。采样时内存约 45–49 GiB，不将其宣称为连续监控得到的精确峰值。

## 产物

- `gpu_tools/inference_benchmark.py`：逐批生成与验证。
- `gpu_tools/run_inference_pairs.py`：资源检查、顺序运行与失败记录。
- `gpu_tools/summarize_inference.py`：完整性、配置和汇总核验。
- `gpu_runs/inference-comparison-20260910.json`、`.csv`：三组汇总与解析后的配置差异。
- `gpu_runs/a-inference-{baseline,aligned}-20260910-{01,02,03}/`：参数、环境、日志、逐批记录和退出状态。
- `gpu_runs/inference-source-hashes-20260910.json`：评测脚本与关键推理源码哈希。

## 项目表述与下一步

可写：“建立训推一致性正确性门禁，并分阶段评估性能代价：在 8×A100、EP=4 的受控负载下，训练阶段对齐增量约 2%–3%；独立推理配置对照中，固定长度生成耗时增加约 23%–24%。”两侧 baseline 不同，需保留“分阶段、受控”限定，不将它们包装成总开销。

如需与 PR 的 e2e 口径进一步接近，应在真实在线循环中从 rollout 发起计时，到本轮训练、全量权重同步及下一轮推理就绪完成为止；分别记录 rollout/reward、training、offload/onload、weight sync，并建立包含/移除确定性约束的明确 baseline。不能直接套用本次两侧独立时间。大 batch、长输出、标准有状态 Adam、收敛实验仍需另外验证。

后续进展：真实在线计时已接入，并完成每种配置六轮执行（首轮预热、五轮正式）。不需要再重复接入计时；接下来需要延长预热、在线测量及独立重复，检验稳定开销。
