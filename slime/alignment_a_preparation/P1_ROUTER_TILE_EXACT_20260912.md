# Qwen3 router 小 tile 模型级 exact gate（2026-09-12）

**router 小 tile 已通过本轮有限真实模型 exact gate。** 独立 off、shadow、candidate 三个 Engine 的 4 个 case 中，tokens、routes、response logprobs 及其 bits 全部一致；各模式的原第 1/前 3 条输入在 batch 8/3/1 中也一致。shadow 覆盖 8 ranks × 48 层，逐 bit 比较 3,538,944 个 FP32 输出元素，零差异、零非有限值。**实际候选输入只覆盖 M=1/2，CUDA graphs 关闭；这不是性能或跨框架验证，默认仍保持 `off`。**

## 本轮 GPU 执行结果

| Run | 结果 | 证据与解释 |
|---|---|---|
| `router-exact-off-20260912-01` | 初始化失败，exit 1 | 外部 GPU0 进程占用约 64 GiB，world 最小可用显存 13.736 GiB，其余卡约 76.6 GiB；原显存平衡检查正确拒绝启动。占用进程的父 PID 与本次 worker 不同，随后自行退出；本轮没有终止该外部进程。 |
| `router-exact-off-20260912-02` | **完整通过，exit 0** | 4 个 case 全部保存，`complete.json` 为 `off / 4 cases`。原第 1/前 3 条跨 batch 的完整 tokens、routes、response logprobs 及 bits 全部一致。 |
| `router-exact-shadow-20260912-01` | 初始化失败，exit 1 | 主任务进程检查发现 GPU6/7 有另一个 Qwen3-4B PD 服务；初始化时最小可用显存 68.562 GiB，其他卡约 76.6 GiB，低于 90% 平衡阈值。**未执行到 router shadow，不能记作数值 gate 失败。** |
| `router-exact-shadow-20260912-02` | **完整通过，exit 0** | 4 个 case 完成；真实逐层 shadow 比较零差异，模型产物与 off 一致。 |
| `router-exact-candidate-20260912-01` | **完整通过，exit 0** | 4 个 case 完成；官方三模式比较 `model_exact=true`、`batch_exact=true`，独立原始产物复核亦通过。 |

早期失败记录与基线证据：[off01 日志](gpu_runs/router-exact-off-20260912-01/job.log)、[off02 manifest](gpu_runs/router-exact-off-20260912-02/capture/manifest.json)、[off02 完成标记](gpu_runs/router-exact-off-20260912-02/capture/complete.json)、[基线跨 batch 比较](gpu_runs/router-exact-off-20260912-02/capture/baseline-batch-comparison.json)、[shadow01 日志](gpu_runs/router-exact-shadow-20260912-01/job.log)。显存日志直接证明失败阶段；外部服务身份由主任务当时的进程检查确认。

### 完成的候选 gate 与独立审计

用户确认 GPU 空闲后，在连续可用窗口重跑 shadow02 并运行 candidate01。两轮均完成 4 个 case 并正常退出，冻结源码哈希与 off02 保持一致。官方 [comparison.json](gpu_runs/router-exact-candidate-20260912-01/comparison.json) 显示 `model_exact=true`、`batch_exact=true`，两类 failure 列表均为空。

独立审阅原始 JSONL 与 `.npy` 后，补充保存 [shadow 审计汇总](gpu_runs/router-exact-shadow-20260912-02/capture/audit-summary.json) 和 [candidate 审计汇总](gpu_runs/router-exact-candidate-20260912-01/capture/audit-summary.json)：

| 检查 | 实际结果 |
|---|---|
| 各模式实际 Engine 配置 | off/shadow/candidate 的完整 `resolved-server-args.json` 相同 |
| shadow 覆盖 | 8 ranks × 48 layers = 384 个 rank/layer 组合，调用序号均连续 |
| shadow 总调用 / 候选适用调用 | 23,232 / 21,504；另 1,728 次较大 M 使用预先限定的原 mm 路径 |
| 真实候选 shape | `1×2048 @ 2048×128`：15,360 次；`2×2048 @ 2048×128`：6,144 次 |
| shadow bit 检查 | 3,538,944 个 FP32 输出元素，bit mismatches = 0，nonfinite = 0 |
| candidate 覆盖 | 同样 23,232 次调用、21,504 次适用调用，384 个 rank/layer 全覆盖；真实候选 M 同为 1/2 |
| 4 个 case 的模型产物 | off→shadow、off→candidate、shadow→candidate：完整 tokens、routes、response logprob 数值及 uint32 bits 的 dtype、shape、bytes 全部相同 |
| 各模式 batch 不变性 | batch8 的前 3/前 1 样本分别与 batch3/batch1 完全一致 |

candidate audit 记录实际候选执行覆盖，不额外运行逐调用原 mm；候选模型级精确性来自独立 Engine 产物比较，不能把 candidate audit 的零 mismatch 字段当作额外 shadow 比较元素。较大 M 的 prefill 形状在独立运行间有调度差异，但均在候选范围外，最终模型产物仍完全一致。

**M=3..16 尚未由此次真实模型输入覆盖。** 虽然实验 dispatch 的预先限定范围为 M≤16，当前证据只能支持所运行的有限输入；上线前应补齐目标 batch 范围或进一步收窄候选适用范围。此次没有修改默认模式，也没有开启线上候选。

从 off02 各样本原始 `.npy` 文件核算，全部 logprob float32 数组与保存的 uint32 bits 一致：

| Case | 请求数 | 完整 token 数 | response logprob 数 | 有 routes 的 token 位置数 | expert ID 数（48 层 × top8） |
|---|---:|---:|---:|---:|---:|
| batch8 | 8 | 1,346 | 128 | 1,338 | 513,792 |
| batch3 | 3 | 473 | 48 | 470 | 180,480 |
| batch1 | 1 | 160 | 16 | 159 | 61,056 |
| variable3 | 3 | 257 | 48 | 254 | 97,536 |
| 合计 | 15 | 2,236 | 240 | 2,221 | 852,864 |

以上 15 请求包含重复输入，不应写成 15 条独立原始样本。batch8 的 8 条原始 prompt 共 1,218 tokens，生成 128 tokens；最终生成 token 尚未再次前向，因此每请求 routes 比完整 tokens 少 1。基线比较文件 `{"3": [], "1": []}` 仅记录 off 自身两个跨 batch 比较；完整候选比较见上方官方 comparison.json。

首批 8 条已转换为 [B1 冻结 rollout](routepack_b_20260912/router-off02-b1-rollout.pt)，包含供训练 replay 使用的 tokens、routes 和真实推理 logprobs。有效 logprob 温度仍为 1.0。

shadow02、candidate01 与三模式比较已完成。下一步执行复用 off02 rollout 的 **B1 训练端 topology / dispatch / replay gate**（另行报告），之后才能结合跨框架结果决定是否扩大候选验证。性能收益仍需关闭审计、按原重复计时协议独立测量；本轮含逐 bit 同步、路由捕获与日志，不能作为整模型性能对照。保留显存平衡检查；若源码或输入变更，必须重新建立可配对基线。

## 候选与默认行为

`experiments/sglang-main-sm80/python/sglang/srt/models/qwen3_router_tile.py` 提供默认关闭的 router 专用候选。`SLIME_QWEN3_ROUTER_TILE_MODE=off|shadow|candidate`，默认 `off` 仍调用原 `torch.mm`。只在 deterministic Qwen3 `_router_logits` 中调用；没有修改 BF16 attention 投影、router top-k、路由捕获或通用 kernel 注册。

候选严格限定 FP32、CUDA 同设备、`1 <= M <= 16`、`K=2048`、`N=128`。沿用原 persistent Triton kernel 与 FP32 `BLOCK_K=32`、`GROUP_M=8`、3 stages，只将 M/N tile 调成 16×64、warps 调成 4。范围外形状显式记录 excluded 并使用原 mm，这属于预先限定的 dispatch 范围；范围内 kernel 异常直接失败。

`shadow` 对真实逐层激活和真实 gate 权重分别计算候选与原 mm，通过 int32 view 比较所有输出 bits（包括 +0/-0），记录 rank、layer、shape、调用序号、比较元素和不一致数；存在 bit 差异或非有限输出时记录后立即抛错。数据流仍返回原 mm。`candidate` 返回候选结果，记录实际覆盖，随后由独立 Engine 产物做模型级比较。实验模式要求 audit 路径，禁止 CUDA graph capture，防止 Python shadow 检查在 replay 时被跳过。

## 模型 gate 协议与 B1 复用

独立脚本 `alignment_a_preparation/gpu_tools/router_tile_exact.py`：

- 三个独立 Engine 进程：off、shadow、candidate。使用同一 frozen aligned manifest，显式关闭 prefill/decode CUDA graphs。两次 candidate capture 不复用 baseline Engine。
- 固定 workload 依次为：原冻结 8 prompts、原前 3 prompts、原第 1 prompt、前 3 prompts 截为 17/63/129 tokens。每批 flush cache；每请求 greedy 生成 16 tokens，ignore_eos。
- greedy `temperature=0`，但返回 logprob 的有效温度是 **1.0**：当前 `sampler.py` greedy 分支直接 `log_softmax(logits)`，无温度缩放。此项已传达给 RoutePack B1 训练 gate。
- 每批保存完整 tokens、原始 routes（`[tokens-1,48,8]`）、response logprobs float32 和 uint32 bits。首批 `baseline/case-batch8/captured` 可直接构建 B1 frozen rollout。
- 比较器要求三轮请求配置、采样、输入、源码哈希一致，所有 case 的 tokens/routes/logprob 字节一致，并单独检查原第 1/前 3 条在 8/3/1 batch 中逐 bit 一致。shadow 与 candidate audit 均须覆盖 **8 ranks × 48 layers**，shadow 每个 eligible 记录必须包含真实比较元素。
- 仅在完整产物通过后给出 gate 成功；缺失完成标记、缺失 audit coverage、不同源码或比较不一致均失败。

启动 worker（由外部 allocation/memory guard 管理资源）：

```bash
python slime/alignment_a_preparation/gpu_tools/router_tile_exact.py capture \
  --mode off \
  --reference-manifest slime/alignment_a_preparation/gpu_runs/a-matmul-attribution-20260912-01/manifest.json \
  --frozen-rollout /public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt \
  --output RUN/baseline --response-tokens 16
```

另外两轮将 mode/output 分别换为 shadow/RUN/shadow、candidate/RUN/candidate。三轮结束：

```bash
python slime/alignment_a_preparation/gpu_tools/router_tile_exact.py compare \
  --baseline RUN/baseline --shadow RUN/shadow --candidate RUN/candidate \
  --output RUN/comparison.json
```

## 已执行的 CPU 验证

`test_router_tile_exact.py` **14 项通过**（psamp Python 3.12、pytest 8.3.4）。验证默认原 mm、边界形状、非法模式、拒绝 CPU 实验、signed-zero bit 差异写审计并失败、kernel 错误不回退、字节比较、缺失 rank/layer 或无比较元素拒绝以及 workload 同行/变长构造。CPU 测试不检验 GPU kernel 数值正确性。

## 结论边界

这是有限输入、greedy、关闭 CUDA graphs 的模型 gate；此次通过的结果仍不能替代跨框架训练 exact、图模式验证或不带审计的完整推理计时。此次脚本和审计含同步与序列化开销，不可用来宣称模型提速。checkpoint 路径/config 冻结，但未计算所有权重 shard 内容哈希。

## 2026-10-07：关闭审计的计时路径

Router 候选现在收窄到真实模型历史 gate 已覆盖的 M=1/2（K=2048、N=128、FP32）。其他形状使用原 mm，默认模式仍为 off。

新增 SLIME_QWEN3_ROUTER_TILE_AUDIT=0|1，默认 1。shadow 必须保留审计；candidate 可显式设为 0，以跳过逐调用 rank 查询、计数、JSON 和文件写入。候选仍拒绝 CUDA graph capture，仍保持 BLOCK_K=32、3 stages 和原 persistent kernel 的数值实现。候选执行异常直接失败，不静默回退。

router_tile_exact.py capture --mode candidate --disable-audit 验证实际计时路径；比较器用独立 off、audited shadow 和 audit-free candidate 的完整 tokens/routes/logprob bits 验收。性能脚本 inference_benchmark.py --variant aligned --router-tile-mode candidate 自动关闭审计。新启动器 router_tile_performance.py 在模型加载前检查八卡 P2P，运行单卡 microcheck、三模式模型 gate，再按三轮轮换顺序执行 baseline/aligned/candidate 计时。

本轮进展及最终性能结果见 [P1_ROUTER_TILE_PERFORMANCE_20261007.md](P1_ROUTER_TILE_PERFORMANCE_20261007.md)。历史 2026-09-12 的审计文件和结论不受此追加说明改写。
