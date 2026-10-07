# Qwen3 Megatron–SGLang 对齐交接报告（停于 v66）

日期：2026-08-22

## 当前结论

Qwen3-30B-A3B 在 EP4/8GPU 配置下已经实现 layer 0 的 attention、router、DeepEP、expert、MoE 和整层输出精确对齐。

正式 gate：

```text
layer=0 tokens=1428 max_abs=0 mean_abs=0
```

全 48 层 gate 显示首个真实分叉位于 layer 1：

```text
layer 0: max_abs=0
layer 1: max_abs=0.0234375
         mean_abs=0.000109332307608
```

因此接下来不应修改 router、DeepEP 或 MoE combine，而应定位 layer 1 attention/residual 输入链路。

## Layer 0 如何实现精确对齐

### Attention 与 RMSNorm

- 统一 Megatron 与 SGLang 的 attention decode/layout。
- 修正 attention output projection 路径。
- 对齐 residual 的 BF16 可见输出。
- RMSNorm mean 使用确定性的 FP32 计算。
- 处理 SGLang 分离的 `delta + residual` 表示，避免在层内过早 BF16 舍入。

关键阶段：

- v46：output projection 精确对齐。
- v47：post-attention residual 精确，pre-MLP norm 尚有差异。
- v50：layer 0 attention 精确对齐。

### Router

- 将 Qwen 的普通 FP32 `torch.mm` 替换为 SGLang/GLM 已验证的 batch-invariant router GEMM。
- 正式 comparator 支持 attention-TP=2 的本地 token 映射。
- 精确校验 router weight。
- top-k weights 先按 expert ID 对齐，再比较权重。
- 分别统计 expert 顺序变化和 expert 集合变化。

v53 时 layer 0 attention/router 已全部精确。

### Expert route probability

v56–v58 证明：

- route ID 和 route probability 一致；
- 未加权 expert core 输出精确一致；
- 差异来自 route probability 的应用位置。

SGLang 在 expert FC2 内融合 route probability，而 Megatron 原先在后面乘。Qwen wrapper 因此改为直接把 `permuted_probs` 传入 SGLang Triton MoE。

中间版本：

- v59：通用 DeepEP bridge 覆盖了 Qwen flag，导致概率重复相乘，结果无效。
- v60：修正初始化顺序，weighted route output 精确，但最终 MoE 仍差 `0.00390625`。

### DeepEP combine 归约拓扑

剩余误差来自 BF16 归约树不同：

- SGLang 先在接收 expert 的 rank 上聚合同一 DeepEP token 的 weighted routes，再用原始 DeepEP handle 跨 rank combine。
- 旧 bridge 让每条 route 分别返回 source rank，再聚合 top-8。

最终改为：

- Qwen fused-probability 路径先做接收端本地 route reduction；
- 使用原始 DeepEP dispatch handle combine；
- GLM/deferred-probability 路径保持原逻辑；
- backward 使用独立 gradient buffer，避免复用 no-grad expert output storage 导致的 view/in-place 错误。

v63 完整 8GPU forward/backward 成功：

```text
layer=0 tokens=1428
mlp_input_max_abs=0
mlp_output_max_abs=0

layer=0 tokens=1428
max_abs=0
mean_abs=0
```

结果文件：

- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer0-20260821-v63/moe-comparison.json`
- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer0-20260821-v63/layerwise-comparison.json`

## Layer 1 当前状态

v64 对 0–47 层做正式 layerwise gate：

- layer 0 精确；
- layer 1 首次分叉；
- 差异随后逐层累积；
- layer 47 约为 `max_abs=19`、`mean_abs=0.094234`。

结果：

`/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-all48-20260822-v64/layerwise-comparison.json`

v65 layer 1 诊断结果：

```text
mlp_input_max_abs=0.03125
mlp_output_max_abs=0.017578125

post_attention_residual_max_abs=0.0078125
post_attention_norm_max_abs=0.03125
router_input_max_abs=0.03125
router_logits_max_abs=0.0270414352417
router_weight_exact=True
aligned_topk_weights_max_abs=0.00241009891033
ID order mismatched tokens=42
ID set mismatched tokens=25
```

结果：

- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer1diag-20260822-v65/moe-comparison.json`
- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer1diag-20260822-v65/router-comparison.json`

注意：router JSON 的 `layer_input_max_abs=1.2939453125` 不能用来判定真实分叉。SGLang layer prehook 抓到的是分离的 `delta/residual` 分量之一，不是 effective combined layer input。

可靠结论是：layer 1 在 post-attention residual 已出现 `0.0078125` 差异，经过 pre-MLP RMSNorm 放大到 `0.03125`，因此分叉在 router/MLP 之前。

当前最可能但尚未证实的原因是跨层 residual carry 语义：SGLang 把上一层 `delta` 和 `residual` 分开传入下一层，Megatron 可能先合并并舍入为 BF16。这样即使 layer 0 可见输出精确，layer 1 的 effective RMSNorm/QKV 输入仍可能不同。

## 主要修改文件

实现：

- `slime/backends/megatron_utils/alignment/qwen3.py`
- `slime/backends/megatron_utils/alignment/deepgemm_forward.py`
- `slime/backends/megatron_utils/alignment/deepgemm_moe_forward.py`
- `slime/backends/megatron_utils/alignment/layerwise_alignment.py`

正式比较工具：

- `slime/utils/compare_qwen3_attention.py`
- `slime/utils/compare_qwen3_router.py`
- `slime/utils/compare_qwen3_moe.py`
- `slime/utils/compare_qwen3_moe_routes.py`
- `slime/utils/compare_glm52_layerwise.py`

对应测试覆盖 attention-TP=2 映射、expert-ID 权重对齐、EP4 synthetic MoE、DeepEP metadata、初始化顺序，以及最后一层 full-token 输出映射。

## 精确终止点

最后的实验是 v66：

```text
job id: qwen3-v66-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer1diag-params-20260822-v66
```

v66 相对 v65 只增加了 module parameter capture：

```bash
SLIME_LAYERWISE_ALIGNMENT_CAPTURE_MODULE_PARAMS=1
SGLANG_TENSOR_DUMP_MODULE_PARAMS=1
```

其目的是让正式 attention comparator 获得 layer 1 的 QKV/O-proj 参数与中间 stage。

v66 在刚提交后被停止。Ray 状态明确报告：

```text
Job 'qwen3-v66-ai-hgx-03' was stopped
```

8 张 GPU 当前无计算进程；Ray 集群仍在，但只有 idle actor。v66 目录中只有 `launcher.log`，没有有效 tensor dump 或 JSON，不能作为实验结果。

当前 launcher 已配置为 v66：

- `/public/home/yingxy2024/.tmp-codex/run_qwen3_v24_ai_hgx_03.sh`
- `/public/home/yingxy2024/.tmp-codex/qwen3_v24_runtime_env.yaml`
- `/public/home/yingxy2024/.tmp-codex/submit_qwen3_v24.py`

## 恢复方式

重新运行 v66：

```bash
ssh ai_hgx_03
bash /public/home/yingxy2024/.tmp-codex/run_qwen3_v24_ai_hgx_03.sh
```

完成后运行正式 layer 1 attention gate：

```bash
ROOT=/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer1diag-params-20260822-v66

/public/home/yingxy2024/.conda/envs/slime/bin/python \
  -m slime.utils.compare_qwen3_attention \
  --megatron-dir "$ROOT/megatron" \
  --sglang-dir "$ROOT/sglang" \
  --layers 1 \
  --sglang-attn-tp-size 2 \
  --num-query-heads 32 \
  --num-kv-heads 4 \
  --head-dim 128 \
  --max-stage-diff 0 \
  --output-json "$ROOT/attention-comparison.json"
```

随后严格停在第一个非零 stage：effective RMSNorm/QKV input、QKV projection、Q/K norm、RoPE、attention core、O-proj、post-attention residual、pre-MLP RMSNorm。

若首个分叉在 layer 1 effective input/RMSNorm，应复刻 SGLang 跨层 `delta + residual` carry 语义。不要修改 router 或 MoE。修复后重跑 0–47 正式 gate；全部精确后才能进入 logits/logprobs/KL。

## Git 基线（2026-08-22）

### slime

- `e528fa7 fix: defer heavy SGLang imports in Ray workers`
- `2dc8604 feat: add formal layerwise alignment gates`
- `53ede01 fix: align Qwen3 layer 0 with SGLang`

### SGLang worktree `a100-deepep-bf16`

- `514ca5a40 fix: expose deterministic Qwen3 attention and MoE paths`
- `2ab7bfe91 feat(debug): capture Qwen3 alignment stages`

### 提交前验证

```text
slime:  45 passed, 2 skipped
SGLang: 8 passed
```

提交前测试还发现并修正了 CPU RMSNorm 单测回归：CUDA/XPU 继续使用 SGLang batch-invariant mean，CPU fallback 使用 `torch.mean`。该修正包含在 `53ede01` 中，不改变生产 GPU 路径。

## 后续 Git 纪律

- 每个新的诊断版本先记录其假设和正式 gate，再启动作业。
- 每个得到可复现 JSON 或修复一个明确分叉的阶段单独 commit。
- 每个重要 commit 都在本文档追加改动、验证结果、输出路径和下一步。

## v67：Layer 1 attention 正式 gate

原 v66 submission ID 已被 Ray 永久登记为 stopped，不能再次提交。v67 仅更换 job ID 和输出目录，诊断配置不变：

```text
job id: qwen3-v67-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer1diag-params-20260822-v67
status: succeeded
```

正式 attention comparator 首次运行发现 gate 工具仍假定 `input_layernorm` 行数为全局 token 数；真实 attention-TP=2 dump 的 `layer_input/input_norm` 可以是本地 token 行。修复后同时支持 full-token 与 local-token 两种布局，并新增参数化回归测试：

```text
commit: fe6389b fix: map local attention-TP norm rows
tests:  5 passed
```

正式 JSON：

`/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer1diag-params-20260822-v67/attention-comparison.json`

关键结果：

```text
all attention/RMSNorm weights exact
input_norm_max_abs=0.001953125
qkv_max_abs=0.001953125
q_norm_max_abs=0.125
k_norm_max_abs=0.25
core_attention_max_abs=0.001220703125
output_projection_max_abs=0.00390625
post_attention_residual_max_abs=0.0078125
pre_mlp_norm_max_abs=0.03125
```

`layer_input_max_abs=1.2939453125` 仍不能作为真实首分叉：该 prehook 捕获的是 SGLang 分离 residual 表示中的一个分量，不是 effective combined input。当前最早可靠分叉是 layer 1 `input_norm`，权重完全相同，因此下一步应验证并复刻跨层 `delta + residual` 的 effective input/RMSNorm 语义，而不是修改 QKV 权重、router 或 MoE。

## v69：跨层 residual carry 修复与 layer 1 精确对齐

代码审计确认：SGLang 在层末保留 BF16 MLP delta 与 BF16 residual 两个分量，到下一层 fused RMSNorm 才在 FP32 中相加；Megatron 原路径先把两者相加并舍入成单个 BF16 layer output，下一层已无法恢复未舍入的和。

修复内容：

- layer 0 的 `mlp_bda` 保存 `mlp_delta.float() + residual.float()`；对外层输出仍为相同 BF16 dtype；
- layer 1 的 fused input RMSNorm/QKV 在 visible forward 中消费该 FP32 sum；
- analytic backward 仍以 Megatron 原 BF16 layer input 为入口，不改变既有 autograd 拓扑；
- 只在两个连续 selected layers 之间启用 carry，缺失 residual source 时立即报错；
- 新增 FP32 carry 与 layer-output residual 两个单元测试。

```text
commit: 2149d1c fix: preserve Qwen3 cross-layer residual sum
tests:  16 passed
```

v68 是一次外层 launcher 误提交为 Ray entrypoint 的操作错误；它因嵌套提交同名 job 被 Ray 立即拒绝，没有启动模型实验。为避免复用版本号，正式实验升为 v69：

```text
job id: qwen3-v69-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer1diag-params-20260822-v69
status: succeeded
```

正式 JSON：

- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer1diag-params-20260822-v69/layerwise-comparison.json`
- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer1diag-params-20260822-v69/attention-comparison.json`

正式结果：

```text
layer 1 output: tokens=1428, max_abs=0, mean_abs=0
input_norm through pre_mlp_norm: every reliable stage max_abs=0
all compared attention/RMSNorm weights: exact
```

attention JSON 中 `layer_input` 与派生的 `pre_mlp_norm_input_fp32` 仍为 `max_abs=1.2939453125`。这两个字段比较的是 SGLang prehook 的分离 residual 分量与 Megatron combined tensor，不是同一语义对象；所有从 effective `input_norm` 开始的可靠 stage 以及最终 layer output 都为 0，因此它们不能被解释为真实分叉。

下一步：把 selected/dump layer 扩展到 2，正式验证 layer 2；若同一 carry 机制成立，再逐步扩大到 0–47，并在每次扩展后保存 JSON gate。不要提前进入 logits/logprobs/KL。

## v70：Layer 2 连续 carry gate

v70 不修改代码，只把连续 selected layers 从 `0 1` 扩为 `0 1 2`，并把 SGLang 诊断 dump 移到 layer 2：

```text
job id: qwen3-v70-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer2diag-params-20260822-v70
status: succeeded
```

正式 JSON：

- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer2diag-params-20260822-v70/layerwise-comparison.json`
- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-layer2diag-params-20260822-v70/attention-comparison.json`

结果：

```text
layer 2 output: tokens=1428, max_abs=0, mean_abs=0
input_norm through pre_mlp_norm: every reliable stage max_abs=0
all compared attention/RMSNorm weights: exact
```

`layer_input_max_abs=2.66796875` 与派生字段仍是不可比的分离 residual prehook 表示；它没有进入 effective RMSNorm，不能作为分叉。

结论：同一 FP32 carry 已连续让 layer 1 和 layer 2 精确对齐。下一步运行 0–47 仅 layer-output 的正式 gate；该版本应关闭逐模块、router、MoE stage dump，减少存储与诊断噪音。

## v71：全 48 层 layer-output 精确 gate

v71 不修改代码，将 Megatron alignment selected layers 与 SGLang layer-output dump 同时扩展为 `0–47`。为控制存储量与诊断噪音，本次关闭逐模块输入/输出、参数、router、MoE stage 与 fused layernorm dump，只保留每层最终输出。

```text
job id: qwen3-v71-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-all48-output-20260822-v71
status: succeeded
```

正式 JSON：

`/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-all48-output-20260822-v71/layerwise-comparison.json`

正式 comparator 使用 attention-TP=2 映射，并把 `max_hidden_diff` 设为 0。结果：

```text
layers: 0–47
tokens per layer: 1428
max_abs for every layer: 0
mean_abs for every layer: 0
```

结论：Qwen3-30B-A3B 在当前 EP4/8GPU、固定样本与确定性路径下，Megatron 和 SGLang 的全部 48 个 transformer layer output 已逐元素完全一致。此前“layer 1 当前不对齐”的状态已被跨层 FP32 residual carry 修复取代；当前不存在已知的 layer 内或跨 layer 分叉。

下一步进入模型末端 gate：先比较 final RMSNorm 与 logits，再比较 rollout/ref logprobs，最后才检查 KL。继续遵守逐 gate 规则；任何上游 gate 非零时，不应跳到下游指标解释。

## v72：模型末端 native final RMSNorm 基线

为避免重复生成全层大 dump，先增加两项诊断能力：

- `SLIME_LAYERWISE_ALIGNMENT_DUMP_LAYERS` 将 Megatron dump layers 与 Qwen alignment layers 解耦；
- `SLIME_LAYERWISE_ALIGNMENT_GLOBAL_MODULE_SUFFIXES` 捕获 selected layer 外的模型尾部模块。

```text
commit: 4ebb1e9 feat: add compact model-tail alignment dumps
tests:  7 passed
```

同时新增 attention-TP=2 感知的正式 final RMSNorm comparator：

```text
commit: f5f46d5 feat: add formal Qwen3 model-tail gate
tests:  8 passed
```

v72 保持 Qwen alignment 覆盖 `0–47`，但双方只保存 layer 47，Megatron 额外保存 `decoder.final_layernorm`；本版没有修改 final RMSNorm 数值路径：

```text
job id: qwen3-v72-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-model-tail-native-20260822-v72
status: succeeded
layer 47: tokens=1428, max_abs=0, mean_abs=0
final RMSNorm: tokens=1428, max_abs=0.03125, mean_abs=3.4769376119e-08
final RMSNorm mismatched elements: 10
```

正式 JSON：

- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-model-tail-native-20260822-v72/layerwise-comparison.json`
- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-model-tail-native-20260822-v72/model-tail-comparison.json`

结论：首个已确认的模型末端分叉在 final RMSNorm；layer 47 output 仍完全一致。下一步复用 GLM 已验证的 final RMSNorm visible forward，并确保最后一层也保存未舍入 FP32 residual sum；修复后先把 final RMSNorm gate 收敛到 0，再进入 logits。

## v73：final RMSNorm 配置 gate 未启用

代码修复复用 GLM final RMSNorm forward，并让最后一层保存 Qwen FP32 residual sum：

```text
commit: 7d41db3 fix: align Qwen3 final RMSNorm with SGLang
tests:  Qwen final-RMS residual test passed; Qwen alignment 11 passed
```

v73 作业成功，但结果与 v72 相同：

```text
job id: qwen3-v73-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-model-tail-aligned-20260822-v73
layer 47: max_abs=0
final RMSNorm: max_abs=0.03125, mismatched_elements=10
```

日志与 runtime YAML 审计确认 `MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS` 未设置，因此最后一层 carry 与 final RMS wrapper 均未启用。v73 不能用于判断代码修复效果；v74 必须只增加该环境变量后重跑同一 gate。

## v74–v75：Qwen final RMSNorm 精确对齐

v74 设置 `MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS=1`，同时启用 SGLang final RMSNorm kernel 和最后一层 FP32 residual carry。作业成功，但正式 gate 明显变差：

```text
job id: qwen3-v74-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-model-tail-enabled-20260822-v74
layer 47: max_abs=0
final RMSNorm: max_abs=0.5, mean_abs=0.002278106758
final RMSNorm mismatched elements: 637419
```

该反例证明 Qwen 与 GLM 的最后一层输出语义不同：Qwen SGLang 在 final RMSNorm 前已经把最后一层输出合并并舍入为 BF16；GLM 所需的最后一层 FP32 residual carry 不能直接套用到 Qwen。

修正因此只保留 SGLang final RMSNorm kernel，并让它消费已经对齐的 BF16 layer 47 output；撤销 Qwen residual 属性扩展和最后一层 carry：

```text
commit: 3b919d8 fix: keep Qwen3 final norm input in BF16
tests:  Qwen alignment 11 passed
```

v75 正式结果：

```text
job id: qwen3-v75-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-model-tail-kernel-20260822-v75
status: succeeded
layer 47: tokens=1428, max_abs=0, mean_abs=0
final RMSNorm: tokens=1428, max_abs=0, mean_abs=0
final RMSNorm mismatched elements: 0
```

正式 JSON：

- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-model-tail-kernel-20260822-v75/layerwise-comparison.json`
- `/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-model-tail-kernel-20260822-v75/model-tail-comparison.json`

结论：当前 Qwen3 EP4/8GPU 路径已从 layer 0 一直到 final RMSNorm 全部逐元素精确对齐。下一 gate 是 logits；必须比较同一 request/position 的词表输出或等价的目标 token logprob，并处理 SGLang DP lm-head 与 Megatron vocab layout 后再生成正式 JSON。
## v76–v77：正式 target-token logprob gate 与 NaN 根因

为避免每个 rank 保存完整词表 logits（本批次会产生数百 MiB 至 GiB 的重复数据），logits gate 采用等价且更小的目标 token logprob：SGLang rollout dump 保存生成时的 `rollout_log_probs`，Megatron train debug dump 保存 policy loss 实际计算的 `log_probs`。正式 comparator 会按 `rollout_position` 重新配对样本，并在比较前严格校验完整 token 序列。

```text
commit: 7cc9806 feat: add formal Qwen3 logprob gate
tests:  3 passed
```

v76 复用 v75 的 rollout，只启动 Megatron debug-train-only。它在 forward 前失败，因为该模式跳过 SGLang 参数解析，Qwen alignment hook 无法读取固定 DeepEP 配置。修复允许通过显式环境变量 `SLIME_QWEN3_FIXED_DEEPEP_CONFIG` 保留同一个验证 gate：

```text
commit: dfb733e fix: preserve Qwen3 DeepEP gate in train replay
tests:  Qwen alignment 12 passed
```

v77 成功完成同一批数据的训练重放，但正式结果不是有限的数值误差：

```text
job id: qwen3-v77-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-logprob-replay-20260822-v77
status: succeeded
samples: 8
target tokens: 256
SGLang nonfinite: 0
Megatron nonfinite: 256
```

train dump：

`/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-logprob-replay-20260822-v77/train/1.pt`

正式 comparator 最初把 NaN 差值经 Python `max(0, nan)` 误显示为 `max_abs=0`；现已改为统计双方非有限值、把非有限差值视为无穷大，并要求 `exact=true` 才通过：

```text
commit: 770dffb fix: reject nonfinite Qwen3 logprob gates
v77 corrected gate: max_abs=inf, mean_abs=inf, mismatched_tokens=256, exit=1
JSON: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-logprob-replay-20260822-v77/logprob-comparison.json
```

根因已确认：运行参数为 `--rollout-temperature 0`。SGLang 会把 greedy temperature 0 规范化为 temperature 1，并用 `top_k=1` 控制采样；其返回 logprob 基于未除温度的原始 logits。Megatron 训练端此前直接执行 `logits / 0`，因此 256 个 logprob 全为 NaN。这也解释了 v71 起内置 `train_rollout_logprob_abs_diff` 一直为 NaN，不能把它归因于 reward 或 reducer。

修复统一两条训练 logprob 路径的有效温度语义，temperature 0 现在按 1 处理，value 路径保持不缩放：

```text
commit: 90ea6e1 fix: match greedy rollout logprob temperature
tests:  value/greedy temperature + comparator, 5 passed
```

下一步是 v78：继续重放完全相同的 v75 rollout，要求 256 个 Megatron logprob 全部有限，再用正式 JSON 报告实际差值。只有该 gate 通过后才能进入 KL。

## v78：greedy 温度已对齐，剩余为有限的末端数值差

v78 继续重放完全相同的 v75 rollout。温度修复生效，双方 256 个 target-token logprob 均为有限值；但严格 gate 仍未通过：

```text
job id: qwen3-v78-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-logprob-greedy-20260822-v78
status: succeeded
samples: 8
target tokens: 256
max_abs: 6.10947608948e-07
mean_abs: 1.62870428255e-08
mismatched_tokens: 31
comparator exit: 1
```

正式 JSON：

`/public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-logprob-greedy-20260822-v78/logprob-comparison.json`

这不是 layer gate 回退：截至 final RMSNorm 的逐元素精确对齐仍由 v75 证明。当前首个未通过 gate 已收窄到 LM head / log-softmax。代码审计显示 SGLang greedy sampler 直接执行 `torch.nn.functional.log_softmax(logits, dim=-1)`，Megatron PPO 路径则使用 vocab-parallel 手写 `max -> exp -> sum -> log`；两者数学等价但 reduction/kernel 舍入不同。下一步先在 TP=1、无 top-p mask 的 Qwen alignment 路径复刻 SGLang `F.log_softmax`，以 v79 判断剩余误差是否全部来自 log-softmax；若仍不为零，再单独捕获 LM-head target logit/normalizer。

## v79b–v81：log-softmax 与 LM-head batching 已排除

首先在 Qwen alignment 的 TP=1、无 mask 路径中直接复用
`torch.nn.functional.log_softmax`，并保留原有解析梯度和 entropy 计算：

```text
commit: 6617f33 fix: match SGLang Qwen3 log softmax
tests:  PPO logprob/entropy + temperature, 25 passed
```

v79b 的正式结果只比 v78 少 1 个不等 token，最大差异完全不变：

```text
job id: qwen3-v79b-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-logprob-torch-softmax-20260822-v79b
status: succeeded
target tokens: 256
max_abs: 6.10947608948e-07
mean_abs: 1.5006516918e-08
mismatched_tokens: 30
comparator exit: 1
```

因此 log-softmax kernel 不是主要剩余来源。随后完成两个独立 gate：

1. Megatron DCP 的 `output_layer.weight` 与 HF/SGLang 的
   `lm_head.weight` 均为 `[151936, 2048]` BF16，逐元素完全相同：
   `torch.equal=True`、mismatched=0、max_abs=0。
2. 同一 BF16 input/weight 的 GPU 诊断显示：M=64 单次 GEMM 与 M=8/M=1
   分块计算会产生 6730 个不同 logits，`max_abs=1.0`；而 M=8、M=4、M=2、
   M=1 彼此完全相同。这证明 decode 小 M 与 replay 大 M 确实可能选择不同的
   BF16 GEMM 算法，但尚不能证明它造成当前 target logprob 差异。

为验证该假设，增加了仅由显式环境变量启用的 Qwen3 LM-head wrapper：forward
逐行执行小 M BF16 matmul，backward 保持解析全矩阵梯度。首次 v80 暴露出
Megatron output layer 位于嵌套 wrapper 中，日志为 `LM head=[]`，所以 v80 是无效的
no-op 实验，不应作为数值结论。随后补齐嵌套定位和未命中时的显式失败 gate：

```text
commit: 7fdbdf3 fix: match Qwen3 decode LM head batching
commit: fcef0a5 fix: locate nested Qwen3 LM head
tests:  Qwen alignment + PPO/temperature, 38 passed
```

v81 日志确认所有 8 个 rank 均命中
`LM head=['module.module.output_layer']`，但正式严格 comparator 与 v79b 完全相同：

```text
job id: qwen3-v81-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-logprob-rowwise-lmhead-20260822-v81
status: succeeded
target tokens: 256
max_abs: 6.10947608948e-07
mean_abs: 1.5006516918e-08
mismatched_tokens: 30
comparator exit: 1
JSON: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-logprob-rowwise-lmhead-20260822-v81/logprob-comparison.json
```

结论：逐行小 M LM-head 不影响这 30 个 target logprob，LM-head batching 假设已被
正式反证。下一步不得继续盲改 GEMM；应对同一 256 个位置双侧保存 target logit 与
`logsumexp`（或完整 log-softmax normalizer），先判断差异位于 target logit、normalizer，
还是 rollout 返回/重排语义。只有形成可重复 JSON 子 gate 后，才能决定下一处修改。

## v82：复用 SGLang batch-invariant log-softmax，target logprob 精确对齐

v75 SGLang tensor trace 已保存每次 decode 的完整 FP32 `logits_processor`，因此无需
增加新 hook 即可把剩余差异拆开。按内部 request id 和 decode position 串起 logits 与
下一 token 后，对 4 个可直接映射请求的 120 个中间 token 完成了两个离线 gate：

```text
普通 PyTorch F.log_softmax(trace logits) vs v79b Megatron: mismatched=0
普通 PyTorch F.log_softmax(trace logits) vs rollout API:   有同样的微小差异
SGLang batch_invariant_ops.log_softmax(trace logits) vs rollout API: mismatched=0
```

源码中 sampler 虽调用 `F.log_softmax`，但 deterministic/batch-invariant runtime 会把
`aten::_log_softmax` 分派到 SGLang 自定义 Triton kernel。此前 v79b 仅复刻了 Python
表层调用，实际仍执行普通 PyTorch kernel，所以它与 Megatron 自身一致、却不与 rollout
返回值一致。根因不是 target token 重排，也不是 target logit 或 LM-head 权重，而是
SGLang batch-invariant log-softmax 的 reduction/舍入路径。

修复在 Qwen alignment 的 TP=1、无 top-p mask 路径直接复用 SGLang
`batch_invariant_ops.log_softmax`；CPU 单测保留 PyTorch fallback，现有解析 backward
继续复用 forward 得到的 softmax：

```text
commit: 1b4aeac fix: match SGLang batch-invariant log softmax
tests:  PPO logprob/entropy + temperature + Qwen alignment, 40 passed
```

v82 关闭已被反证的逐行 LM-head（日志确认 `LM head=[]`），继续重放同一份 v75
rollout。正式严格 gate 通过：

```text
job id: qwen3-v82-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-logprob-batch-invariant-softmax-20260822-v82
status: succeeded
samples: 8
target tokens: 256
max_abs: 0
mean_abs: 0
mismatched_tokens: 0
comparator exit: 0
JSON: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-logprob-batch-invariant-softmax-20260822-v82/logprob-comparison.json
```

当前精确链条为：v75 证明 layer 0–47 与 final RMSNorm 对齐，v82 证明同一批 256 个
训练 target logprob 对齐。下一步进入 KL/loss gate；应首先用同一批数据验证
`train_rollout_logprob_abs_diff=0`、importance ratio=1、approx KL=0，再扩大到非 debug
短跑，不能跳过 reducer/loss 的正式 JSON 验收。

v82 的同一次训练日志已经报告 reducer/loss 指标：

```text
train/loss: 0.0
train/pg_loss: 0.0
train/ppo_kl: 0.0
train/train_rollout_logprob_abs_diff: 0.0
```

comparator 现同时把逐 token 派生 gate 写入同一个 JSON：

```text
commit: b81a43e feat: report Qwen3 KL alignment gates
importance_ratio max_abs_from_one=0, exact_one=true
ppo_kl max_abs=0, exact_zero=true
low_var_kl max_abs=0, exact_zero=true
tests: 4 passed
```

因此同批次的 KL/loss gate 已通过，无需重复模型重放。下一步是非 debug 的最小短跑，
验证真实 rollout → train 数据流和一次非零学习率更新前后的可重复性；该实验应使用新
输出目录和正式 JSON/训练指标，不覆盖 v75/v82 基线。

## v83/v84：真实 rollout → train 数据流精确对齐

v83 原计划执行一次非 debug 在线短跑，但把 `--num-rollout` 设成了 1。slime 的训练循环
从 rollout id 1 开始，并把 `num_rollout` 作为不含上界，因此该配置执行零次迭代；作业虽
显示 succeeded，却没有 rollout/train dump。v83 是配置 no-op，不作为任何数值结论。

v84 将 `--num-rollout` 修正为 2，实际执行且只执行 rollout 1。它不加载 debug rollout，
也不启用 SGLang 全量 tensor trace；数据经过真实的 SGLang 在线生成、slime 数据传递与
Megatron train forward。为隔离数据流变量，仍固定 temperature=0、学习率=0，使用
8 samples × 32 response tokens，并分别保存最小 rollout/train dump。

```text
job id: qwen3-v84-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-real-rollout-20260822-v84
status: succeeded
rollout dump: .../rollout/1.pt (22 KiB)
train dump:   .../train/1.pt (34 KiB)
samples: 8
target tokens: 256
response length: 32 for every sample
```

在线训练日志的 step 1 指标全部通过：

```text
train/loss: 0.0
train/pg_loss: 0.0
train/ppo_kl: 0.0
train/train_rollout_logprob_abs_diff: 0.0
train/grad_norm: 0.0
```

随后对两份真实数据运行正式严格 comparator（`--max-logprob-diff 0`）：

```text
max_abs: 0
mean_abs: 0
mismatched_tokens: 0
importance_ratio exact_one: true
ppo_kl exact_zero: true
low_var_kl exact_zero: true
comparator exit: 0
JSON: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-real-rollout-20260822-v84/logprob-comparison.json
```

因此已经完成两层独立验收：v75/v82 的固定数据逐 tensor/replay gate，以及 v84 的真实
在线 rollout → train gate。当前在 temperature=0、lr=0 条件下，Qwen3-30B-A3B 的
256 个 target logprob、importance ratio、PPO KL 与 low-variance KL 均逐 token 精确一致。

下一步一次只引入一个变量：先保持 lr=0，把 temperature 改为非零值（建议沿用训练默认
0.6）验证随机采样与 temperature 缩放路径；通过后再恢复非零学习率，运行至少两个
rollout step，验证参数更新、权重同步以及更新后下一轮 rollout/train 对齐。非零
temperature 很可能首先暴露 SGLang BF16 logits temperature 路径与 Megatron FP32 除法
之间的差异；若 gate 失败，应先比较 temperature 缩放后的 target logit/normalizer，
不要同时修改 optimizer 或权重同步。

## v85/v86：非零 temperature 精确对齐

v85 只把 v84 的 temperature 从 0 改为 0.6，学习率继续为 0，其余在线 rollout →
train 配置不变。作业成功并生成 8 samples × 32 tokens，但严格 gate 首次失败：

```text
job id: qwen3-v85-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-temperature-0p6-20260822-v85
status: succeeded
samples: 8
target tokens: 256
max_abs: 2.6226043701171875e-06
mean_abs: 4.137109499424696e-08
mismatched_tokens: 9
importance_ratio exact_one: false
ppo_kl exact_zero: false
low_var_kl exact_zero: false
comparator exit: 1
JSON: .../v85/logprob-comparison.json
```

差异只在非零 temperature 出现。SGLang 的 standard deterministic sampler 对 FP32
logits 执行 `logits.div_(sampling_info.temperatures)`；`temperatures` 是在 CPU 以
`torch.float` 构造后搬到 GPU 的 `[batch, 1]` FP32 tensor。Megatron 此前则执行
`logits / rollout_temperature`，其中 divisor 是 Python float。

单 GPU 算子 gate 对 1,000,003 个 FP32 输入比较两种除法：

```text
Python float 0.6 vs GPU FP32 temperature tensor
mismatched elements: 230680 / 1000003
max_abs: 1.52587890625e-05
GPU 0-D FP32 tensor vs SGLang [1,1] FP32 tensor: mismatched=0, max_abs=0
```

因此根因不是 BF16 logits，也不是 log-softmax、模型层或随机 token 重排，而是 CUDA
除法对 Python scalar 与 device FP32 tensor 选择了不同的 reciprocal/舍入路径。修复让
Qwen alignment 的非零 temperature 使用与 logits 同 device/dtype 的 tensor divisor；
其他模型仍保留原有 scalar 路径，greedy temperature=0 仍映射为 1，不影响 value head。

```text
commit: 296d8a1 fix: match SGLang temperature division
tests: value temperature + PPO logprob/entropy + Qwen alignment, 40 passed
```

v86 固定加载 v85 的同一份 rollout，只重新计算 Megatron train forward。在线指标恢复
为 `train/train_rollout_logprob_abs_diff=0`，正式严格 comparator 也通过：

```text
job id: qwen3-v86-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-temperature-tensor-replay-20260822-v86
status: succeeded
rollout input: v85/rollout/1.pt
train dump: v86/train/1.pt
samples: 8
target tokens: 256
max_abs: 0
mean_abs: 0
mismatched_tokens: 0
importance_ratio exact_one: true
ppo_kl exact_zero: true
low_var_kl exact_zero: true
comparator exit: 0
JSON: .../v86/logprob-comparison.json
```
至此 temperature=0 与 temperature=0.6 两条真实/固定数据 gate 都已逐 token 精确通过。
下一步可恢复非零学习率并运行至少两个 rollout step：第一步完成真实反向与参数更新、
同步到 SGLang；第二步验证更新后模型的新 rollout logprob 与 Megatron train logprob 仍
逐 token 精确。不要同时扩大 batch、response length、top-p 或 `n_samples_per_prompt`。

## v87：非零 advantage、零学习率 gate

在恢复非零学习率前，先排除此前 `n_samples_per_prompt=1` 导致 GRPO 组内 advantage
恒为 0 的假通过风险。v87 保持 v85/v86 已验证的 temperature=0.6 与 lr=0，只把
rollout 改为 4 prompts × 2 samples，并使用现有 `rm-type=random` 形成组内奖励差异：

```text
job id: qwen3-v87-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-group-advantage-20260822-v87
status: succeeded
rollout batch size: 4 prompts
n_samples_per_prompt: 2
global batch size: 8
temperature: 0.6
lr: 0
```

保存的 4 个 group 原始 reward 为 `[0,0]`、`[1,1]`、`[1,0]`、`[1,0]`；后两组经
GRPO 组内去均值后明确产生非零 advantage。训练侧也给出非零梯度范数，而 lr=0 保证
这一关只验证多样本/advantage 数据映射，不更新参数：

```text
train/grad_norm: 20.76210863639088
train/train_rollout_logprob_abs_diff: 0.0
samples: 8
target tokens: 256
max_abs: 0
mean_abs: 0
mismatched_tokens: 0
comparator exit: 0
JSON: .../v87/logprob-comparison.json
```

因此已排除“只有零 advantage 才能对齐”的可能。下一关保持 v87 的 batch、group、
temperature、reward 与所有对齐算子不变，只把 lr 改为非零并运行两个训练 rollout：
step 1 必须同时出现非零 grad norm 和非零参数更新，随后把新权重同步给 SGLang；step 2
必须对更新后模型的 rollout/train dump 再运行 `--max-logprob-diff 0` 严格 comparator。

## v88：真实参数更新后的第二轮精确 gate

v88 保持 v87 的 4 prompts × 2 samples、random reward、temperature=0.6 与全部对齐算子
不变，只把学习率改为 `1e-6`，并执行两个 rollout/train step：

```text
job id: qwen3-v88-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-updated-model-20260822-v88
status: succeeded
lr: 1e-6
num-rollout: 3（实际执行 step 1 和 step 2）
```

step 1 的 4 个 reward group 中有 3 个组内 reward 不同，训练明确产生非零梯度并使用
非零学习率；更新前的 rollout/train forward 仍完全一致：

```text
train/grad_norm: 17.870827781490792
train/lr-pg_0,1,2: 1e-6
train/train_rollout_logprob_abs_diff: 0.0
post-step-1 full weight sync: succeeded（2 dense groups + 48 expert layers）
```

step 1 后的完整权重同步于 20:19:03 开始、20:19:20 结束；所有 SGLang
`/update_weights_from_tensor` 请求返回 200。rollout 2 在 20:19:26 才开始落盘，所以它
使用的是更新后的 SGLang 权重，而不是初始 checkpoint。step 2 仍有 3 个非零 advantage
group，并再次产生非零梯度：

```text
train/grad_norm: 17.31536599134025
train/train_rollout_logprob_abs_diff: 0.0
samples: 8
target tokens: 256
max_abs: 0
mean_abs: 0
mismatched_tokens: 0
importance_ratio exact_one: true
ppo_kl exact_zero: true
low_var_kl exact_zero: true
comparator exit: 0
JSON: .../v88/logprob-comparison-step2.json
```

step 1 的独立严格 comparator 也同样为 256/256 tokens、`max_abs=0`，JSON 位于
`.../v88/logprob-comparison-step1.json`。至此已经证明 Qwen3 alignment 不仅能对齐初始
模型和零梯度 forward，也能在真实 Adam 更新、完整权重同步之后继续逐 token 精确对齐。
下一步不应再改核心算子；应一次只扩大一个训练变量（优先 response length），并保留
v88 作为最小可重复回归 gate。

## v89：response length 128 的更新后模型 gate

v89 只把 v88 的 `rollout-max-response-len` 从 32 提高到 128；非零学习率、两轮
rollout/train、4 prompts × 2 samples、temperature=0.6、DeepEP/page 配置均保持不变。
该实验覆盖每个样本两个 64-token page 以及每轮 1024 个 target token：

```text
job id: qwen3-v89-ai-hgx-03
root: /public/home/yingxy2024/.tmp-codex/qwen3-30b-ep4-response128-20260822-v89
status: succeeded
response length: 128
target tokens per step: 1024
lr: 1e-6
train steps: 2
```

两轮都包含组内不同的 random reward，并产生非零梯度；step 1 后再次完整同步更新后的
dense/expert 权重，随后才生成 rollout 2：

```text
step 1 grad_norm: 8.293760817979946
step 1 online logprob abs diff: 0.0
step 2 grad_norm: 3.415624754295794
step 2 online logprob abs diff: 0.0
```

对两轮 dump 分别运行 `--max-logprob-diff 0` 的正式 comparator：

```text
step 1: samples=8, tokens=1024, max_abs=0, mismatched_tokens=0
step 2: samples=8, tokens=1024, max_abs=0, mismatched_tokens=0
importance_ratio exact_one: true
ppo_kl exact_zero: true
low_var_kl exact_zero: true
```

JSON 分别位于 `.../v89/logprob-comparison-step1.json` 与
`.../v89/logprob-comparison-step2.json`。因此精确对齐不依赖 32-token 短响应，并已跨过
64-token page 边界；长序列下真实 Adam 更新与更新后权重同步仍保持逐 token 精确。
v89 可作为比 v88 更强、但仍可快速复现的长序列回归 gate。

## Final conclusion 与 regression policy

最终结论：在下述固定 scope 内，Qwen3-30B-A3B 的 SGLang rollout 与 Megatron train
forward 已达到逐 token、bitwise exact 的 target logprob 对齐；真实 Adam 更新并完整同步
权重后，下一轮仍保持 exact。v89 是本任务的最终 regression gate，不再继续扩张诊断。

### v89 regression gate

可运行 gate 与保留基线位于：
`/public/home/yingxy2024/.codex/qwen3-v89-regression/`。执行 `run.sh` 后，必须分别对
rollout/train step 1 和 step 2 运行 `slime/utils/compare_qwen3_logprobs.py`，阈值固定为
`--max-logprob-diff 0`。两轮都必须满足 8 samples、1024 target tokens、`max_abs=0`、
`mismatched_tokens=0`、importance ratio exact_one、PPO KL/low-var KL exact_zero，并有
非零 grad norm。以后修改 Qwen alignment、DeepEP、logprob 或 weight sync 都先跑此 gate；
只有 gate 失败才重新进入 layer/router/combine/logsoftmax 逐级诊断。

### 已证明的 scope

- 模型：Qwen3-30B-A3B，BF16，48 个 MoE layer，128 experts，top-k 8。
- SGLang：DP=2、attention-CP=2、EP=4、normal DeepEP、Triton attention/MoE、固定 DeepEP config。
- Megatron：TP=1、PP=1、CP=1、EP=4，deterministic mode，完整 0–47 layer alignment。
- 数据路径：真实在线 rollout → slime batch → Megatron train forward，而非仅固定 tensor replay。
- 训练条件：temperature=0.6、4 prompts × 2 samples、response length 128、lr=1e-6、两次更新。
- 数值结论：每轮 1024 个 target token bitwise exact；更新后权重同步的下一轮仍 exact。

### 未证明的 scope

- 其他模型、精度、并行拓扑、attention/MoE backend 或非 deterministic inference。
- top-p mask、可变长度提前 EOS、超长上下文及长时间多轮训练的统计稳定性。
- rollout 质量、reward 正确性、收敛质量或训练吞吐；v89 使用 random reward 只保证非零 advantage。
- “Megatron backward 与 SGLang backward 相同”；SGLang 在此仅作为 rollout forward 参考。

### 最终代码与清理状态

正式 implementation 保留 router GEMM、attention、Triton MoE/DeepEP、final RMSNorm、
batch-invariant log-softmax、temperature tensor divisor 与 weight sync 路径。已删除 v80/v81
row-wise LM-head 试验分支、Qwen router trace 生命周期 hooks、自动 layerwise dump 接入及
对应废弃参数/测试；独立 `layerwise_alignment.py` 与正式 comparator 仍作为失败后诊断工具。
相关单测共 47 passed。62 个 `.tmp-codex/qwen3-30b-ep4-*` 临时目录和我们创建的
`/tmp/slime-cuda-runtime-qwen-align-v41` 已删除；共享 Ray session 未改动。

v90 在 SGLang 权重加载阶段按用户要求主动终止，没有 rollout/train dump，不构成任何结论。
