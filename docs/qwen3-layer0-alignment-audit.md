# Qwen3 第 0 层逐步骤数值对齐与默认开关审计

审计日期：2026-10-04。审计方式：读取当前源码、启动脚本及保存的 v89 配置；本次没有修改算法，没有执行 GPU 回归。

## 1. 结论与适用范围

**普通运行 Qwen3 不会自动启用本项目的完整 Megatron–SGLang 数值对齐修复。** 模型名字、`--qk-layernorm`、`--normalization RMSNorm` 或选择 Triton/DeepEP，都不等于启用了这些修复。

本项目的主要修复对象是 Qwen3-30B-A3B 的训练 forward 与推理 forward 的逐元素一致性。数学表达式相同，不代表有限精度下执行结果相同：GEMM 分块、归约次序、概率乘法位置、BF16 舍入时机，都可能引入差异。

主要判断如下：

- 输入 RMSNorm、QKV 投影、Q/K norm、attention core、输出投影、两处残差边界、MoE 前 RMSNorm、router、专家 FFN、MoE 汇总，均有明确对齐代码。
- 第 0 层输入 hidden states 本身，以及 RoPE，在此次审查的 Qwen3 专门对齐路径中未见独立替换补丁。
- dispatch 有上层元数据、排序和校验适配；不能据此说 DeepEP CUDA 通信 kernel 被重写。
- v89 专用配置显式启用了主要修复；普通示例脚本与当前两层 SGLang 启动脚本没有启用整套对齐。
- 历史成功结果与当前代码是否兼容必须分开看。当前工作区和 v89 实际引用的 Megatron 路径不同，残差代码也不同，详见第 6 节。

这里的“有修复”指存在专门改变数值执行路径或数据组织方式的实现，不意味着原始算法在数学上错误，也不意味着该实现正在某个在线进程中运行。

## 2. 代码范围与层编号

| 角色 | 本次检查位置 |
|---|---|
| Qwen 专用训练端适配 | [qwen3.py](../slime/slime/backends/megatron_utils/alignment/qwen3.py) |
| RMSNorm、router 与自定义 autograd | [deepgemm_forward.py](../slime/slime/backends/megatron_utils/alignment/deepgemm_forward.py) |
| DeepEP bridge、排列与汇总 | [deepgemm_moe_forward.py](../slime/slime/backends/megatron_utils/alignment/deepgemm_moe_forward.py) |
| top-k 捕获 | [routing_replay.py](../slime/slime/utils/routing_replay.py) |
| 实验版 SGLang Qwen 模型 | [qwen3_moe.py](../experiments/sglang-main-sm80/python/sglang/srt/models/qwen3_moe.py) |
| 当前工作区 Megatron 层实现 | [transformer_layer.py](../Megatron-LM/megatron/core/transformer/transformer_layer.py) |

`qwen3.py:685` 附近用 `global_layer = int(layer_number) - 1` 转成零起始层编号，因此这里的第 0 层就是第一个 Transformer 层。模型配置为 48 层，`FIRST_K_DENSE_REPLACE=0`，第 0 层已经是 MoE 层。Qwen3-30B-A3B 配置为 hidden size 2048、32 个 Q heads、4 个 KV heads、head dim 128、128 个 experts、top-k 8。依据：[模型参数脚本](../slime/scripts/models/qwen3-30B-A3B.sh)。两层实验模型只保留前两层，不能把它与完整 48 层验证范围混为一谈。

## 3. 按计算顺序详细解释

### 3.1 输入 hidden states

**作用。** 第 0 层接收 embedding 等上游路径提供的表示，可概念性记为 `h0[token, hidden]`。后续层的输入则来自上一层的残差输出。

**有没有修复。** Qwen 专门对齐入口没有替换第 0 层 embedding，也没有通过覆盖输入 tensor 来强制匹配参考结果。修复从输入 RMSNorm 开始。上游 token 顺序、权重、dtype 和位置必须本来就正确。

**与后续层的区别。** 第 1 层及以后可以额外接收上一层保存的未舍入 FP32 残差和。不能把这种跨层输入修复说成第 0 层 embedding 修复。

**入口。** `qwen3.py:256` 的 `_wrap_sglang_input_norm_qkv`；无 `residual_source` 时使用当前 `value`，有时才读取上一层的 FP32 sum。

### 3.2 输入 RMSNorm

**作用。** 对每个 token 的 hidden 维计算：

```text
rms2 = mean(x²)
y = x × rsqrt(rms2 + eps) × weight
```

它没有 LayerNorm 的减均值步骤。公式里的 mean 是平方值的平均。

**为什么需要对齐。** 相同公式可以由不同 kernel 实现。平方和归约次序、内部精度、乘权重前后是否先转换 BF16，都影响末位结果。仅把参数统一成 `eps=1e-6` 不足以保证逐元素一致。

**修复。** `_wrap_sglang_input_norm_qkv` 替换 Megatron 的 `TELayerNormColumnParallelLinear.forward`，先通过 `_SGLangRMSNormWithAnalyticBackward` 得到对齐的 norm 输出，再调用对齐的 QKV 投影。第 0 层使用 SGLang batch-invariant RMSNorm；后续层如果带 FP32 sum 则走专门的 FP32-sum norm。

**forward/backward。** forward 选用 SGLang 数值路径，backward 保留解析 RMSNorm 求导路径；不是把训练禁用，也不是声称与 SGLang backward 逐位相同。

**默认条件。** 必须调用 Qwen alignment hook、选中该层，且 `MEGATRON_USE_SGLANG_QWEN3_INPUT_NORM_QKV=1`。这个环境变量默认 `0`。

**代码。** `qwen3.py:256–305`；`deepgemm_forward.py:234–275`、`:686–713`。

### 3.3 Attention：Q/K/V 投影

**作用。** 对归一化后的输入做线性映射，产生 Q、K、V，再按 GQA 布局组织 heads。Q heads 数与 KV heads 数不同。

**修复。** `_SGLangBatchInvariantLinear.forward` 将输入展平为二维，要求输入和权重都是 BF16，调用 `matmul_persistent(value_2d, weight.T)` 后恢复形状。它代替该 wrapper 中原来 TE 线性层的 forward。

**原因。** GEMM 的分块和累加策略可能随输入行数变化；推理 batch 与训练 packed batch 的 token 数经常不同。相同权重并不保证两个不同 GEMM 实现给出完全相同结果。

**默认条件。** 与输入 RMSNorm 共用 `MEGATRON_USE_SGLANG_QWEN3_INPUT_NORM_QKV`。开启这个开关会同时替换这两个部分，不是只改 norm。

**代码。** `qwen3.py:95–120`、`:289–300`。backward 使用普通矩阵乘法求输入和权重梯度。

### 3.4 Attention：Q/K RMSNorm 与 RoPE

**Q/K RMSNorm 的作用。** 对 Q 和 K 的对应向量进行归一化；它与进入整层时的输入 RMSNorm 是不同模块。

**修复。** `_wrap_sglang_rms_norm` 把 `q_layernorm` 和 `k_layernorm` 的 forward 替换为 SGLang 对齐 norm，同时检查普通 RMSNorm、无 bias、非 zero-centered gamma 等条件。数值路径不支持时会报错，而不是静默退化。

**默认条件。** `MEGATRON_USE_SGLANG_QWEN3_QK_NORM=1`，默认 `0`；另外仍需 hook 与选层。

**RoPE。** RoPE 利用位置对 Q/K 做旋转。当前 Qwen 专门 wrapper 没有单独替换 RoPE。模型配置中的 rotary base 等需要一致，但“配置一致”“整段 Attention 对齐成功”都不等于“RoPE kernel 被修复”。

**代码。** `qwen3.py:123–150`、`:711–717`；模型脚本的 `--position-embedding-type rope`、`--rotary-base`。

### 3.5 Attention：注意力核心计算

**作用。** 概念上计算 `softmax(QKᵀ / sqrt(head_dim) + causal_mask)V`。实际使用融合 kernel，不必显式生成整个注意力矩阵。

**修复。** `_wrap_sglang_core_attention` 包装 Megatron attention；可见 forward 由 `_run_sglang_unified_attention` 执行 SGLang 的 `extend_attention_fwd_unified`。它检查 BF16、THD 形状、GQA 兼容性和 Q/KV 序列边界；用 packed sequence 的 `cu_seqlens` 构造 indptr 与 KV indices。

**重要细节。** 这个训练端调用使用连续 packed KV、`page_size=1`、零 prefix。它是适配训练输入布局，不代表把推理端实际 KV cache page size 改成 1；v89 推理配置使用 page size 64。

**推理端配套。** 实验 SGLang 的 `SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE=1` 让 decode 使用配套 unified attention 路径；该选项默认 false，且要求 deterministic inference。只改训练端 attention 不能推导出 rollout 一定使用相同数值实现。

**默认条件。** 训练端开关 `MEGATRON_USE_SGLANG_QWEN3_CORE_ATTENTION` 默认 `0`；推理端还要正确设置 deterministic 与 decode 开关。

**代码。** `qwen3.py:349–498`；实验 SGLang `layers/attention/triton_backend.py:221`。

### 3.6 Attention：输出投影

**作用。** 多头 attention 的输出通过 O projection 映射回 hidden size。

**修复。** `_wrap_sglang_output_projection` 按参考 SGLang 的 attention TP 数，将输入与权重沿输入宽度分片；每片调用对齐 GEMM，然后按代码顺序相加。

```text
partial_i = aligned_mm(attention_output_i, weight_i.T)
output = partial_0 + partial_1 + ...
```

**原因。** 一个完整矩阵乘法，与多个分块矩阵乘法先各自舍入再相加，在 BF16 下不保证一致。这里复现的是参考并行计算的数值边界，训练端不必真的具有相同 TP 拓扑。

**默认条件。** `MEGATRON_USE_SGLANG_QWEN3_OUTPUT_PROJECTION` 默认 `0`；`MEGATRON_QWEN3_SGLANG_ATTENTION_TP_SIZE` 默认 `1`，v89 显式为 `2`。TP 数设置错误会改变模拟的分块路径。

**代码。** `qwen3.py:308–345`。

### 3.7 Attention 后与残差相加

**作用。** 计算 attention delta 与进入 attention 前残差的和。

**原始差异。** 如果训练端先执行 BF16 add 并把结果交给下一步 norm，而参考端在 fused add+norm 中用 FP32 的和计算 norm，则进入 norm 的有效值不同。

**修复。** Qwen wrapper 替换 `self_attn_bda`：

```text
exact_sum = attention_delta.float() + residual.float()
visible_output = exact_sum.to(BF16)
visible_output._slime_qwen3_pre_mlp_sum_fp32 = exact_sum
```

对外仍然有 BF16 tensor，但额外保存未舍入 FP32 sum，供 MoE 前 norm 使用。要求 dropout=0、无 attention bias。

**默认条件。** 与下一步共用 `MEGATRON_USE_SGLANG_QWEN3_PRE_MLP_NORM`，默认 `0`。不是开了 router 就会顺便修复这里。

**代码。** `qwen3.py:153–213`。当前工作区 Megatron 还有另一条直接残差分支，详见第 6 节。

### 3.8 MoE 前 RMSNorm

**作用。** 将 post-attention 表示归一化后，交给 router 和 experts。

**修复。** norm wrapper 读取上一步保存的 FP32 sum；如果属性缺失会报错。计算平方、确定性 FP32 mean、rsqrt、乘 norm 权重，最后转换回输出 dtype。

**核心区别。** `RMSNorm(round_bf16(a+b))` 与 `round_bf16(RMSNorm_fp32(a+b))` 不应当视为相同操作。这里修复的是舍入边界，不仅仅是把“加法 dtype”换成 FP32。

**默认条件。** 与 3.7 相同。CUDA/XPU 路径使用 `mean_batch_invariant`；CPU fallback 使用普通 mean，CPU fallback 不能作为 GPU bitwise 结论。

**代码。** `qwen3.py:195–206`；`deepgemm_forward.py:696–713`。

### 3.9 Router：专家分数、top-k 与路由权重

**计算。** 对 norm 输出 `x` 计算 gate logits：`logits = x W_gateᵀ`。选择 k 个专家，计算选中专家的概率。当前 Qwen 对齐路径要求 FP32 router、softmax score、无 capacity dropping、无 gate bias，并禁止 pre-softmax 模式。

**训练端 GEMM 修复。** `_patch_router` 替换 router.gating；`_SGLangRouterGEMMWithMegatronBackward` 将输入和权重转换 FP32 后调用 `matmul_persistent`。

**top-k 顺序修复。** 注册 router 的 ordered top-k capture。Megatron 的 `moe_utils.py:714` 把 compute_topk 接到 routing replay wrapper；对注册且非 grouped 的路由器，使用 `torch.topk(..., sorted=False)` 并捕获列顺序。专家集合相同但列顺序不同，也会影响后续有限精度汇总，因而需要保存顺序，而不仅保存布尔 routing map。

**softmax 两端处理。** Megatron 在非 pre-softmax 分支对选中 logits 计算 FP32 softmax。实验 SGLang 的 `_select_experts` 在 deterministic 模式下也显式执行 `topk(sorted=False)`、gather 选中 logits、FP32 softmax。因此上一轮“没有 Qwen 专门修改 softmax”的说法仅适用于训练端公式；**SGLang 实验分支确实有专门选择这条计算路径的实现。**

```text
ids = topk(logits, k, sorted=False).indices
weights = softmax(gather(logits, ids), dtype=FP32)
```

**SGLang GEMM 的阅读陷阱。** `_router_logits` 调用 `router_matmul`，router tile 实验默认 off 时内部写的是 `torch.mm`。但 deterministic model runner 会注册 batch-invariant 的 `aten::mm` 实现。因此不能只看 `torch.mm` 字面就认定它必然走普通 cuBLAS。额外 router tile candidate 实验 `SLIME_QWEN3_ROUTER_TILE_MODE` 默认 `off`，不是完整对齐的必需开关。

**默认条件。** 训练端一旦进入有效 Qwen hook 且选中层，router patch 就会安装，没有另一个 Qwen router 环境开关；hook 自身默认不调用。SGLang 的这两个专用分支由 deterministic inference 控制，默认关闭。

**代码。** `qwen3.py:500–529`；`deepgemm_forward.py:278–300`；`routing_replay.py:49–70`；实验 SGLang `models/qwen3_moe.py:325–355`、`models/qwen3_router_tile.py:42`、`model_executor/model_runner.py:714`、`batch_invariant_ops/batch_invariant_ops.py:980`。

### 3.10 将 token 分发给专家：DeepEP dispatch

**作用。** 根据 top-k 结果把 hidden states 发给持有对应专家的 rank。一个 token 可能访问多个 rank，也可能访问同一 rank 上多个专家。通信 token 行、expert-major 行、top-k slot 不是同一个索引空间。

**修复/适配。** bridge 包装 comm manager：保留 source top-k 顺序，调用原始 dispatch，再增加 route-preserving metadata dispatch。元数据包含用于识别 route 的 fingerprint、专家 ID 和权重等，帮助验证原始 token 接收布局与 route 级布局一致。

接收端使用 `_scatter_deepep_routes_with_padding` 生成 expert-major 输入、概率、output index 和 route positions，保存恢复时需要的映射。metadata 校验能发现 token 或专家映射错误，不能单独证明 FFN 数值完全一致。

**改动边界。** 这里仍然调用 `original_dispatch`。增加的是上层 bridge、排列及校验，不应声称此次对齐实现重新编写了 DeepEP 网络传输 kernel。

**配置约束。** Qwen hook 要求 `--moe-enable-deepep`，并要求固定 normal dispatch/combine 配置；两者 `num_sms` 必须相同且为正。固定配置避免各 rank 自动选择不同运行参数，但不是独立的 bitwise 证明。

**默认条件。** 随 Qwen hook 的 bridge 安装；普通 Qwen 示例默认使用 alltoall，没有开启这套 DeepEP bridge。

**代码。** `qwen3.py:46–72`、`:748–756`；`deepgemm_moe_forward.py:2739`、`:2828–2942`。

### 3.11 专家 FFN 与路由概率加权

**作用。** Qwen 的 SwiGLU 专家可概念性写成：

```text
u = SiLU(x W_gate) * (x W_up)
y_expert = u W_down
contribution = route_weight * y_expert
```

**FFN forward 修复。** `_wrap_sglang_triton_moe` 获取各本地专家 FC1/FC2 权重，构造 SGLang `TritonMoeQuantInfo` 和 `TritonRunnerCore`，用其 BF16 Triton forward 生成可见结果。接收数据已经按 expert 展开，所以局部 runner 配置 `top_k=1`；这不意味着模型从 top-8 改成 top-1。

**概率乘法位置修复。** 传入真实 `permuted_probs`，使 route weight 按 SGLang 的 FC2 路径融合。先得到 BF16 expert output 再乘概率，与在 FC2 累加/输出过程中乘概率，舍入结果可能不同。

**避免重复乘概率。** 通用 bridge 默认 `_slime_defer_router_probabilities=True`，Qwen 在 bridge 安装后再把已包装专家的该值恢复为 False。由此明确：概率已经归专家 forward 所有，combine 不应再乘一次。

**backward。** 无梯度 forward 直接返回 Triton 输出。有梯度时执行原始专家路径取得 autograd 关系，再通过 `_SGLangForwardMegatronBackward` 组合参考 forward 与训练 backward。因此这是 forward 一致性修复，不是新写一套与 SGLang 对齐的 backward。

**过时注释。** `qwen3.py:535–537` 仍写“单位权重、延后乘概率”，与实际实现不一致；实际调用和 bridge 后标志恢复才是判断依据。

**默认条件。** Qwen hook 对选中 MoE 层安装，没有独立环境变量；未配置 hook 时不安装。

**代码。** `qwen3.py:532–638`、`:757–764`。

### 3.12 专家贡献汇总与 DeepEP combine

**作用。** 最终需要把同一 token 的各专家贡献合成一个 hidden 向量并送回 source rank。

**关键差异。** 以下两种做法数学上等价，但 BF16 归约树不同：

```text
A: 接收 rank 先合并本地专家贡献 → 跨 rank combine
B: 每条 route 单独返回 source rank → source rank 再汇总 top-k
```

**Qwen 修复采用 A。** `_SGLangEPGatherWithBF16Backward` 在接收端本地汇总，传单位权重是因为真实概率已在 FFN 内乘过。之后使用原始 `manager.handle` 调用 `fused_combine`。

**另一个 handle 的含义。** bridge 也保存 route metadata handle；deferred-probability 分支可以使用它。但 Qwen 把 defer 置 False，选择原始 handle，不能因为代码中出现 route_handle 就误判 Qwen 走 B。

**默认条件。** 依赖 bridge 已安装以及 Qwen 概率所有权标志正确。它与专家概率融合必须成套使用。

**代码。** `deepgemm_moe_forward.py:2960–2977`、`:3049–3075`。

### 3.13 MoE 后与残差相加，以及进入下一层

**作用。** `layer_output = moe_delta + post_attention_residual`。

**Qwen 修复。** `_wrap_sglang_layer_output_residual` 用 FP32 加法保存 `_slime_qwen3_layer_output_sum_fp32`，对外输出仍转回 BF16。下一层 input norm/QKV wrapper 从前一层模块读取这个 sum，用它计算可见 norm forward。

**为什么层输出一致仍不够。** 两个实现输出的 BF16 tensor 可以一致，但参考端还保留了 delta/residual 两个分量；下一层 fused norm 能使用未舍入的和。如果训练端只剩 BF16 输出，就无法恢复这部分信息。跨层 carry 修复针对这一差异。

**安装条件。** Qwen hook 在处理第 L 层输入 norm 时，只有前一层 L-1 也被选中、且前一层模块位于当前 model chunk，才包装前一层的残差输出。因此仅选择第 0 层时，不会因这段 Qwen 逻辑自动安装第 0→1 层 carry；选择 `[0,1]` 并打开输入 norm/QKV 开关才会。完整 `[0..47]` 可覆盖相邻层，而不是自动把最后一层 carry 送入 final norm。

**默认条件。** 依赖 `MEGATRON_USE_SGLANG_QWEN3_INPUT_NORM_QKV=1`、相邻选层及 hook；默认不启用。

**代码。** `qwen3.py:216–253`、`:694–709`。工作区 Megatron 的通用残差分支另有行为，不能混用属性名称，见第 6 节。

## 4. 默认开关与调用链

### 4.1 第一层门：是否调用 Qwen alignment hook

参数定义 `slime/slime/utils/arguments.py:1463–1471`：

```text
--custom-megatron-before-log-prob-hook-path     默认 None
--custom-megatron-before-train-step-hook-path   默认 None
--qwen3-alignment-layers                       默认 None
```

`model.py:452` 与 `:555` 只有在对应路径非空时才 load 并调用 hook。代码没有依据“这是 Qwen3 模型”自动调用 `enable_qwen3_moe_alignment`。

Qwen hook 要求 deterministic mode、DeepEP、固定配置和非空 selected layers，否则报错。函数内部创建 SGLang deterministic runtime context 也只有在调用 hook 后才发生；它不是全局默认。

### 4.2 第二层门：各计算段的开关

| 条件/开关 | 代码默认 | 控制内容 | v89 保存配置 |
|---|---|---|---|
| 两个 custom hook path | None | logprob 与训练阶段安装修复 | 显式设置 |
| qwen3 alignment layers | None | 哪些层被包装 | 0–47 |
| QWEN3_INPUT_NORM_QKV | 0 | 输入 norm、QKV、相邻层 carry | 1 |
| QWEN3_QK_NORM | 0 | Q/K RMSNorm | 1 |
| QWEN3_PRE_MLP_NORM | 0 | attention residual 与 pre-MoE norm | 1 |
| QWEN3_CORE_ATTENTION | 0 | attention core forward | 1 |
| QWEN3_OUTPUT_PROJECTION | 0 | O projection | 1 |
| MEGATRON_QWEN3_SGLANG_ATTENTION_TP_SIZE | 1 | O projection 模拟的分片数 | 2 |
| MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS | 0 | final norm wrapper；工作区 Megatron 还有通用残差分支 | 1 |
| SGLang enable_deterministic_inference | False | batch-invariant、Qwen router 特殊分支等 | True |
| SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE | false | unified decode attention | 1 |
| SLIME_QWEN3_ROUTER_TILE_MODE | off | 独立的 router 小 tile 实验 | 未显式设置 |

表中 `QWEN3_*` 五项均省略共同前缀 `MEGATRON_USE_SGLANG_`。Router、Triton experts、DeepEP bridge 没有对应的额外五项式开关，但仍受 hook 与选层控制。

SGLang 的例外是显式设置非空 `rl_on_policy_target` 可以自动启用 deterministic inference，见实验版 `server_args.py:7644`。这仍然不表示训练端 Qwen hooks 自动启用，也不表示上述所有开关随之开启。

## 5. 三种启动入口分别是什么状态

### 5.1 普通完整 Qwen3 示例

入口：[run-qwen3-30B-A3B.sh](../slime/scripts/run-qwen3-30B-A3B.sh)。

脚本未设置两个 Qwen hook、选层列表、五个环境变量，也未显式开启 SGLang deterministic inference 或 Megatron DeepEP。模型参数指定 `--moe-token-dispatcher-type alltoall`。因此**脚本自身不默认开启完整对齐**。

它的 `--qk-layernorm` 只是启用模型本来应有的 Q/K norm，不是启用 SGLang norm 替换。`--moe-router-dtype fp32` 也仅指定精度，不等于使用对齐 GEMM。

本次只读脚本，没有执行其清理和启动命令。

### 5.2 当前两层 A100 SGLang 实验启动脚本

入口：[run_sglang_qwen3_a100_deepep_triton.sh](../experiments/run_sglang_qwen3_a100_deepep_triton.sh)。

它明确导入 `experiments/sglang-main-sm80/python`，默认使用两层模型，设置 TP4/EP4、DeepEP normal、Triton MoE/attention 与 BF16 dispatcher output。

但是脚本没有传 `--enable-deterministic-inference`，没有设置 unified decode 开关，也没有固定 DeepEP config。它仅启动 SGLang server，根本没有训练端模型可供 Qwen hook 包装。因此**使用 DeepEP+Triton 不代表启动了完整修复**。

外部环境或上层 launcher 可以改变最终运行行为；这里判断的是该文件自身提供的默认设置，未声称已读到某个在线进程的最终参数。

### 5.3 v89 专用回归配置

依据：[v89_run.sh.txt](../slime/alignment_a_preparation/source_snapshots/v89_run.sh.txt) 与 [v89_runtime_env.yaml.txt](../slime/alignment_a_preparation/source_snapshots/v89_runtime_env.yaml.txt)。

v89 脚本在 183–187 行显式指定 DeepEP、deterministic、两个 Qwen hooks、全部 48 层；环境文件在 30–38 行显式设置五个 Qwen 开关、FUSED_RESIDUAL_RMS、attention TP=2、unified decode。推理端还显式设置 deterministic、Triton、FP32 router、normal DeepEP 固定配置。

因此答案是：**v89 配方显式开启；普通 Qwen3 默认不开启。** 保存的配置是历史配方，不等于现在直接执行仍可成功；其中路径和依赖需要重新核实。

## 6. 当前代码组合有一处必须单独核实的残差兼容性问题

v89 的 PYTHONPATH 引用的是：

```text
/public/home/yingxy2024/slime-deps/Megatron-LM
```

不是当前项目内：

```text
/public/home/yingxy2024/moe-consistency/Megatron-LM
```

本次读取两处层代码：外部依赖版本没有工作区版本的 `_use_sglang_fused_residual_rmsnorm` / `_sglang_residual_sum_fp32` 分支；工作区版本则有以下实现。

1. `transformer_layer.py:672` 在 FUSED_RESIDUAL_RMS=1 时直接处理 attention residual，绕过 `self_attn_bda`。
2. `:727` 直接用本文件 `_sglang_native_rmsnorm_from_fp32_sum` 计算 pre-MLP norm；该 helper 使用 `.mean`，不是 Qwen helper 对 `mean_batch_invariant` 的显式调用。是否被全局 dispatcher 替换需看真实进程初始化。
3. `:885` 在该开关开启时直接处理 MLP residual，绕过 Qwen 包装的 `mlp_bda`。
4. 工作区代码保存 `_sglang_residual_sum_fp32`，而 Qwen 下一层输入 wrapper 要读取 `_slime_qwen3_layer_output_sum_fp32`。两者不是同名属性。

**静态推断：** 如果把 v89 的全部开关直接套到当前工作区 Megatron，并运行经过这些分支的普通层 forward，Qwen wrapper 的写入可能被绕过，后续层可能报“missing the previous layer's exact FP32 residual sum”。这次没有启动 GPU，不能把此推断当作已复现的运行错误；但可以确认这两条代码路径存在绕过和属性不一致。

此外，当前 final RMSNorm wrapper 会寻找通用 `_sglang_residual_sum_fp32`；工作区 Megatron 最后一层也会生成该值。历史 Qwen 交接报告曾明确要求 final norm 消费已舍入 BF16 层输出，而非套用 GLM 最后一层 FP32 carry。因此这也是更换依赖后必须重新验证的边界。

**实际含义：** 不建议只复制环境变量就宣称“全开启且精确”。首先确认实际 import 的 Megatron 路径，再确认 residual wrapper 是否真正执行。本次任务是审计，没有擅自合并或删除两套实现。

## 7. 如何确认某次运行真的开启

以下是核验顺序，不是已经执行成功的 GPU 结果。

1. **实际依赖路径。** 在同一个 Ray worker/runtime env 中记录 `sglang.__file__`、`megatron.core.__file__` 与 Qwen alignment 模块路径。登录 shell 的 PYTHONPATH 不一定等于 worker 的路径。
2. **最终参数。** 确认两个 hook path、selected layers、deterministic、DeepEP、固定 config 以及五个 Qwen 环境变量。环境变量要传入真正运行训练的进程。
3. **安装日志。** 搜索 `Enabled Qwen3 alignment for global MoE layers`；日志会列出 input-norm/QKV、Q/K norm、attention、output projection、pre-MLP norm、Triton MoE 和 layer-output residual 的安装层列表。
4. **不要只相信 wrapped 标志。** 包装成功不代表分支一定调用它；第 6 节是反例。检查实际执行的 residual 与 norm 分支。
5. **检查双方。** 训练端开启不代表推理端 deterministic/unified decode 开启；推理端 deterministic 也不会自动安装训练 hooks。
6. **执行精确 gate。** 比较同一权重、token、位置与语义相同的中间 tensor，再比较层输出与最终 logprob。SGLang 分离的 delta/residual 之一不能直接与 Megatron 合并输出比较。

CPU 静态测试能检验包装和路由元数据逻辑，但无法替代 A100 上的 BF16 GEMM、attention 与 DeepEP 数值验证。

## 8. 历史验证与本次审计的界限

[QWEN3_ALIGNMENT_HANDOFF_V66.md](../slime/QWEN3_ALIGNMENT_HANDOFF_V66.md) 的后续章节记录了 v69 的跨层 carry、v71 的 48 层输出、v75 的 final RMSNorm，以及 v89 两轮各 1024 个 target token 的精确 logprob gate。文件名虽含 V66，内容已追加到 v89，不能只读开头的早期状态。

这些是历史报告记录，本次未重新生成结果，也没有验证所有历史结果文件仍存在。报告的证明范围是特定 BF16 模型、并行拓扑、kernel、固定配置和样本；不能推出任意 Qwen3、任意 backend、任意精度默认精确。

本次还纠正了两个容易误读的注释：`qwen3.py` 文件头称“只对齐 router”已经不符合当前完整实现；专家 wrapper 的“单位概率”说明也已过时。以函数执行路径、参数传递和安装条件为准。

## 9. GPU 状态交接

本轮为静态审计，不需要正式 GPU 计算。此前同一会话已查到 Slurm 作业 1024115 在 ai_hgx_04 上运行，但访问该节点的 SSH 认证失败；无法确认当前执行的是正式任务还是保号任务，也无法提供经过验证的训练日志或保号 PID。本次没有重复提交保号任务、没有停止现有作业，也没有把作业存在视为保号验证成功。
