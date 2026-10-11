# 非确定性 baseline＋Breakable 的可迁移优化

已把专家计算 Graph、DeepEP 固定容量 dispatch 分别及组合接入非确定性 baseline，
通过受控完整模型门禁，并完成四组同条件 generate 测量。当前组合版本最快：
3.110243 秒，相比原 baseline＋Breakable 的 5.179104 秒，延迟下降 39.946%。

## 哪些适用

| 前序改动 | 非确定性 baseline 的情况 | 本次处理 |
|---|---|---|
| 专家计算独立 Graph | 不要求 batch-invariant 数学路径，BF16/Triton core 可以捕获 | 已接入、验证并计时 |
| DeepEP 固定容量 dispatch | 可以跳过 normal CPU 接收计数等待，需处理填充行 | 已单独及组合接入、验证并计时 |
| 残差 RMS 候选融合 | 为确定性 `forward_native` 的 FP32 运算顺序设计；baseline 已使用 native fused add＋RMSNorm | 保留 baseline 现有融合，不替换 |
| 联合 Q/K 候选 | 显式限定 batch-invariant 分支；baseline 已有 eligible JIT fused inplace QK norm | 保留 baseline 现有路径，不替换 |
| KV 每批一次检查 | 修复的是 unified deterministic decode；baseline unified decode 关闭，不执行此检查 | 代码仍存在，baseline 路径不触发 |
| QKV/O 小 tile | 前序仅验证了确定性 matmul 的张量微基准；baseline 使用其它线性算子路径 | 未移植或计时 |
| full decode Graph | 是替换图后端，并非保持 Breakable 的叠加项 | 本次按用户要求保持 Breakable |

baseline 的 `models/utils.py:apply_qk_norm` 在 CUDA、equal epsilon、非确定性、
prefill tc compiler 非 inductor、head/dtype 支持时调用 fused inplace QK norm。
本试验为 BF16/head128、tc compiler=eager，未打开 fused QK+RoPE 的另一选项。
baseline 的 residual norm 走 `fused_add_rmsnorm`，普通 norm 走 `rmsnorm`。
因此旧的确定性候选不应当被当作 baseline 新增加速项；这不意味着不能另写
baseline 专用数学融合，只表示本次原候选不直接适用。

## 同条件计时

| baseline decode 配置 | 正式批数 | 中位数秒 | P95 秒 | 输出 tokens/s | 相对原版延迟变化 |
|---|---:|---:|---:|---:|---|
| 原 Breakable | 20 | 5.179104 | 5.244751 | 197.72 | — |
| ＋专家 Graph | 20 | 3.271475 | 3.359833 | 313.01 | 减少 36.833% |
| ＋固定容量 dispatch，专家 eager | 20 | 5.701649 | 5.830163 | 179.60 | 增加 10.089% |
| ＋固定容量 dispatch＋专家 Graph | 20 | 3.110243 | 3.169973 | 329.23 | 减少 39.946% |

组合版比专家 Graph 单独启用再减少 4.928% 的中位延迟。固定容量独立启用虽然
通过数值门禁，但本轮更慢，不建议作为独立加速开关。它的填充清理仍产生 eager
算子发射和额外工作；组合版把这些清理一并捕获进专家 Graph。本次未采集新的
profiler launch 计数，不能精确归因各成本的份额。

同一 ai_hgx_03，8×A100-SXM4-80GB，Slurm 1031236，SSH extern；
Qwen3-30B-A3B BF16，TP8/DP2/attention CP2/EP4/MoE DP2，Triton attention/MoE，
DeepEP normal。使用 baseline 原有 DeepEP config 选择，没有套入确定性版本的
显式 config。四组均验证 `enable_deterministic_inference=False`、unified decode=0、
RMS 候选 off、decode backend=breakable、prefill backend=disabled。

每组一个独立 Engine，额外小 bucket 预热在计时之外，再 3 批标准预热＋20 批
正式。8 个冻结 prompts、1218 输入 tokens、每请求输出 128 tokens，共 1024
输出 tokens。合计 80 批正式、92 批标准生成样本。范围为 blocking generate，
包括调度、prefill、decode、采样、logprob、detokenization/IPC；排除加载、捕获、
预热、cache flush、训练及权重同步。四组无 profiler/数值 shadow/逐调用文件 I/O、
零缓存命中、退出码 0，运行过程中源码 SHA256 不变，回放证据写盘均在预热完成前。

四组只各启动一轮，顺序为 original、experts、fixed、fixed-experts，未做反序复测。
这里的百分比是本轮观测，不是高精度稳定加速率；也不与此前确定性 full Graph
时间混算确定性额外开销。

## 组合实现

外层仍有 49 个 Graph 段、48 个整个 MoE 的 eager 回调。

- 专家 Graph 单独启用：每 rank 为 48 层×接收行数 1–16，共 768 个缓存。
- 组合启用：fixed dispatch 输出恒为 16 行，缓存收窄为每层一个，共 48 个。
- 安装 fixed hook 后，专家 Graph 捕获包括设备侧填充清理和原专家 core。
  每次更新 hidden states、路由 ID、权重；没有缓存路由或 DeepEP handle。
- dispatch/combine 保持 eager。组合的清理 GPU 工作进入专家 Graph，数值
  shadow 仍可把有效行取出，与原 actual-M core 对照。
- `expert_graph_hooks.py` 新增可选 `cache_rows` 参数；省略时仍为原 1–16。
  所选缓存之外的输入回到原 core，权重存储替换仍要求重建。

这不是 full 后端。显式 baseline 工具不启用 RMS/QK 候选，也不改变生产默认配置。

## baseline 的 batching 与正确性验证

第一次运行 `...-01` 的专家 Graph 完整比较通过，但 fixed 的 batch3 两条样本
出现 routes/logprob 位差异，greedy tokens 相同，最大 logprob 绝对差约 0.1172。
路由差异从 prompt prefill 开始；fixed hook 只修改 decode。日志显示原版把两条
请求一起 prefill，而 fixed 验证进程分开执行。该失败被门禁拦下，未开始计时。

第二次运行 `...-02` 只在**数值验证**中统一 `prefill_max_requests=1`，使各版本
按相同请求分组计算；正式计时保持默认 `prefill_max_requests=None`，不加调度延迟。
受控验证中，三个候选与同一个原 baseline 参考的 batch8、batch3、batch1、
variable3（每请求 16 输出 tokens）全 tokens/routes/logprob bits 一致。
baseline 不提供跨 batch 数值不变性，不把这一项当作它的要求。

第二次运行的组件 shadow 汇总，均为 8 ranks 合计：

| 候选 | 比较对象 | 实际调用 | 比较 BF16 元素 | 结果 |
|---|---|---:|---:|---|
| experts | 专家 Graph 对原同 M core | 20239 | 170950656 | 零 bit 差异 |
| fixed | 固定容量有效行对原 actual-M core | 26496 | 189480960 | 零 bit 差异 |
| fixed-experts | 专家 Graph 对同容量 core，包含填充行 | 24576 | 805306368 | 零 bit 差异 |
| fixed-experts | 固定容量有效行对原 actual-M core | 26304 | 189480960 | 零 bit 差异 |

另有专家 Graph 的第 0/47 层变化输入/路由合成检查；各 rank 的实际成功回放和
缓存数检查通过。未运行跨框架训练、参数重绑/offload 或其它拓扑的门禁。

自然调度的性能样本仍可能采用不同 prefill grouping。与 original 的 23 组采样
输出哈希匹配数分别为：experts 18、fixed 18、fixed-experts 17。因此不宣称正式
计时的输出逐 bit 不变，也不能仅由哈希差异归因为候选算法回归。性能比较固定
输入和输出长度，但不是固定每步 token 身份或路由的 GPU 工作量。

## 可复跑工具与证据

- `gpu_tools/baseline_graph_experiment.py`：四模式配置和 baseline 环境。
- `gpu_tools/baseline_graph_model_gate.py`：受控 prefill 的同 workload 比较。
- `gpu_tools/baseline_graph_inference_benchmark.py`：默认调度 generate 测量。
- `gpu_tools/run_baseline_graph_optimizations.py`：资源锁、门禁和四组计时。
- `gpu_runs/baseline-breakable-opt-hgx03-20261012-02/`：完整结果、测量校验、
  `compare-*.json`、shadow/回放证据、`source-snapshots/`。
- `gpu_runs/baseline-breakable-opt-hgx03-20261012-01/`：保留首次 batching 差异失败。

## 交接

正式 driver PID/PGID 77630 和全部 worker 已退出；首次 driver 57865 也已退出。
8 张 GPU 均 1 MiB、0% 利用率、无 compute process；Slurm 1031236 保留 RUNNING。
按此前用户要求未恢复保号。日志位于项目
`.a-tmp/baseline-breakable-opt-hgx03-20261012-02-driver.log`。
本阶段实现与报告保存于 `codex/baseline-breakable-optimizations` 分支。
本次提交范围仅包含上述已完成的 baseline Breakable 优化；后续 baseline full
实验与整个目录的总览文档另行整理，不混入此阶段提交。
