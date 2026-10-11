# Breakable CUDA Graph 与 RMS、Q/K 融合叠加

已实现两个候选与 Breakable Graph 的叠加，并通过有限完整模型的数值和 batch
门禁。此次同条件测量中，RMS 融合的延迟下降为 5.239%；RMS＋Q/K 联合版本下降
4.885%。联合版比 RMS 单独版本慢 0.373%，差异小；单轮顺序测量没有证明 Q/K
带来额外稳定收益，也不足以认定它造成性能回归。

## 当前版本的完整 generate 测量

三组都是**确定性版本**，使用修复后的每批一次 KV 检查、decode Breakable、
prefill eager。此表不包含非确定性 baseline，也不使用先前 KV 修复前的计时。

| 配置 | 正式批数 | 中位数秒 | P95 秒 | 输出 tokens/s | 相对原 Breakable 延迟下降 |
|---|---:|---:|---:|---:|---:|
| 原 Breakable，RMS/QK 候选关闭 | 20 | 6.460628 | 6.570308 | 158.50 | — |
| Breakable＋残差 RMS 融合 | 20 | 6.122183 | 6.218834 | 167.26 | 5.239% |
| Breakable＋残差 RMS＋联合 Q/K | 20 | 6.145010 | 6.287728 | 166.64 | 4.885% |

ai_hgx_03，8×A100-SXM4-80GB，Slurm 1031236，SSH extern；Qwen3-30B-A3B BF16，
TP8/DP2/attention CP2/EP4/MoE DP2；Triton attention/MoE，DeepEP normal 固定 20 SM。
每组独立 Engine，先进行一个不计时的 3-prompt、2-output-token 小 bucket 预热，
再执行 3 批标准预热＋20 批正式测量。每批 8 个冻结 prompts，共 1218 输入 tokens，
每个 prompt 固定输出 128 tokens，共 1024 输出 tokens。总计 60 批正式、69 批
标准生成样本；额外的小 bucket 预热不纳入统计。

范围为 blocking `Engine.generate`：包含调度、prefill、decode、采样、输出 logprob
和 detokenization/IPC；不包含加载、捕获、预热、cache flush、训练或权重同步。
三组均无 profiler、逐算子审计或融合 shadow 检查。所有正式批次零缓存命中。
第一轮回放证据写盘全部发生在最后一批预热完成前，没有进入正式测量。
23 组三模式采样输出哈希全部一致；所有 benchmark worker 退出码 0，运行期间源码
SHA256 不变。

三组仅各启动一次，顺序为原 Breakable、RMS、RMS＋Q/K；没有多轮启动顺序轮换。
不能把上面的百分比视作高精度的稳定加速率，尤其不能对 0.373% 的小差异作强结论。
当前已有收益的选择是 RMS 融合；Q/K 叠加开关保留为可选实验。

## 使候选真正进入 Graph 的改动

1. 残差 RMS 候选原先明确拒绝 CUDA Graph。新增显式
   `SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH=1`，只允许 `candidate` 且 audit 关闭。
   默认开关为 0；有审计/shadow 的路径继续拒绝捕获，kernel 的运算顺序不变。
2. Q/K hook 原先在 `is_current_stream_capturing()` 时回退。新增显式
   `allow_cuda_graph=True`，只允许无审计的 joint/strided 模式；默认仍为 False。
3. SGLang 原来在所有 Graph 捕获后才注册 forward hooks，算子替换因此无法进入
   已捕获的 Graph。新增 hook spec 的显式 `before_cuda_graph_capture` 选项，
   让需要替换算子的工厂在 eager runner 预热和捕获之前安装。普通诊断 hooks
   继续默认在捕获后注册。Q/K 候选与捕获证据工厂显式选择新时机。
4. 新增 `breakable-rms` 与 `breakable-rms-qk` 实验模式和捕获证据。
   生产默认配置没有自动启用任何候选。现有 BF16、小 M、hidden/head dimension、
   权重 dtype 和布局范围检查保留，超出范围仍回到原实现。

## 捕获与完整模型验证

两种融合模式各 8 ranks×2 buckets（2、4），每个 rank/bucket 均有：

| 模式 | 实际捕获的 RMS 候选调用 | 实际捕获的 Q/K 联合调用 | Graph 段 | Eager MoE 边界 |
|---|---:|---:|---:|---:|
| RMS | 95 | 0 | 49 | 48 |
| RMS＋Q/K | 95 | 48 | 49 | 48 |

95 次 RMS 对应 48 个 pre-MLP norm 和 47 个带残差的 input norm；第一层 input
norm 没有残差，未进入该候选。证据计数在捕获时采集，在捕获结束后写盘。
另有所有 rank/bucket 的实际成功回放证据，不能把仅设置 backend 当作 Graph 生效。

- 原 RMS dispatch/audit CPU 检查：9 项通过。
- 原生 CUDA Graph 张量检查：M=1/2/4、BF16/FP32 权重、packed QKV stride；
  在捕获后改变输入和权重，30 次回放、448000 个 BF16 输出元素零 bit 差异，
  包括零值、符号零、残差抵消和不同数值尺度。
- 两个融合版本各执行 batch8、batch3、batch1、variable3 四组有限 greedy 输入，
  每请求 16 输出 tokens；与当前 KV-once 原 Breakable 参考相比，tokens、专家
  routes、response logprob bits 全部一致，batch8 对 batch3/batch1 门禁通过。
- 本次没有运行跨框架训练门禁，不宣称任意输入或训练的一致性。

第一轮尝试（run 01）被捕获证据检查拦下：当时 proof hook 安装在捕获之后，
无法提供捕获计数，因此没有继续计时。run 02 在 GPU 启动前被资源检查拦下：
旧 SSH 会话丢失了 memory cgroup 归属。重新 SSH 后恢复到原 allocation 的
`step_extern`；正式结果来自完整通过的 run 03。失败目录保留，未并入本表。

## 后续加速调查方向

Breakable Graph 已经把段内的 CPU kernel 发射变成 Graph replay。融合仍可减少
GPU kernel 和布局读写，但其收益不能照搬原 eager 测量。

源代码与边界证据表明，当前 48 个边界的整个 MoE body 仍是 eager：dispatch、
专家计算、combine 和写入静态桥接输出。值得下一步验证的是把专家计算单独
Graph 化，继续让需要 CPU 动态接收数量的 dispatch/combine eager。这需要处理
变化的接收行数、静态输入输出地址和不同路由，尚未实现或测量，不能预报收益。
此前 QKV/O 小 tile 的张量微基准也可作为后续候选，尚未与本次 Graph 叠加。

## 证据与交接

完整运行目录（相对本报告）：`gpu_runs/breakable-fusions-hgx03-20261011-03/`。

- `results.json`、各模式 `summary.json` 和 `measurements.jsonl`：测量及百分比。
- `measurement-validation.json`：计时协议、缓存、采样输出哈希、回放证据与源码锁验证。
- `graph-tensor-gate.json`、`compare-breakable-rms.json`、
  `compare-breakable-rms-qk.json`：独立回放与完整模型/batch 门禁。
- 各模式 `graph-evidence/fusion-rank-*-size-*.json`：实际捕获的候选调用计数。
- `source-snapshots/`、`manifest.json`：运行时源码和 SHA256。
- Driver 日志：项目 `.a-tmp/breakable-fusions-hgx03-20261011-03-driver.log`。

Driver PID/PGID 79298 及所有正式 worker 已退出。交接时 8 张 GPU 均为 1 MiB、
0% 利用率，无 compute process；Slurm 1031236 保留 RUNNING。按用户先前要求未
恢复保号。代码与本报告保留在当前工作分支，尚未新建提交或推送。
