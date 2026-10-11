# Breakable Graph 内部的专家计算 Graph 调查

已把 DeepEP normal 的专家计算单独放入原生 CUDA Graph，并与原 Breakable＋RMS
做同条件完整 generate 对比。中位数从 6.171170 秒降至 4.307625 秒，延迟下降
30.198%，吞吐提升 43.262%。有限张量、完整模型和 batch 数值门禁均通过。

## 同条件测量

| 配置 | 正式批数 | 中位数秒 | P95 秒 | 输出 tokens/s |
|---|---:|---:|---:|---:|
| Breakable＋RMS，专家计算 eager | 20 | 6.171170 | 6.250004 | 165.93 |
| Breakable＋RMS＋专家计算 Graph | 20 | 4.307625 | 4.375092 | 237.72 |

两组均为确定性版本，使用相同的 KV-once 检查、残差 RMS 融合、decode Breakable、
prefill eager，Q/K 联合候选关闭；TP8/DP2/attention CP2/EP4/MoE DP2，Triton
attention/MoE、DeepEP normal 固定 20 SM。模型为 Qwen3-30B-A3B BF16，节点
ai_hgx_03 的 8×A100-SXM4-80GB，Slurm 1031236，SSH extern。

每组一个独立 Engine；一个不计时的 3-prompt、2-token 小 bucket 预热，随后
3 批标准预热＋20 批正式测量。每批固定 8 prompts、1218 输入 tokens，每请求
输出 128 tokens，共 1024 输出 tokens。总计 40 批正式、46 批标准生成样本。
范围为 blocking `Engine.generate`，包括调度、prefill、decode、采样、输出
logprob 和 detokenization/IPC；不包括加载、捕获、预热、cache flush、训练或权重同步。

所有正式样本零缓存命中，无 profiler、数值 shadow 或逐调用文件 I/O；运行期间
源码 SHA256 不变。两组 23 对采样输出哈希全部一致，所有 worker 退出码 0。
外层和专家 Graph 的首次回放证据都在最后一批预热完成前写完。

这是每组一次顺序启动的测量，未做多轮顺序轮换。30.198% 是本轮观测结果；
不宣称高精度稳定加速率。不能把这里的确定性专家 Graph 与此前未启用专家
Graph 的非确定性 baseline 直接比较，作为确定性额外开销的估计。

## 切分方式

```mermaid
flowchart LR
    A[外层 Graph：Attention / Gate / TopK / RMS] --> B[Eager DeepEP dispatch]
    B --> C[一次 GPU 输入复制]
    C --> D[按实际接收行数选择专家 Graph]
    D --> E[Eager DeepEP combine]
    E --> F[写入原静态桥接输出]
    F --> G[后续外层 Graph]
```

外层仍是 49 个 Graph 段、48 个整个 MoE 的 eager 回调。变化发生在回调内部：
`run_moe_core` 的路由类型转换/布局整理、专家 gate/up GEMM、激活、down GEMM
和本地专家归约被捕获。dispatch/combine 的通信、CPU 接收计数等待及外层桥接
复制保留原路径。这不属于整个模型的 full CUDA Graph。

实现位于可选实验 hook `expert_graph_hooks.py`。新增
`breakable-rms-experts` 模式，沿用捕获前安装机制，未改动专家数学 kernel，也未
修改 DeepEP 通信实现或生产默认开关。

## 动态形状、路由和地址

- 在第一次外层 eager 预热完成后、实际外层捕获之前，使用每层已加载的真实
  BF16 权重，预建接收行数 1–16 的原生 Graph。每 rank 为 48×16＝768 个 Graph。
  16 是本试验 decode bucket≤4、EP4 的保守容量界限；实际调用不做行数填充。
- 每次读取真实 dispatch 输出的行数 M，选择相同 M 的缓存 Graph，保持原专家
  计算形状和 config 选择。缓存的是计算，不是路由或 DeepEP handle。
- 一个 Triton 复制 kernel 把本次 hidden states、INT64 top-k IDs、FP32 top-k
  weights 写入三个固定地址的输入 buffer。下一次路由变化会更新这些 buffer，
  捕获图内的布局整理据此重新执行。
- Graph 输出保持有效引用，由原 combine 消费；原 communication handle 仍由
  每次 dispatch 产生并使用。每次支持的调用需要一次复制发射和一次 Graph 回放。
- 零行保留原 SM80 兼容路径，避免已有零行 invalid-grid/通信 buffer 问题。
  其他 dtype、布局或行数范围回退原计算；权重存储地址替换会报错，要求重建缓存。
- 限定固定模型、未量化 BF16 权重、H=2048、top-k=8、Triton runner 和无专家
  内部 fused all-reduce 的配置。参数重绑、不同量化/拓扑及 offload 生命周期
  尚未接入缓存重建，不能视为通用生产实现。

初始化 graph cache 的成本在预热阶段支付，没有正式调用期间的首次捕获。
验证进程中每 rank 的缓存构建耗时约 1.26–4.37 秒，该值包含验证模式下的
合成路由检查，不能当作纯捕获耗时。

## 数值证据

在验证 Engine 中，Graph 输出用于真实模型生成，同时把每次非空实际输入复制
一份送入原 `run_moe_core`，对照完整 BF16 输出位模式。参考复制独立于 Graph 的
静态输入，保留原实现可能的 inplace 行为。

| 检查 | 数量 | 结果 |
|---|---:|---|
| 第 0/47 层，M=1–16，三种数值尺度及变化的路由/权重 | 768 次，13369344 元素，8 ranks 合计 | 零 bit 差异 |
| 完整模型中的真实专家回放 | 24246 次，229115904 元素，8 ranks 合计 | 零 bit 差异 |
| 四组 greedy 输入：batch8、batch3、batch1、variable3；16 输出 tokens/请求 | 两版本 tokens/routes/logprob bits | 全部一致 |
| batch8 对 batch3/batch1 | 两版本 | 通过 |
| 128-token 采样输出哈希，含标准预热 | 23 对 | 全部一致 |

真实验证中接收行数出现 1、2、3、4、6、8；330 次零行调用保留原路径，未发生
不支持输入的回退。每个 rank 的 48 个专家层都出现实际 Graph 回放。没有执行
跨框架训练门禁，不宣称任意输入、训练或参数迁移场景一致。

## 显存抽样

专家 Graph 版本驻留显存抽样为每卡约 55.6–55.9 GiB，与参照运行中的实时
`nvidia-smi` 抽样相比，每张卡增加 1678 MiB，约 **1.64 GiB**。这是运行期间的
驻留抽样差，未采集峰值。

缓存记录里的 `allocated_memory_delta_bytes=27855360` 仅统计仍活跃 Tensor
的 allocator allocated 字节，未包含保留池/Graph 驱动内存，不能当作总显存开销。
较晚的 `vram-rms.json` 查询碰上参照 worker 清理、只看到 1 MiB，已标记无效，
未用于比较。有效参照保存于 `vram-rms-live-observation.json`，来源是正式测量
期间的交互式 SSH 工具输出；候选保存于 `vram-experts.json`。

## 结论与后续边界

这条切分路线在当前工作负载上有明确测量收益：专家计算保持相同 GPU 数学
工作，减少 eager 专家部分的逐算子 CPU 发射。它也新增了输入复制，因此不能
用 GPU 事件数量下降来代表本方案的收益；此处只报告无 profiler 的 generate 测量。
本轮未采集新的 CPU launch 计数，不量化不同 launch 类型的占比。

当前仍保留 48 次 normal dispatch/combine 的 CPU 路径及同步接收数量逻辑。
后续若尝试进一步去除它，应另行验证 DeepEP 固定容量 dispatch 的接入、填充
路由及数值行为，不能把本次专家 Graph 成功当作 full Graph 已兼容的证据。

## 证据与交接

运行目录：`gpu_runs/expert-graph-hgx03-20261011-01/`（相对本报告）。

- `results.json`、各模式 `summary.json`、`measurements.jsonl`：同轮性能数据。
- `measurement-validation.json`：源码锁、缓存、哈希、回放证据、显存和清理校验。
- `shadow-summary.json`、`exact-comparison.json`：数值及模型/batch 门禁。
- `exact-experts/graph-evidence/expert-shadow-rank-*.json`：真实调用逐 bit 检查汇总。
- 各模式 `graph-evidence/expert-cache-rank-*.json` 和
  `expert-first-replay-rank-*.json`：缓存、实际回放证据。
- `source-snapshots/`、`manifest.json`：运行时源码和 SHA256。
- Driver 日志：项目 `.a-tmp/expert-graph-hgx03-20261011-01-driver.log`。

Driver PID/PGID 113561 和正式 worker 已退出。8 张 GPU 均 1 MiB、0% 利用率、
无 compute process；Slurm 1031236 保留 RUNNING。按此前用户要求未恢复保号。
本次代码和报告留在当前工作分支，未新建提交或推送。
