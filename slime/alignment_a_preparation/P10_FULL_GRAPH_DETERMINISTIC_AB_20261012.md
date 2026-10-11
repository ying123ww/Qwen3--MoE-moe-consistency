# P10：Full Graph 三模式公平重复对照

P10完成。9个独立Engine、三轮顺序轮换、180正式批已核验。固定通信配置下，
完整确定性方案C比非确定性B慢 **69.939%**（三轮配对比例中位数，范围
69.214%–70.055%）。固定配置B比默认配置A快约 **1.243%**（配对比例中位数）。
这些是完整generate配置差距，不能解释成单个batch-invariant GEMM或RMS的代价。

## 实验配置

| 模式 | 确定性/unified | Norm | 通信配置 | Graph |
|---|---|---|---|---|
| A | 关闭 | baseline原生融合 | DeepEP默认选择 | decode full、prefill eager |
| B | 关闭 | baseline原生融合 | 显式固定20SM | 同A |
| C | 开启 | 已验证残差RMS融合 | 与B相同固定20SM | 同A |

同一提交 `68b8fef38eb83261f00c98b1dd2f44b746e7471b`、同一HF checkpoint路径、
Qwen3-30B-A3B BF16、同一8×A100-SXM4-80GB/ai_hgx_03/Slurm1031869/SSH extern。
TP8/DP2/attention CP2/EP4/MoE DP2/dense TP1、Triton attention/MoE、DeepEP normal。
capture buckets2/4、固定接收容量16，Q/K联合和router tile候选关闭。
每Engine额外3prompt/2token的小bucket预热后，3标准预热+20正式。
固定8prompts/1218输入tokens/每请求128输出，共1024输出tokens，temperature0.6、ignore_eos。

三个模式均返回实际routes，计时包括该回传成本；验证、NPZ保存、差异分析和
控制审计在generate区间外。时间包括调度/prefill/decode/sampling/logprob/结果返回，
排除加载、捕获、预热、cache flush、训练和权重同步。不能拿P9无route回传的
1.890481秒与此表直接比较，称为同条件性能回归。

## 三轮完整结果

| 轮次/顺序 | A P50/P95秒 | B P50/P95秒 | C P50/P95秒 | B/A变化 | C/B额外开销 | C/A额外开销 |
|---|---|---|---|---:|---:|---:|
| 1：ABC | 2.019609 / 2.061242 | 1.994497 / 2.069718 | 3.391734 / 3.538088 | −1.243% | +70.055% | +67.940% |
| 2：BCA | 2.022057 / 2.131250 | 1.990809 / 2.004468 | 3.383158 / 3.417857 | −1.545% | +69.939% | +67.313% |
| 3：CAB | 2.007811 / 2.128332 | 2.002917 / 2.108620 | 3.389224 / 3.521910 | −0.244% | +69.214% | +68.802% |

每模式3次独立启动，各60正式批；不是每模式60次独立Engine。
各Engine的20批用于P50/P95，独立启动数为3。以上比例先按同轮Engine中位数
计算，再汇总；没有用所有180批当独立重复，也没有删除较慢批次。

| 汇总（3个Engine中位数再取中位数） | 秒 | 输出tokens/s |
|---|---:|---:|
| A | 2.019609 | 507.03 |
| B | 1.994497 | 513.41 |
| C | 3.389224 | 302.13 |

跨启动配对比例中位数：B/A −1.2434095%，C/B +69.9388980%，C/A +67.9401360%。
B/A的小幅效果只有3独立重复，作为本负载观测，不宣称统计显著或全局最优通信参数。

## 配置轴确实生效

A默认也为20 SM，实际区别还包括分块容量：

| 实际传入 | SM | NVL send/recv | RDMA send/recv |
|---|---:|---|---|
| A dispatch | 20 | 28 / 720 | 12 / 128 |
| A combine | 20 | 2 / 720 | 8 / 128 |
| B/C dispatch与combine | 20 | 9 / 256 | 6 / 128 |

通过Config真实构造参数和实际通信调用对象审计，未只看CLI字符串或猜opaque对象。
B/C相同配置在各rank、各启动均核验；C还改变matmul、attention、sampling、
all-reduce等执行方式，所以C/B不是“纯GEMM开销”。

历史eager的约10%剩余开销与此处约70%使用不同执行方式及分母。Full Graph使
baseline也显著加速；应比较这里的同轮Full/Full，不能将旧比例当常数。当前
B→C的中位差约1.395秒，但未完成P11前不将其分解给任何特定kernel或CPU阶段。

## 正确性与执行验证

配置/数值门禁来自本次独立五Engine，详见
[P10门禁报告](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/P10_FULL_GRAPH_AB_GATE_20261012.md)。
B/C full对各自eager的四case tokens/routes/logprob bits全部相同；C跨batch门禁通过。
A/B有限greedy对照也相同。门禁不是180个随机生成批次之间的bitwise等价承诺。

正式测量的九Engine均有：8rank×bucket2/4的全部48层dispatch/expert捕获证明；
每rank正式区间2560次 `FullCudaGraphBackend.replay`，纯全局decode eager回退0。
九Engine共184320个跨rank回放调用计数，该总数不是性能时间可相加的依据。
DP局部decode发生在全局prefill时允许eager，原始计数和is_extend_in_batch /
can_run_dp_cuda_graph元数据明确保存，不冒称整个请求或每个局部调用都进入图。

九worker PID唯一、退出码0；207标准批（含27预热）全部零缓存、固定1024输出、
有限logprob；源码SHA不变、git commit相同、模型路径/拓扑/dtype/seed一致。
每个实际tokens/routes NPZ都重读，shape/身份/hash核验，首次回放证据在正式测量前写完。
独立核验重新计算九组P50/P95、三轮比例和实际workload差异。

## 实际生成差异

以下是三轮共60对正式batch，每对8请求/128输出，480请求对，61440输出token位置。
路由只在双方历史输入前缀相同的位置比较，分开报告ordered IDs与expert集合。

| 比较 | 输出token不同位置 | 完全相同response请求 | 完全相同route数组请求 | 共同输入token×layer行 | ordered ID不同数 | expert集合不同的行 |
|---|---:|---:|---:|---:|---:|---:|
| A/B | 1549 / 61440 | 454 / 480 | 447 / 480 | 6357216 | 396006 | 52527 |
| B/C | 51623 / 61440 | 0 / 480 | 0 / 480 | 3901344 | 24509810 | 766915 |

输入prompt和长度相同不代表每步生成token/专家负载相同。B/C含不同sampling路径，
不能把输出差异全部归到数值误差或某个算子；ordered ID差异也不等于expert集合差异。
本试验评估自然随机生成配置下的方案性能，不是固定所有GPU操作负载的因果消融。
后续P11/P12用真实shape和独立受控算子验证解释剩余成本。

## 工具、证据和交接

- `gpu_tools/run_full_graph_deterministic_ab.py`：门禁/三轮顺序和进程守护。
- `gpu_tools/full_graph_ab_*`：显式配置、广播审计、真实workload、独立Engine基准。
- `gpu_tools/validate_full_graph_ab_run.py`：全部实际记录的独立完整性核验。
- `gpu_runs/p10-full-ab-gate-hgx03-20261012-04/`：有限数值/容量/配置门禁。
- `gpu_runs/p10-full-ab-timing-hgx03-20261012-01/`：results.json、measurement-validation.json、
  每组summary/measurements/NPZ、runtime记录、源码快照、启动和退出状态。

测量driver113286及全部worker已退出，Slurm1031869仍保留，下一步进入P11。
本报告与独立验证工具按用户要求保存main；raw运行数据保留忽略目录。
P10完成不表示P11–P14完成；整体目标继续有效，在线更新生命周期尚未验收。
