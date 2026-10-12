# P11 Full Decode Graph 的 CPU/GPU 归因

**A/C Nsight Systems 采集与来源关联完成；P11 按用户调整后的范围完成。**
选定 kernel 的 Nsight Compute 在当前账户下返回 `ERR_NVGPUCTRPERM`，
没有取得硬件计数器数据。不能把尝试执行 NCU 记成成功采集。
2026-10-12 用户明确表示“我没有这个nsight，那算了”，取消剩余 NCU
硬件计数器采集要求。以下时间线与来源归因仍有效；占用率等硬件指标保持未测状态。
P12–P14 仍待完成；本报告只提交经过验证的 Nsight Systems 工具与结果。

## 条件与采集边界

成功运行：`gpu_runs/p11-full-profile-nsys-hgx03-20261012-04/`。
8×A100，Qwen3-30B-A3B BF16，TP8/DP2/CP2/EP4/MoE DP2，
DeepEP normal，固定接收容量16，decode full buckets `[2,4]`，prefill eager。
A 为非确定性默认通信配置；C 为完整 alignment 和固定20SM通信配置。
C 的 residual RMS candidate 已开启，router tile candidate 关闭，QKV/O使用原始tile。
底层数值和 launch 实现没有为 profiler 更换。

两个独立 Engine，各用固定8prompt、3批128输出请求预热，然后选定32次
`FullCudaGraphBackend.replay`。保留实际256输出token、logprob与专家路由。
A/C自然生成的token/路由不保证相同，不能把时间差称为某一个kernel的纯因果代价。
性能对照仍引用P10的无profiler、3独立启动配对结果；本文所有时间均带采集开销。

Nsight Systems **2025.5.2.266**，CUDA/NVTX/OSRT，`cuda-graph-trace=node`，
不采CPU sampling/context switch，关闭CUDA event trace以免工具推断额外依赖。
使用两段cudaProfilerApi窗口：

1. 完整请求预热之后重新捕获两个bucket，取得Graph节点的代码来源。
2. 32个选定原生full回放及其间的metadata/sampling活动。

每个模式取得2份 `.nsys-rep` 和2份SQLite；8rank各32个选定回放和完整48层
组件记录，两个bucket的16份捕获/初始回放证明均通过校验。重新捕获窗口用于
来源关联；选定32步窗口使用bucket4，不声称该窗口还测试了bucket2。
8rank的最后一次选定回放保存source CUDA-event区间，rank0/4保存有限投影操作数。

首次 JIT kernel 载入/发射与 Nsight 采集开始的相对顺序曾导致崩溃。
最终先预热实际prefill、decode、sampling、logprob和route路径，再开启采集，
在完整A/C运行中通过。不修改模型kernel来绕过采集问题，不据此泛化所有崩溃根因。
显式固定与P10相同的NCCL2.28.9库。

## 时间线完整性与计算口径

rank0/4各32次原生 `cudaGraphLaunch_v10000`，用CPU correlation ID关联其GPU节点。
A每rank选定Graph kernel 57,088次；C为64,768次，**增加7,680次，即240次/回放**。
这些Graph kernel全部有捕获来源，未关联来源的Graph kernel为0。
少量Graph外GPU活动保留为unmapped，不猜测归属。

Nsight的Graph节点编号在不同进程复用；关联键必须包含进程标识。
捕获节点按同线程内最内层NVTX范围关联，克隆节点沿original node链追溯。
结果保留rank、layer、module path、实际input shape、kernel名和次数。
例如QKV `(4,2048)`、Q norm `(64,128)`、K norm `(8,128)`。

GPU忙碌时间合并同rank/device各stream上的kernel、memcpy、memset区间。
duration sum和并集之差为重叠的重复累计量，不是整个任务的关键路径。
host/GPU envelope从选定step与实际活动的最早开始至最晚结束；其中GPU非活跃
部分包含host处理、依赖和采集开销，不能全部称为CPU launch开销。

| 模式/rank | 窗口envelope ms | GPU活动并集 ms | activity duration sum ms | 重复累计 ms | GPU非活跃 ms |
|---|---:|---:|---:|---:|---:|
| A/0 | 765.127 | 456.066 | 457.880 | 1.815 | 309.061 |
| A/4 | 777.462 | 463.992 | 465.829 | 1.837 | 313.469 |
| C/0 | 1141.176 | 814.080 | 815.903 | 1.823 | 327.096 |
| C/4 | 1161.995 | 813.732 | 815.755 | 2.023 | 348.263 |

选定Graph的kernel envelope中位数：A rank0/4为20.761/21.115ms，
C为31.599/32.203ms。它不是无profiler生成延迟。
例如A rank0的32个cudaGraphLaunch累计CPU API时间234.325ms，
cudaEventSynchronize31次累计179.200ms；节点跟踪显著放大发射成本。
不能用这组API数字替代正常运行的CPU瓶颈证据。

## 逐组件结果

下表为该类别GPU kernel duration sum除以32；**ms/选定回放**，
按最内层来源分类，没有把父组件区间再次加入子组件。
各stream可能重叠，表中数值不相加冒充端到端关键路径。

| 来源 | A rank0 | C rank0 | A rank4 | C rank4 |
|---|---:|---:|---:|---:|
| Router | 0.3773 | 4.0848 | 0.3771 | 4.1164 |
| QKV | 0.7440 | 2.3155 | 0.7480 | 2.3184 |
| O projection | 0.6130 | 2.1277 | 0.6110 | 2.1447 |
| Attention core | 0.4785 | 1.6518 | 0.4783 | 1.6579 |
| Experts | 3.8338 | 5.0888 | 3.8470 | 5.1100 |
| QK/RoPE（排除内层norm） | 0.2452 | 0.5522 | 0.2457 | 0.5558 |
| Other norm（含Q/K norm） | 0.0084 | 0.2888 | 0.0084 | 0.2908 |
| Residual norm | 0.3932 | 0.4209 | 0.3927 | 0.4248 |
| DeepEP dispatch | 2.4633 | 2.5490 | 2.3451 | 2.5475 |
| DeepEP combine | 2.0142 | 1.9910 | 1.9863 | 2.0441 |

剩余差异有多个来源；本轮不支持“主要由batch-invariant RMSNorm导致”或
“额外launch基本都在DeepEP里”的结论。Router、QKV/O、attention和expert路径
都有明显kernel时间差；DeepEP kernel数量在A/C相同。
C Router还包含为Megatron严格对齐而使用的FP32输入/权重和路由路径变化，
A使用普通gate路径，差异不只是GEMM的tile。

rank0 Graph kernel数量变化可以直接关联到来源：other norm +3072，QK/RoPE
+1536，attention core +3072，router +1536，router select +1536；
QKV和O各减少1536，合计+7680。A QKV/O各是CUTLASS GEMM加split-K reduce，
C各为一个persistent GEMM；GEMM变慢并不意味着其kernel数量更多。

Graph内D2H memcpy为0。整个选定窗口rank0/4的Graph外D2H为A各124次、
C各155次；包含sampling/logprob/结果等活动，不能重新解释为每层KV `.item()`。
不声称消除了所有Graph外CPU/GPU同步。

## CUDA event区间与异步通信

最后一次选定model forward的inclusive event区间：A rank0/4约20.366/20.810ms，
C约31.815/31.773ms。它仅代表该回放，层间分布不等于独立实验分布。
Attention、MoE等父范围包含子范围，不能将它们相加。

Buffer.dispatch/combine的event标记在调用线程的current stream上。
DeepEP可在通信stream异步执行，因此方法周围的event区间不会完整包住
其它stream上的通信kernel。以actual Nsight kernel来源/时间线分析通信，
不能把dispatch约0.17ms的source-event总和解释成所有通信GPU时间。

## Nsight Compute 的当前缺口

选定QKV/O真实操作数上的原始persistent kernel实际尝试了SpeedOfLight+
LaunchStats、仅LaunchStats、仅launch指标及instruction-level source指标。
当前账户均返回 `ERR_NVGPUCTRPERM`；没有成功的NCU报告，也没有测得
achieved occupancy、SM利用率或memory throughput。用户已取消补采要求，
不再等待管理员权限处理；这些指标不得作为已有测量结果引用。
失败日志和临时NCU入口只保留本地，未作为成功实现提交。

Nsight Systems已有实际launch metadata，但它不替代硬件计数器。
rank0的每种persistent GEMM各1536次，实际参数如下：

| Kernel来源 | grid | block threads | registers/thread | dynamic shared bytes |
|---|---:|---:|---:|---:|
| QKV | 20×1×1 | 256 | 169 | 98304 |
| O | 16×1×1 | 256 | 169 | 98304 |
| FP32 Router | 1×1×1 | 256 | 170 | 81920 |

A100有108SM，尤其Router只有1个CTA；这证明launch规模很小，
但不等于测得achieved occupancy或全部性能损失的原因。
有限QKV standalone原kernel与保存输出逐bit一致；O保存输出可能被后续
inplace操作改写，不能作为raw GEMM逐bit参考。P12仍需原始输出快照及完整模型门禁。

原始输出快照已在独立运行中补齐：见
[P11原始投影门禁](P11_RAW_PROJECTION_GATE_20261012.md)。24份实际矩阵、
112,128个输出元素及批次前缀全部逐bit一致。原Nsight运行中保存的O模块
输出仍保持其限制，不能用新门禁追认旧数据为raw参考。

## 复跑与验证

```bash
python gpu_tools/run_full_graph_profile.py --phase nsys \
  --p10-run gpu_runs/p10-full-ab-timing-hgx03-20261012-01 \
  --run-id NEW_UNIQUE_RUN_ID --nsys-path /absolute/path/to/nsys
python gpu_tools/summarize_full_graph_components.py --run gpu_runs/NEW_UNIQUE_RUN_ID
python gpu_tools/summarize_full_graph_timeline.py --run gpu_runs/NEW_UNIQUE_RUN_ID
```

在ai_hgx_03、已分配8GPU和相同slime环境中执行。入口记录source SHA、
请求/解析配置、实际工作负载、采样窗口、资源/内存保护和退出状态。
raw `.nsys-rep`、SQLite、组件JSON和实际操作数位于忽略的gpu_runs目录。

CPU检查共10项：source event嵌套/完整性6项，活动并集、线程来源、克隆链、
进程隔离4项。实际A/C图内kernel来源完整性也经SQLite检验。
O 原始操作数参考已由上述独立门禁补齐；NCU 补采经用户取消。
P12 的实际完整模型门禁和无 profiler 计时见
[P12报告](P12_FULL_GRAPH_TILES_FUSIONS_20261012.md)。后续继续 P13–P14。
