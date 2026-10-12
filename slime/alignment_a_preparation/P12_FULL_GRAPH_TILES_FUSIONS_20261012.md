# P12 Full Graph 小tile与融合消融

**P12已完成：选择QKV/O小tile + residual RMS融合 + Q/K联合norm/layout +
FP32 Router小tile。** 全部数值门禁通过；三次独立Engine配对，完整生成中位
延迟下降的配对中位数为 **19.93%**。Split-K2/4均有bit mismatch，拒绝进入
严格alignment；其实现保留本地，不提交main。
这是推理有限回归与性能结果，不代表P13真实在线更新或P14容量泛化已经完成。

## 配置与范围

同Qwen3-30B-A3B BF16、8×A100、TP8/DP2/CP2/EP4/MoE DP2、dense TP1，
DeepEP normal固定20SM通信配置、接收容量16、decode full buckets `[2,4]`，
prefill eager、Triton attention/MoE、alignment数值路径。
与P10相同8prompt、1218输入、128输出/请求、1024输出/批、temperature0.6、
ignore_eos。实际route返回包含在generate计时内；load/capture、预热、cache flush、
RPC审计和NPZ验证/写盘在计时之外。没有profiler、NVTX或CUDA-event组件计时插桩。

测量运行 `gpu_runs/p12-full-timing-hgx03-20261012-01/`：
7个消融方案各独立启动1次，3批预热+20批正式；初轮选择最快方案后，再做
2轮reference/胜出方案独立配对，第二轮倒序。共11个Engine、253批实际workload、
220批正式数据。其它消融仅有单轮pilot，不把它们描述成3轮独立复现。
所有worker源commit为 `52196c0b`，源码SHA、checkpoint、拓扑及共用配置一致。

## 数值门禁

实际原始矩阵来自P11逐bucket即时快照：rank0/4、第0/47层、两个bucket、
QKV/O/FP32 Router24份，112,128个输出元素。原kernel逐bit与批次前缀门禁均通过。
小tile在同24份原始数据上，ordinary与CUDA Graph输出逐bit差异0，
适用的M前缀1/2/4差异0。不是仅对随机矩阵或容差的检查。

模型门禁：
`gpu_runs/p12-model-gates-hgx03-20261012-01/`（reference/dense），
`gpu_runs/p12-model-gates-hgx03-20261012-02/`（其余5方案）。
每个方案独立Engine，用batch8、batch3、batch1、variable3四组有限greedy请求，
16输出/请求；token、实际路由、FP32 logprob bits与reference一致，batch前缀一致。
8rank各64次原生FullBackend回放；纯global decode eager回退0。
两个bucket、8rank各48层QKV/O候选捕获覆盖通过。RMS on捕获95个候选调用/bucket；
RMS off为0；Q/K on捕获48个调用；Router on捕获48层各1个调用。
不是只设置环境变量就宣称候选进了graph。

## 候选实现

QKV/O只接受指定module、BF16、M1/2/4、K2048与N2048/2560，无bias。
原始 BM128/BN128/BK64/8warps 改为 **BM16/BN64/BK64/4warps**，
GROUP_M8、3stages、K循环和tl.dot累加顺序保留；不支持的调用执行原路径。
属于opt-in实验hook，不改变通用matmul注册或生产默认值。

FP32 Router复用已经通过有限eager门禁的BM16/BN64/BK32/4warps，
新增显式full实验adapter，在无audit/文件I/O/GPU读回的条件下允许捕获。
原公共Router实验入口仍保留其graph禁用保护；不通过开启旧audit路径来捕获。
Router实际M为1/2（bucket2/4，CP分片后），不是QKV/O的M2/4。

Q/K联合norm/layout复用 `qk_runtime_hooks` 的joint模式，allow_cuda_graph=True，
没有修改Rope或attention归约。RMS使用已验证candidate，进行同full条件下off/on消融。

## 完整生成结果

首轮各20批正式数据：

| 方案 | 中位数 s | P95 s |
|---|---:|---:|
| Reference（原QKV/O，RMS on） | 3.390534 | 3.402511 |
| QKV/O小tile，RMS on | 3.075585 | 3.124919 |
| 原QKV/O，RMS off | 3.666562 | 3.809668 |
| QKV/O小tile，RMS off | 3.342247 | 3.382743 |
| 小tile + Q/K，RMS on | 3.050546 | 3.193955 |
| 小tile + Router，RMS on | 2.751210 | 2.889395 |
| 小tile + Q/K + Router，RMS on | **2.715794** | **2.852725** |

QKV/O小tile初轮约−9.29%；Router叠加收益明显。
Q/K的边际差异较小，不能用单轮差距宣称稳定独立收益。
关闭RMS融合在两种QKV/O路径均变慢，因此最终保留RMS on。
该消融不支持将所有剩余alignment开销归因于RMSNorm。

胜出方案的独立确认：

| 独立轮次 | Reference中位 s | 优化中位 s | 配对变化 |
|---|---:|---:|---:|
| 1（pilot） | 3.390534 | 2.715794 | −19.9007% |
| 2（优化先跑） | 3.386710 | 2.701414 | −20.2348% |
| 3（reference先跑） | 3.389239 | 2.713707 | −19.9317% |

配对变化中位 **−19.9317%**；优化Engine中位数的中位约2.713707s，
对应1024/2.713707约377.3输出tokens/s。不是端到端在线RL加速或所有硬件的保证。
不能与P9的无route返回baseline绝对数字混算新确定性开销。

## 条件性Split-K探索

P11实际来源归因确认GEMM为热点；小tile QKV/O仍只有32/40个CTA，A100有108SM。
据此进行独立离线探索，而不是先改变严格数值路径。
运行 `gpu_runs/p12-splitk-exploration-hgx03-20261012-01/`，16份实际BF16
QKV/O矩阵，M2/4、K2048、N2048/2560，保持BM16/BN64/BK64，分K2/4后
输出FP32 partial并固定顺序合并。改变了FP32归约顺序。

| Split | 检查元素 | bit mismatch | 有差异矩阵数 | 严格alignment |
|---|---:|---:|---:|---|
| 1（小tile参考） | 110592 | 0 | 0/16 | 有限门禁通过 |
| 2 | 110592 | 155 | 16/16 | 拒绝 |
| 4 | 110592 | 159 | 16/16 | 拒绝 |

例如某M4 QKV小tile约12.78µs，Split2约9.67µs，Split4约8.20µs。
这些是同权重warm-cache、100组kernel装进单张graph后区间/100的单GPU结果，
包含partial/reduce和graph节点开销，不包含逐kernel Python发射循环。
更快不抵消bit mismatch；未做拒绝候选的完整模型集成或实际上线。

## 独立测量审核

`measurement-validation.json`证明11个不同worker PID、同一commit，
253份NPZ实际token/route/logprob shape与SHA、220批正式中位/P95重算、
所有cache token0、每rank正式2560次native full回放、纯decode eager回退0，
候选捕获与三轮配对结果重算一致。
同时记录模式间实际自然生成的token/route差异；随机生成负载不保证相同，
不将完整generate差异直接解释成纯GEMM因果代价。

本次8组比较（首轮6个消融、确认轮2个优化/reference）各20批，实际生成
token和路由均完全相同。另逐bit比较每组20,480个目标logprob，共163,840个，
全部差异0；记录在 `timed-logprob-bit-comparison.json`。这些是本次观测，
不替代对任意未来输入或更新权重的回归。

这是有限greedy模型/批次门禁与实际随机生成性能验收。
P13还必须完成更新权重后的3×1024目标token严格回归、旧图/新图、sleep/wake
和storage替换；P14还必须完成容量、显存、多batch与回退边界，不能由本文替代。

## 工具与复跑

`p12_experiment.py`定义7种轴；`dense_tile_hooks.py`与`router_full_tile_hooks.py`
提供显式候选；`full_fusion_proof.py`保存48层capture/replay证明。
`p12_model_gate.py`、`run_p12_model_gates.py`复用token/routes/LP-bit模型比较。
`p12_benchmark.py`复用P10计时边界；`run_p12_timing.py`先检查所有门禁再计时。
`projection_tile_probe.py`检查实际矩阵，`validate_p12_timing.py`独立审核全部记录。

```bash
python gpu_tools/run_p12_model_gates.py --run-id UNIQUE_GATE \
  --variants reference dense rms-off dense-rms-off dense-qk dense-router dense-qk-router
python gpu_tools/run_p12_timing.py --run-id UNIQUE_TIMING \
  --gate1 gpu_runs/UNIQUE_GATE --gate2 gpu_runs/UNIQUE_GATE
python gpu_tools/validate_p12_timing.py --run gpu_runs/UNIQUE_TIMING
```

在已分配ai_hgx_03、相同slime环境执行，使用唯一run ID。
成功runtime/collector代码以验证过的字节纳入Git；raw权重、NPZ、运行日志、
失败/拒绝的Split-K源码留忽略目录。P13/P14临时实现尚未提交。
