# P10–P14 逐项验收与证据索引

**P10–P14 的实际实验均已通过下述范围验收。** P11 prefill补采也已完成并由
发布版解析器验证。NCU仅因用户明确取消而免于补采，不称硬件计数器采集成功。
最后的Git提交/推送与GPU交接另以实际命令核验，不能由本清单文字代替。

需求来源为用户的P10–P14目标、附件 `pasted-text-1.txt`、
[执行清单](P10_P14_EXPERIMENT_PLAN_20261012.md)。用户2026-10-12明确取消
剩余Nsight Compute硬件计数器采集；取消项不记为测量成功。用户同时要求直接
提交main、失败实现不进入main、整理整个alignment_a_preparation目录。

## P10：同轮Full Graph A/B/C

| 验收项 | 实际证明 | 状态 |
|---|---|---|
| A默认非确定性、B固定配置非确定性、C同B配置确定性 | 三模式resolved args与实际dispatch/combine Config记录；A/B数值路径与C分别启用 | 通过 |
| B不能删除显式DeepEP配置 | B全部rank真实20SM/9/256/6/128；B eager shadow与full模型门禁 | 通过 |
| 同commit/checkpoint/topology、full decode2/4、cap16、prefill eager | 九worker manifest同68b8fef3、冻结数据SHA一致；TP8/DP2/CP2/EP4/MoE DP2，resolved args一致 | 通过 |
| 8prompts/1218输入/128输出每请求/temperature0.6 | 207标准批实际NPZ身份与shape检查；每批1024输出、cache tokens0 | 通过 |
| 3轮顺序轮换、每Engine3warm+20formal | ABC/BCA/CAB，9个唯一PID，180正式批；2026-10-12再次独立复算通过 | 通过 |
| 真实full捕获/回放，无纯decode eager回退 | 各rank两bucket48层证明；正式每rank2560次native replay，纯全局decode回退0；全局prefill期间局部decode另记 | 通过 |
| 实际tokens/routes差异与公平口径 | 所有NPZ校验token/route SHA；自然生成差异保留，route回传计时内、写盘计时外 | 通过 |
| P50/P95及跨启动配对 | 原记录重算；B/A −1.243%、C/B +69.939%配对中位数；不归因为单个norm/GEMM | 通过 |

证据运行 `gpu_runs/p10-full-ab-gate-hgx03-20261012-04`、
`gpu_runs/p10-full-ab-timing-hgx03-20261012-01`；后者
`measurement-validation.json` 与 `validate_full_graph_ab_run.py` 再次运行通过。
说明见 [P10报告](P10_FULL_GRAPH_DETERMINISTIC_AB_20261012.md) 与
[模型门禁](P10_FULL_GRAPH_AB_GATE_20261012.md)。成功提交68b8fef3/ff82e68d。

## P11：实际CPU/GPU归因

| 验收项 | 实际证明 | 状态 |
|---|---|---|
| A/C相同32decode steps，rank0/4 | 8rank记录32replays，4个原始Nsight报告/SQLite；rank0/4各32原生Graph launches | 通过 |
| GEMM/attention/norm/QK/DeepEP/experts来源 | GPU graphNode按process隔离，clone链+同线程内层NVTX来源；图内未归因kernel0 | 通过 |
| GPU活动并集、重叠、等待、host envelope | union/duration sum/envelope分开；CPU API膨胀和异步通信current-stream event限制已说明 | 通过 |
| Graph外sampling/metadata | 选定窗口实际D2H/API及source范围保留，不能重解释为每层KV `.item()` | 通过 |
| Graph外实际prefill独立窗口与来源 | A/C各8rank2次prefill和1次附带native replay，rank0/4 runtime/driver+NVTX归因，prefill/局部decode/idle/采样分别记录 | 通过 |
| 真实shape/raw GEMM reference | 24packet、112128输出元素、batch前缀逐bit0；两bucket即时快照避免inactive pool与O输出inplace问题 | 通过 |
| 选定NCU硬件计数器 | 实际ERR_NVGPUCTRPERM；用户明确取消补采，不宣称occupancy/SM利用率已测 | 用户取消 |
| 普通性能结论来自无profiler | 使用P10/P12，不能用profiler中cudaGraphLaunch时间替代正常性能 | 通过 |

证据：`gpu_runs/p11-full-profile-nsys-hgx03-20261012-04` 的timeline/source summary，
`gpu_runs/p11-raw-projection-gate-hgx03-20261012-05/results.json`；
[P11报告](P11_FULL_GRAPH_PROFILE_20261012.md)、[raw门禁](P11_RAW_PROJECTION_GATE_20261012.md)。
prefill补采 `p11-outer-prefill-hgx03-20261012-03` 两worker exit0、8rank报告齐全，
source SHA不变、1218实际输入/8targets及route门禁通过，发布版SQLite解析器
再次归因通过。见 [图外补充报告](P11_PREFILL_OUTSIDE_GRAPH_20261012.md)。
此前补采的过严no-decode假设和rank报告落盘/关闭竞争均保留失败证据，不发布main。

## P12：小tile、融合消融和条件性Split-K

| 验收项 | 实际证明 | 状态 |
|---|---|---|
| QKV/O BM16/BN64/BK64保留K累加顺序 | 成功runtime实现；24真实操作数ordinary/graph/前缀gate逐bit0 | 通过 |
| 完整模型与candidate实际进入捕获图 | 七轴完整模型门禁；各rank两bucketQKV/O各48层，小tile/fusion计数齐全 | 通过 |
| RMS off/on，Q/K叠加 | 两种投影路径RMS on/off均实测；Q/K48calls、RMS95/0；关闭RMS变慢，最终保留on | 通过 |
| 仅低并行度证据后做Split-K2/4 | P11 GEMM hotspot及grid/108SM证据；16真实矩阵保持BM/BN/BK；split2/4各155/159bit mismatch | 通过，拒绝候选 |
| bit mismatch候选不能进入严格alignment/main | split实现仅忽略目录，成功runtime不含split候选 | 通过 |
| 无profiler完整generate加速与独立重复 | 11唯一Engine、253实际批/220正式；胜出方案三轮配对−19.9007/−20.2348/−19.9317%，中位−19.9317% | 通过 |
| 数值与workload审计 | 160批对tokens/routes一致，163840 target LP bits0；2026-10-12独立测量验收器再次复算通过 | 通过 |

证据：`p12-raw-small-tile-probe-hgx03-20261012-01`、
`p12-model-gates-hgx03-20261012-01/02`、`p12-full-timing-hgx03-20261012-01`、
`p12-splitk-exploration-hgx03-20261012-01`（均在gpu_runs）。
详见 [P12报告](P12_FULL_GRAPH_TILES_FUSIONS_20261012.md)，成功提交9419498a。

## P13：真实在线更新与Graph生命周期

| 验收项 | 实际证明 | 状态 |
|---|---|---|
| v89实际路径、两轮非零更新 | 真生成/训练/原optimizer.step；8rank前两轮global grad10.8996/11.3917、各rank抽样CPU master变化>0 | 通过 |
| 明确实际optimizer与reward | StatelessAdam每步reset；诊断0/1组内reward确保非零advantage，未称原随机reward/学习效果或标准持久Adam | 通过 |
| 全量权重同步实际完成 | 实际colocate tensor路径，训练NCCL收集后Gloo/IPC；每恢复50成功bucket/全部rank，不凭CLI称纯NCCL广播 | 通过 |
| v0/v1/v2各1024targets bitwise LP | 真实rollout/train完整token序列、位置和有限LP检查，三个版本均0int32 bit mismatch | 通过 |
| 全模型weight/KV/request地址与版本 | 每rank435 weights含LMhead、96KV及req_to_token；地址digest与最终两bucket快照一致 | 通过 |
| 原地更新后旧图读新值 | 更新后replay仍用旧graph object/pool ID；v1/v2输出改变，随后重捕获同权重输出逐bit一致 | 通过 |
| sleep/wake、offload/onload、恢复顺序 | 实际release/分阶段resume，恢复50bucket后KV wake，地址无变化，旧图正常回放；不称partial update恢复全模型 | 通过 |
| 地址替换失效负例与恢复 | 各rank真实clone layer0 norm存储，同一replay指针门禁拒绝，重建后输出恢复；每rank10bucket captures | 通过 |
| 不能用synthetic替代真实训练 | 隔离tensor smoke只补机制；验收依据实际online06三版本数据 | 通过 |

证据：`gpu_runs/p13-full-online-hgx03-20261012-06` 的actual rollout/train、
update witnesses、lifecycle、三版本greedy比较、bit gate和独立online audit。
发布版独立验收器重新审核通过，13项CPU机制检查通过，实际argv/hooks与发布
路径适配后配置一致。详见 [P13报告](P13_ONLINE_FULL_GRAPH_LIFECYCLE_20261012.md)。
成功提交147a09a5，失败内存/零梯度/路由注册尝试没有被追认为成功。

## P14：容量、显存、多batch与边界

| 验收项 | 实际证明 | 状态 |
|---|---|---|
| cap16保持EP4×≤4，cap8须全rank≤2 | globalmaxRunning8→DP2每scheduler4→CP2/buckets2/4物理send≤2；实际8rank×2bucket48层source rows验证 | 通过 |
| 超界不能进入unsafe通信 | 每rank实际cap16 5行/cap8 3行GPU输入经同一dispatch入口拒绝；6项CPU拓扑/上界检查 | 通过 |
| 多batch/变长/多response/repeat prompt | 每容量10case/53requests/837targets，batch1/2/3/4/7/8/12，输入实际17/63/129/137，输出1/4/16/32 | 通过 |
| cap8/16与eager/full数值一致 | 逐请求tokens/LPbits/已验证route哈希与shape相同；每cap67584expert shadow、687022080元素bit0 | 通过 |
| 零接收/idle及路由偏斜 | 8rank×5受控normal dispatch/combine，真实receive counts0/2/6/8、idle source0，两cap输出bit0 | 通过 |
| Torch allocated/reserved与总驻留 | 每rank模型/pool设置后reset峰值，NVML 1s采样；两者范围明确，cap8allocated只省约0.01MiB | 通过 |
| 捕获与初始化成本 | 两bucket成本、graph/pool数与Engine构造wall齐全；冷编译/执行顺序不能误称cap8加速 | 通过 |
| 实际full回放、回退/边界 | 0纯decode eager回退，prefill eager；12请求排队到允许shape，超界拒绝，不承诺任意并发自动扩图 | 通过 |
| 权重/graph失效与泛化边界 | P13实际重建/地址负例；单节点EP4/Qwen3 BF16范围，不外推跨节点/量化/其他拓扑 | 通过 |

三运行：`p14-capacity-model-hgx03-20261012-01`、
`p14-controlled-routing-hgx03-20261012-01`、`p14-capacity-shadow-hgx03-20261012-01`。
发布版 `validate_capacity.py` 对三运行独立复查通过。
详见 [P14报告](P14_CAPACITY_MEMORY_BOUNDARIES_20261012.md)，成功提交a555e824。

## 交付与运行状态

- 每阶段报告、配置、manifest、source SHA、数值/测量证明已保留；未测指标有说明。
- 成功源码直接main；raw模型/NPZ/trace、失败探针和本机状态留忽略目录，不force-add。
- [目录总览](WORK_SUMMARY_20261012.md)覆盖历史训练/推理、optimizer、RoutePack/
  AutoPack、P1–P9、P10–P14，README指向本清单与阶段报告；不是只列近期成功数字。
- 已完成阶段CPU与GPU证据范围分开。跨条件历史百分比不混作当前同轮结论。
- 原分配1031869被管理员取消；补采使用新分配1032144且任务已完成、GPU compute
  列表为空。不恢复用户已取消的保号，不停止任何无关任务，不用旧PID替代存活检查。

实验与数值/性能/归因门禁已完成；交付需确认这些文档和成功源码已提交并推送main，
工作区无未处理改动，运行交接准确。失败源码和大型原始数据继续留在忽略目录。
