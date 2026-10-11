# P10–P14 执行要求与验收清单

用户要求五阶段依次全部完成，重要实现验证后提交并推送main。原要求来自
`/public/home/yingxy2024/.codex/attachments/8d3813b0-b7d2-4b83-b46f-f4c56bf37e33/pasted-text-1.txt`。
以当前main和运行证据为准，不依赖原文本中过时的分支位置。五阶段未全部验收前，
不将整体目标标为完成。继续沿用用户取消保号的偏好；仅操作本任务进程。

## P10 公平 Full Graph A/B/C

- A：非确定性，baseline默认DeepEP配置。
- B：非确定性，显式固定20 SM配置，先验证实际使用和完整模型可执行性。
- C：确定性alignment，使用与B相同的显式20 SM配置；沿用已验证残差RMS融合。
- 同一git commit/checkpoint/8×A100/TP8/DP2/CP2/EP4/MoE DP2，full decode，
  buckets2/4，接收容量16，prefill eager；8冻结prompts/1218输入/每请求128输出，temperature0.6。
- 至少3轮，每轮模式顺序轮换；9独立Engine，各3标准预热+20正式，共180正式批。
- 先做B配置实际通信参数审计、固定容量shadow及各模式full/eager有限数值检查。
- 各rank完整捕获/回放证明；正式区间计数检查无decode eager回退，prefill允许eager。
- 所有模式统一启用真实route返回，以保存各正式batch实际tokens和路由；路由回传
  成本包含在generate中，文件保存/差异分析在计时外。不得混用无route观测的P9绝对时间。
- 报告每Engine P50/P95、跨独立启动配对A→B与B→C、tokens/routes差异和全部源码/配置证据。
- C包含matmul/attention/sampling/all-reduce等整体变化，差值不解释成单个算子成本。

## P11 Full Graph CPU/GPU Profile

- A与C用相同16或32个decode steps的短窗口，保存rank0/4时间线和实际full回放证明。
- 优先Nsight Systems，确认工具版本、graph节点跟踪与采样边界；选主要kernel做Nsight Compute。
- 覆盖persistent GEMM、attention、norm/QK、DeepEP/experts、图外metadata/sampling/prefill。
- 同时报告GPU活动并集、重叠、依赖/等待及host窗口；duration sum不等于generate关键路径。
- 使用实际shape/来源关联，不能按一个kernel名猜测全部调用归属。性能数字来自无profiler运行。

## P12 候选完整模型与Full Graph验证

- 首先QKV/O小tile BM16/BN64/BK64，保留K归约与累加顺序；真实张量、模型、batch及graph门禁。
- 同条件full下RMS off/on，并检查Q/K叠加是否有额外收益；候选必须实际进入捕获图。
- 仅P11/小tile数据表明有低并行度瓶颈时做Split-K2/4；bit mismatch候选不得进入严格alignment。
- 通过门禁的候选用完整generate对照验收，记录未获收益/未通过候选并保留本地失败证据。
- 不预设每个候选必然加速；阶段验收要求有经验证的最终选择和实际性能证据。

## P13 真实在线更新后的Graph生命周期

- 将成功full方案接回v89实际在线路径，验证两轮非零更新及全量权重同步后目标logprob严格一致。
- v89实际使用StatelessAdam/reset states，明确说明，不冒称标准有状态Adam长期验证。
- 记录全模型weight storage地址/版本；原地更新后旧图必须读取新值。
- 用更新后replay与重新捕获参考比较，验证sleep/wake、weights/KV offload/onload和graph pool。
- 若地址替换，必须失效/重建而非重放旧指针；包含可检测的负例和恢复检查。
- 每轮1024 target tokens bitwise logprob gate、非零gradient、实际同步完成和graph回放证据。
- 不用synthetic tensor更新替代两轮真实Megatron/Slime更新；隔离张量测试仅补机制证据。

## P14 容量、显存、多batch与边界

- 保留cap16的EP4×每发送rank≤4上界。cap8只有在所有rank发送上界≤2得到证明及guard后才试。
- 多请求batch、变长输入和不同路由，包括idle/零接收、bucket覆盖、允许范围外行为。
- 报告正确性、PyTorch峰值allocated/reserved及可用总驻留显存观测，区分统计范围。
- 记录捕获耗时/graph数/初始化成本、实际回放、容量guard、图/权重失效与fallback行为。
- 不外推到未验证的跨节点、量化、其它模型/拓扑；边界要有实际拒绝或安全回退证据。

## 交付与证据要求

每阶段独立报告、配置、入口、source SHA、manifest、gate与测量/采样结果。
成功源码和完整报告按阶段git；raw运行数据留在忽略目录，失败探针不发布main。
最终更新WORK_SUMMARY和README，并逐项检查P10–P14完整证据后才宣布全部完成。
