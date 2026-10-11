# P10 Full Graph A/B/C：配置、数值与执行门禁

门禁已完成，性能三轮测量尚未开始。本文件记录实现和有限正确性证据，不发布
确定性开销百分比。P10完整性能报告在9个独立Engine/180正式批完成后补充。

## 三模式和实际通信参数

| 模式 | 数值配置 | DeepEP选择 | decode/prefill |
|---|---|---|---|
| A | 非确定性、原生norm | 默认配置 | full/eager |
| B | 非确定性、原生norm | 显式固定配置 | full/eager |
| C | 确定性、unified decode、残差RMS候选 | 与B相同固定配置 | full/eager |

均为Qwen3-30B-A3B BF16、TP8/DP2/CP2/EP4/MoE DP2、8×A100、接收容量16，
capture buckets2/4。Q/K联合、router小tile候选均关闭。

已安装deep_ep_cpp.Config没有公开字段getter，因此在实验hook中观测真实
Config构造参数和实际传入dispatch/combine的对象，未根据不透明repr推测：

| 路径 | SM | NVL send | NVL recv | RDMA send | RDMA recv |
|---|---:|---:|---:|---:|---:|
| A dispatch默认 | 20 | 28 | 720 | 12 | 128 |
| A combine默认 | 20 | 2 | 720 | 8 | 128 |
| B/C dispatch固定 | 20 | 9 | 256 | 6 | 128 |
| B/C combine固定 | 20 | 9 | 256 | 6 | 128 |

A也是20 SM，A→B比较的是完整通信配置选择，不能只称SM数变化。

## 已完成门禁

完整run：`gpu_runs/p10-full-ab-gate-hgx03-20261012-04/`。
Slurm1031869、ai_hgx_03、SSH step_extern。五个独立Engine按
A-full、B-eager-shadow、B-full、C-eager-shadow、C-full执行。
数值比较统一prefill_max_requests=1，正式计时将恢复自然调度。
四组greedy输入为batch8/batch3/batch1/variable3，每请求16输出tokens。

- A/B/C各有8 ranks×bucket2/4的48层dispatch/expert捕获及原生full回放证据。
- 每个rank各64次实际full回放，三个模式的纯全局decode eager回退均为0。
- B的full对本轮B-eager：tokens/routes/logprob bits全部相同。
- C的full对本轮C-eager：tokens/routes/logprob bits全部相同，batch8对batch3/1门禁通过。
- A与B四case的描述性比较也没有差异；不将此有限greedy结果当作随机生成永远相同。
- B固定容量有效行shadow：21504次、171687936 BF16元素、1085零接收，零bit差异。
- C固定容量有效行shadow：21504次、206491648 BF16元素、343零接收，零bit差异。
- 所有worker正常退出，driver90456结束，源码SHA保持不变，GPU释放。
- 18项CPU检查通过，覆盖通信配置轴、实际token/route覆盖、signed-zero logprob bits、
  非法观测拒绝、共同输入前缀路由差异和控制消息编码。

### 局部decode与全局prefill

DP异步阶段中，一个组正在prefill，另一组可有局部decode。全局
is_extend_in_batch=True、can_run_dp_cuda_graph=False时，该阶段按prefill eager执行。
新增监控保留原始局部decode计数和元数据，将其与纯全局decode回退分别报告。
A/B各有4个DP1 rank×2次此类调用，C有4个rank×1次；没有隐藏这些事件。
不宣称整个请求或每个局部decode调用都是full图；纯全局decode的完整forward为图内。

## 实现与计时协议

`full_graph_ab_experiment.py`明确保留B/C的固定DeepEP config；A删除显式config。
`full_graph_ab_metrics.py`计数full回放、eager阶段和实际通信参数。
统计reset/flush经Tokenizer控制广播在计时外执行，不通过可能丢失DP1控制消息的
private collective_rpc DEALER通道。特殊数字命令只在本实验hook内识别，不更新真实ServerArgs。
广播有60秒超时，所有rank的最终文件必须齐全；回放中没有文件I/O或tensor shadow。

计时入口：`run_full_graph_deterministic_ab.py --phase timing --gate-run <本门禁目录>`。
每模式3独立Engine，顺序ABC/BCA/CAB；各3预热+20正式，8prompts/1218输入/
每请求128输出，temperature0.6，prefill eager、buckets2/4、cap16。
所有模式都返回真实routes并记录每批实际tokens/routes。route回传计入generate，
NPZ保存和差异分析在计时外；该观察协议不能与P9无routes返回的绝对时间直接混用。
差异分析分别报告ordered IDs与expert集合，且只在共同历史输入前缀比较路由。

源码和门禁通过后先git保存，再开始三轮计时，保证所有正式Engine使用同一提交。
本阶段未验证跨框架训练、权重更新、sleep/wake或其它拓扑；分别属于后续P13/P14。

## 早期运行与资源中断

run01被局部/全局阶段未区分的observer检查拦下；逐元素计算没有报告失败。
run02因private RPC发送给DP1但控制仅从TP0广播而阻塞，已停止仅本任务进程组。
run03初始化期间Slurm1031236被管理员取消（并非两天时限到达），旧PID已确认消失。
按已授权8GPU/64CPU/1TiB/两天规格申请1031869后，run04完成全部门禁。
以上原始失败/中断记录保留忽略目录，不加入main的失败探针。

验收文件：`gate-complete.json`、`compare-B.json`、`compare-C.json`、
`A-B-greedy-differences.json`、各模式`graph-evidence/runtime-gate-rank-*.json`、
`source-snapshots/`、`*-exit.json`。任务总范围仍为P10–P14全部完成。
