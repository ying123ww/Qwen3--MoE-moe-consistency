# P14：容量、显存、多Batch与边界验收

**P14 完成。** cap16/cap8 的完整模型、eager padding shadow 和8rank受控路由
三组实际GPU运行均通过独立验收。cap8在严格发送上界内正确，但没有测得有意义
的总显存下降；本阶段没有无profiler重复计时，不能宣称cap8提升生成性能。

## 允许配置、上界与实际保护

Qwen3-30B-A3B BF16，8×A100/ai_hgx_03，TP8/DP2/attention CP2/EP4/MoE DP2，
DeepEP normal，单节点，decode native full、buckets2/4，prefill eager。采用P12
已验证的小tile+RMS+Q/K+Router组合，数学实现不另作修改。

cap16保留每EP发送rank物理行数≤4，最坏唯一接收token数≤4×4=16。
cap8仅允许每发送rank≤2，最坏≤4×2=8。这是全参与rank的保守上界，不是从
平均或观测路由分位数缩容量。每token最多被某目的rank接收一次，同一目的
rank上选择多个expert不会把唯一token数乘以top-k。

本实验全局 `max_running_requests=8`，SGLang DP2实际每scheduler最大4，
CP2与buckets2/4使每发送rank物理行数最多2。各rank、各bucket实际48层捕获
记录都检查此上界；正常回放保持捕获形状。提交12请求时通过队列分批，
没有把全部12请求同时塞入不受约束的decode shape。

实际Buffer.dispatch入口在通信前检查单节点EP4、BF16、hidden width2048、
非缓存handle、expert_alignment=1与物理发送行数。超界直接拒绝，不能说会
自动扩图或安全执行任意大batch。每rank实际创建cap16的5行/cap8的3行GPU
输入，通过同一dispatch入口验证超界异常，未调用不安全的通信。
CPU上界测试另验证unsupported EP/capacity与非法行数，6项通过。

## 实际完整模型工作负载

每种容量10类、53个请求、837个target tokens。temperature=0，ignore_eos，
真实logprob和route回传；每case先成功flush cache。变长输入由冻结prompt
切片产生，第四个原prompt只有137tokens，因此实际长度137，不能声称192。

| Case | 请求数 | 实际输入/输出变化 |
|---|---:|---|
| batch1/2/3/4/7/8/12 | 1/2/3/4/7/8/12 | 每请求16输出；12请求检验排队边界 |
| variable4 | 4 | 输入17/63/129/137，各16输出 |
| mixed-response4 | 4 | 输出1/4/16/32 |
| repeat-prompt8 | 8 | 相同prompt，各16输出 |

cap8/cap16逐请求的prompt、response token IDs、FP32 logprob int32 bits、
route形状与SHA256全部相同。route在接收时严格解码并验证logical expert范围、
每层top-k唯一性及完整 `(prompt+response−1,48,8)` 形状；保存的是验证后route
哈希和形状，本门禁不声称重新从保存数组复算全部路由，因为没有存全route数组。

所有8rank×2bucket均有48层dispatch/expert捕获与原生FullCudaGraphBackend
回放证明。每rank纯全局decode eager回退0，超界拒绝1次；prefill保持eager，
idle或全局prefill期间的局部行为单列，不称整个请求都进入一个graph。

## 专家padding与受控路由

独立eager shadow对相同10类请求，将padding行剔除后运行原expert core，
与固定cap的实际接收行输出按BF16 int16逐bit比较。两容量各8rank合计
67584次shadow、687022080个输出元素，均0 mismatch；各1710次零接收调用。
eager/full两种执行方式的完整模型tokens、logprob bits及route哈希也全部一致。
这验证padding数学和整模型，不能把额外shadow计算的运行耗时当作生产延迟。

再以两个EP4组进行实际normal graph dispatch/combine，物理发送均为2行，
synthetic identity expert仅作路由补充。8rank×5case，两种容量均零输出bit差异。

| 路由case | 每EP组各rank真实接收行数 | 边界 |
|---|---|---|
| all-to-rank0 | 8/0/0/0 | 最坏集中接收与其他rank零接收 |
| balanced-local | 2/2/2/2 | 本地均衡 |
| spread-all-ranks | 8/8/8/8 | 每发送token分散至所有rank |
| one-idle-sender | 6/0/0/0 | EP rank3逻辑发送0，物理padding仍2 |
| all-idle | 0/0/0/0 | 全idle、全零接收 |

padding route ID/weight和hidden state均在GPU侧处理；独立路由验证没有用
identity expert替代完整模型门禁，也没有假定真实模型总是均衡路由。

## 显存、捕获与初始化成本

PyTorch峰值在模型和pool设置后、hook安装时reset，因此不覆盖此前权重加载
峰值；记录allocated/reserved与每次捕获后/最终状态。NVML每1秒观测整卡驻留，
包括非Torch分配，但采样峰值可能遗漏瞬时峰值。两种统计不能混作同一指标。

| 指标，8rank范围或最慢rank | cap16 | cap8 |
|---|---:|---:|
| Torch peak allocated MiB | 50981.46–50981.52 | 50981.45–50981.51 |
| Torch peak reserved MiB | 51372–51396 | 51370–51396 |
| NVML sampled peak MiB | 55214–55506 | 55214–55506 |
| Engine初始化wall秒 | 308.274 | 125.292 |
| 两bucket捕获+backend warmup之和，最慢rank秒 | 67.450 | 1.741 |

cap8只减少约0.01MiB的allocated峰值，总GPU显存由模型、KV和通信等分配主导。
接收tensor容量减半不等于所有transport buffer、graph pool或整卡占用减半。
初始化测量包含加载、编译和捕获；先cap16后cap8共享运行缓存，首个capture
包含冷编译与各rank等待。不能从308→125或67→1.7宣称容量带来相应加速。
本阶段没有独立交错的cap8/cap16正式generate重复计时，cap8延迟收益未测。

最终保持一般实验默认cap16；cap8作为上述有guard的特定配置选项。未证明跨
节点、其他EP/CP/DP、量化/FP8或任意请求并发上界。Graph权重地址失效、
offload/wake与重建负例由 [P13报告](P13_ONLINE_FULL_GRAPH_LIFECYCLE_20261012.md)
提供实际验证，不能从容量实验额外推广其范围。

## 证据、发布和复跑

成功运行目录：

- `gpu_runs/p14-capacity-model-hgx03-20261012-01`：cap16/8模型、memory证据、
  requested args、capture/replay/guard、NVML样本与初始化成本。
- `gpu_runs/p14-controlled-routing-hgx03-20261012-01`：8rank逐case真实发送/
  接收、零接收、idle、物理shape和输出bit mismatch。
- `gpu_runs/p14-capacity-shadow-hgx03-20261012-01`：eager全模型及各rank实际-M
  padding shadow counters。

三组exit/state完整，source SHA锁定；发布版独立验收器重验三组数据通过，结果
在model run的 `independent-capacity-validation.json`。它检查实际工作负载、
tokens/logprob bits、已验证route哈希、各rank/bucket容量、native graph、guard、
allocator/NVML/capture成本，以及受控路由的预期接收行数，不只检查complete标志。

成功源码位于 `gpu_tools/p14_capacity/`。6份文件与实际GPU源码字节相同，4份
probe/launcher仅将P12依赖路径从临时目录适配到已发布目录；修改映射与SHA
在 `published-source-validation.json`，数学/配置与GPU实验一致。CPU上界检查
6项通过，全部入口编译和CLI检查通过；运行数据、权重和失败实现不发布main。

在相同环境、ai_hgx_03的已分配8GPU且GPU空闲时，从仓库根目录顺序运行：

```bash
python slime/alignment_a_preparation/gpu_tools/p14_capacity/run_capacity_probe.py \
  --run-id NEW_MODEL_RUN \
  --timing-run slime/alignment_a_preparation/gpu_runs/p12-full-timing-hgx03-20261012-01
python slime/alignment_a_preparation/gpu_tools/p14_capacity/run_controlled_routing.py \
  --run-id NEW_ROUTING_RUN
python slime/alignment_a_preparation/gpu_tools/p14_capacity/run_capacity_shadow.py \
  --run-id NEW_SHADOW_RUN \
  --model-run slime/alignment_a_preparation/gpu_runs/NEW_MODEL_RUN
python slime/alignment_a_preparation/gpu_tools/p14_capacity/validate_capacity.py \
  --model-run slime/alignment_a_preparation/gpu_runs/NEW_MODEL_RUN \
  --shadow-run slime/alignment_a_preparation/gpu_runs/NEW_SHADOW_RUN \
  --routing-run slime/alignment_a_preparation/gpu_runs/NEW_ROUTING_RUN
```

入口依赖本地checkpoint/冻结数据及P12结果，使用新的run ID，记录source SHA、
资源/P2P检查、内存与超时保护，退出时仅停止本任务worker。三组任务均已结束，
GPU compute列表为空；按用户指示未恢复保号。
