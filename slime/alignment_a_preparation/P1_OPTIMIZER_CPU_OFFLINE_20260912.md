# CPU optimizer 离线验证（2026-09-12）

本轮仅解析已有四份 GPU trace、核对 CPU optimizer 源码和历史启动配置；没有使用 GPU，没有修改训练路径。

## 互斥墙钟归因

对 P1/optimizer_step 主线程区间做区间扫描：CUDA API 优先，否则归入最内层 CPU 算子；未覆盖部分保留为 unattributed_host。每份结果各项加和严格等于 optimizer 墙钟时间。CPU 算子墙钟包含内部线程等待，不等同 CPU core time。

| 样本 | 总耗时 s | 数学 s | fill_ s | H2D API s | D2H API s | 显式同步 s | 其余 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| native rank0 | 15.402 | 6.770 | 1.604 | 4.501 | 0.530 | 0.148 | 1.849 |
| native rank4 | 15.433 | 6.739 | 1.534 | 4.710 | 0.583 | 0.040 | 1.828 |
| aligned rank0 | 15.411 | 6.743 | 1.506 | 4.700 | 0.550 | 0.148 | 1.764 |
| aligned rank4 | 15.512 | 6.776 | 1.522 | 4.821 | 0.572 | 0.040 | 1.780 |

数学项包括 abs、mul、add_、addcdiv_ 等，已扣除嵌套子算子/API。fill_ 单列，避免误报为纯数学或搬运。native rank0 有 9233 次 fill_；抽查最长六次均嵌套于 abs。项目初始化存在 use_deterministic_algorithms(True) 路径，因此“确定性相关未初始化内存填充”值得验证，但没有调用栈或开关对照，尚不能确认为原因。

## 实际重叠

| 样本 | H2D GPU union s | 与 CPU 数学重叠 ms | H2D 位于其 API 内比例 |
|---|---:|---:|---:|
| native rank0 | 3.081 | 0.235 | 97.44% |
| native rank4 | 3.202 | 0.254 | 97.33% |
| aligned rank0 | 3.277 | 0.190 | 97.56% |
| aligned rank4 | 3.310 | 0.116 | 97.44% |

开启 overlap 的这四份采样中，H2D 与 CPU 数学几乎没有重叠；约 97% 的 H2D GPU 时间落在对应方向 API 执行区间内。结合此前实际 pinning smoke 与源码，pageable FP32 master 回拷仍是明确排查目标。API 时间不是纯 PCIe 传输时间，也不能与 GPU memcpy 时间相加。

每份 H2D 共 15,266,061,312 bytes（14.218 GiB）。rank0 为 1537 次、rank4 为 1612 次，全部发起 API 位于 StatelessAdam 子 optimizer scope 内，该 scope 包含回拷 hook。native rank0 中 768 次 6 MiB 和 768 次 12 MiB 回拷合计占 H2D API 4.409 秒；唯一 734.85 MiB 大参数约 0.092 秒。应优先处理重复的中等大小回拷，而非只盯最大参数。

## CPU 配置与下一步

历史 runtime_env 记录 OMP_NUM_THREADS=4；launcher 允许 64 个逻辑 CPU。没有保存实际训练 worker 的 torch 线程数、CPU affinity、NUMA 内存分布、内存带宽、VmLck 或 memlock 上限，因此不能据此宣称线程超配、NUMA 远端访问或带宽饱和。

建议按以下顺序继续：

1. CPU-only 小实验：复现 StatelessAdam 的 abs→mul→add_→addcdiv_，扫描线程数，并对照确定性开关与未初始化内存填充设置；同时验证输出逐 bit 相同。重点解释额外 fill_，测量重复内存遍历，而不是直接替换数学公式。此实验尚未执行。
2. 后续搬运修复验证：保留 FP32 master 的 pinned 属性，审查 H2D 完成事件当前记录在 D2H stream 的依赖风险；检查正确性后再测真实收益。本轮未修改实现。
3. 将共同 optimizer 耗时与一致性增量分开报告：native/aligned 的 optimizer 均约 15.4 秒，当前证据不支持将这部分全部算作一致性额外开销。

不能从四份 profiler 样本推导稳定吞吐收益、RoutePack 价值，或优化后必然能节省多少 step 时间。没有 GPU 的离线分析能定位候选，但不能验证新的异步搬运实现。

## 可复查产物

- `gpu_tools/analyze_optimizer_exclusive.py`：可重复分析脚本；嵌套/API 优先与区间求交已用合成区间断言验证。
- `gpu_runs/p1-optimizer-exclusive-20260912.json`：四份 trace SHA256、脚本 SHA256、完整互斥计时、传输尺寸和重叠。
- `gpu_runs/p1-optimizer-fill-context-20260912.json`：native rank0 最长 fill_ 的父调用。
- `gpu_runs/p1-optimizer-cpu-config-audit-20260912.json`：历史配置与未观测项。

源码核对反映当前文件，不证明 9 月 10 日文件完全相同。JSON 的 before_first_cpu_math_s 指任何数学算子之前，不能解释为第一轮 Adam 更新前的 D2H 时间。

## 追加：为什么 overlap 没形成流水（CPU 检查）

重新读取实际 dependency 的 HybridDeviceOptimizer：overlap 开关创建 D2H/H2D streams，并把每个参数拆成一个子 optimizer，但 CPU 的 step 和 post hook 在同一线程串行调用。调用顺序为所有梯度 D2H 提交 → 各参数梯度 event.synchronize → CPU 数学 → H2D copy_ post hook → 下一个参数。独立 stream 只提供并行机会；copy_ 和 hook 的主线程执行未结束，下一次 CPU 更新不能开始。

当前源码第 274 行先 pin_memory，第 277 行 clone().float() 后 FP32 master 丢失 pinned 属性；此前实际节点 smoke 也确认该行为。因此 pageable 回拷是主要嫌疑，不能凭 non_blocking=True 断言 host 能立即返回。历史 trace 缺少完整调用栈，不能进一步拆清 CUDA API 内部的 staging、驱动等待和 profiler 开销。

新增逐笔关联结果（gpu_runs/p1-optimizer-pipeline-cpu-20260912.json）：

| 样本 | 有后继子 optimizer 的回拷 | 下一子 optimizer 开始前已经完成 |
|---|---:|---:|
| native rank0 | 1536 | 1533 |
| native rank4 | 1611 | 1607 |
| aligned rank0 | 1536 | 1534 |
| aligned rank4 | 1611 | 1607 |

合计 6281/6294（99.79%）回拷在下一次 CPU 子 optimizer 开始前已经完成，直接支持更新/回拷近似串行。注意：这些 memcpy 均有尾部超过各自发起 API 返回时间，所以不能说“API 等待整笔 GPU copy 完成才返回”。准确描述是 API 占据了大部分回拷时间，再加上后续 hook/调度间隔，几乎耗尽与下一次 CPU 数学重叠的窗口。

另有独立的 stream 依赖问题：第 124 行在 H2D stream 提交回拷，第 125 行却在 D2H stream 记录 event 并让 current stream 等待。它没有直接表达 H2D 完成依赖；这是需要修复前审查的正确性风险，不是本次 CPU 分析已证明的数据错误，也不能直接认定它造成上述主线程阻塞。event.wait(stream) 与 event.synchronize() 的语义不能混为一谈。

本次新增脚本只解析历史 trace，不初始化 CUDA、不改 optimizer。修复方向仍是保留最终 FP32 master 的 pinning、明确 H2D 完成依赖，并在 GPU 可用时重新验证正确性和 overlap。当前证据能说明串行表现与代码机制，不能证明修复后的实际加速比例。

## Upstream 核实与结论更正

2026-09-12 核对 GitHub PR 页面：

- https://github.com/NVIDIA/Megatron-LM/pull/4547 已于 2026-08-21 合并 main；PR 说明在最终 CPU FP32 clone 后、cpu_copy && pin_cpu_params 时重新 pin_memory，正是本地观察到的 pinning 丢失问题。
- https://github.com/NVIDIA/Megatron-LM/pull/3140 已于 2026-02-01 合并 main；PR 确认旧 hook 在 D2H stream 记录 event 无法保证 H2D 完成，会产生竞态，修复为在 H2D stream 上记录 event。

实际依赖 /public/home/yingxy2024/slime-deps/Megatron-LM 的 HEAD 为 1dcf0dafa884ad52ffb243625717a3471643e087。当前 hybrid_optimizer.py 仍在 clone().float() 后直接写入映射，且回拷后仍为 _d2h_stream.record_event()；该文件未列在 git status 修改清单。故当前实际文件没有这两个修复的效果。未据 PR 编号或日期推断提交祖先关系。

更正此前表述：这是 upstream 已修复、当前依赖仍保留的已知问题；同步问题应归为已确认的实现 bug，而非仅未经 upstream 确认的风险。不过，旧实验是否实际发生参数读取竞态，以及这两个修复在本工作负载的性能收益，仍未经实测。此次仅核实并更新报告，没有升级依赖或应用补丁。

## 已移植修复

经用户授权，已在实际依赖 hybrid_optimizer.py 移植 #3140 与 #4547：H2D 完成 event 改为 H2D stream；最终 offloaded FP32 参数在 pin_cpu_params=True 时重新 pin_memory。未整体升级依赖。AST 语法检查与 git diff --check 通过，未进行 GPU 或整训练验证。可复用补丁保存在 gpu_tools/optimizer_upstream_3140_4547.patch，文件哈希记录在 gpu_runs/p1-optimizer-upstream-backport-20260912.json。此前 trace 属于修复前数据；已经加载旧模块的训练进程需重启后才能使用修复。

## 2026-09-14：CPU 数学与 fill_ 实验已完成

已完成 CPU-only 的确定性填充开关对照、1/2/4/8线程扫描及 abs().mul_() 隔离候选验证。当前复现中的 fill_ 来源已确认；候选逐 bit 检查通过，但8线程原实现存在明显计时异常，不能直接推荐生产线程配置。完整结果与限制见 [CPU实验报告](P1_OPTIMIZER_CPU_MATH_20260914.md)。未修改线上 StatelessAdam。
