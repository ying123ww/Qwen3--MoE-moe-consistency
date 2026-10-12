# P11补充：实际Prefill、Sampling和图外执行

**补充验收完成。** A/C真实uncached prefill窗口已采集Nsight Systems，8rank
报告齐全、两worker退出码0、source SHA不变、实际输入/LP/路由门禁通过。
发布版解析器对rank0/4按CUDA runtime/driver correlation和同线程最内层NVTX
归因，区分prefill、局部decode/idle、图外操作及附带Graph回放。与
[P11 decode归因](P11_FULL_GRAPH_PROFILE_20261012.md) 一起覆盖原计划范围。
仍不采集用户已取消的NCU硬件计数器。

## 实际配置与窗口

运行 `gpu_runs/p11-outer-prefill-hgx03-20261012-03`，分配1032144/ai_hgx_03，
相同8×A100、Qwen3-30B-A3B BF16、TP8/DP2/CP2/EP4/MoE DP2，full decode2/4、
cap16、prefill eager。A默认非确定性通信/数学，C固定通信和确定性路径；
沿用P11配置，没有带入P12的QKV/O、Q/K或Router候选。

先预热真实max_new_tokens=1/128/1请求，再开始profiler并请求8个冻结prompts，
共1218输入tokens、各1响应token、temperature0.6、ignore_eos，回传LP/routes。
NPZ检查8targets有限、输入数量、route SHA/完整shape/expert范围/唯一top-k。
相同输入不意味着A/C响应必须相同。

每rank实际2次eager prefill，物理input rows470/140、batch3/1；CP padding
使物理行数不直接等于logical输入数。另有1次原生full replay，且可能有局部
decode/idle，即使生成1token也不能称纯prefill。解析器将Graph kernels排除于
eager组件表、保留在全窗口GPU并集，eager各模式由外层NVTX进一步分开，
sampling/metadata等图外活动另归因。

## 观测结果

| 整个补充窗口 | A rank0 | A rank4 | C rank0 | C rank4 |
|---|---:|---:|---:|---:|
| GPU活动并集 ms | 129.383 | 118.133 | 145.849 | 155.635 |
| GPU活动duration sum ms | 129.408 | 118.166 | 146.035 | 155.804 |
| GPU首尾envelope ms | 351.506 | 349.791 | 434.075 | 436.715 |
| Sampling GPU kernel sum ms | 0.5506 | 0.5525 | 1.2804 | 1.2758 |
| D2H copy次数 | 16 | 16 | 17 | 17 |

前3行包含窗口全部活动及附带Graph，非prefill单项。并集外间隙可含CPU发射、
GPU依赖/通信等待，不能都计为CPU launch；数据带profiler/NVTX/事件节点，
不替代无profiler生成延迟。

下表仅汇总实际eager_prefill范围内kernel，ms/两次prefill，排除局部decode和
附带Graph。按最内层来源分类，不把父inclusive NVTX重复加入子项；跨stream
仍有重叠，不能将各行相加当wall分区。

| Prefill来源 | A rank0 | C rank0 | A rank4 | C rank4 |
|---|---:|---:|---:|---:|
| QKV | 2.8519 | 5.1358 | 2.8716 | 5.1634 |
| O projection | 2.6728 | 4.8536 | 2.6653 | 4.8810 |
| Attention core | 3.9508 | 5.7798 | 3.9814 | 5.7908 |
| Router | 0.9559 | 9.8715 | 0.9633 | 9.8756 |
| Residual norm | 1.0686 | 9.3868 | 1.0740 | 9.3981 |
| Experts | 34.0070 | 34.1761 | 34.0571 | 34.2494 |
| DeepEP dispatch | 22.9357 | 13.6705 | 18.9865 | 19.9405 |
| DeepEP combine | 13.0343 | 9.7149 | 10.1348 | 10.2563 |

Prefill也有Router、投影、norm、attention差异。通信kernel含依赖/等待，不能
由一次窗口宣称通信普遍更快。Sampling在该窗口GPU量级较小，但没有据此
推导整个1024输出请求的CPU/sampling占比。正常性能结果仍取P10/P12。
图外D2H存在，不能将其称为每层decode KV `.item()`。

实际CPU cuda API次数/持续时间、metadata来源、逐kernel计数和并集保存于
`outer-timeline-summary.json`。Nsight把部分cuLaunchKernelEx driver调用也存于
CUPTI_RUNTIME表，按实际schema/correlation处理，未按kernel名猜Triton模块。
未归因活动单列，不伪造分类。

## 落盘竞争、源码与复跑

补采01因错误假定一token不会回放Graph而被协调器拒绝；02关闭Engine后只
收到7个rank报告。成功03在cudaProfilerStop前写报告、增加CPU TP屏障，并
等待8份文件后关闭Engine。失败实现留本地，未发布main；模型数学和kernel
launch没有改变，完整请求路径在任何API开始前预热。

原分配1031869被管理员取消后，SSH确认无活动job，成功补采使用1032144。
不把旧PID或旧资源状态当存活证据。

成功工具 `gpu_tools/p11_outer/` 的hook/launcher与GPU源码字节一致，benchmark
仅增加已发布工具import路径，映射SHA在 `published-source-validation.json`。
发布版CPU解析器重验两模式SQLite通过，真实workload门禁独立复查通过。
原始nsys-rep/SQLite/NPZ保留忽略目录。复跑（同环境、已分配8GPU且空闲）：

```bash
python slime/alignment_a_preparation/gpu_tools/p11_outer/run_outer_profile.py \
  --run-id NEW_UNIQUE_RUN_ID --nsys /absolute/path/to/nsys
python slime/alignment_a_preparation/gpu_tools/p11_outer/summarize_outer.py \
  --run slime/alignment_a_preparation/gpu_runs/NEW_UNIQUE_RUN_ID
```

入口依赖本地checkpoint/冻结数据，不能覆盖旧run ID。加载/导出耗时和带插桩
窗口不是性能benchmark。任务已结束、GPU compute列表为空，未恢复保号。
