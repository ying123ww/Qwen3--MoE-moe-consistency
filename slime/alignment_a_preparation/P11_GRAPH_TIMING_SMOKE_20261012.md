# P11 full Graph 组件计时 smoke

组件计时机制已完成验证；P11整阶段尚未完成，A/C Nsight时间线和选定kernel
Nsight Compute仍待完成。性能结论继续引用P10，不引用本文instrumented延迟。

## 已完成

运行 `gpu_runs/p11-full-profile-smoke-hgx03-20261012-02/`，C确定性full模式，
同Qwen3-30B-A3B BF16/TP8/DP2/CP2/EP4/MoE DP2/cap16/buckets2/4/prefill eager。
8rank各32个选定回放，全部48层QKV/O、attention core、router、expert、dispatch/
combine来源覆盖；每bucket捕获全部48dispatch/expert，16捕获和16实际回放记录齐全。
实际标记共724个捕获区间/rank，包含嵌套父子范围、norm/QK/RoPE、logits和图内metadata。

使用PyTorch2.11的external CUDA Event节点，capture时记录，replay更新timestamp。
第32步同步后读取GPU elapsed_time，保存来源method/module/layer/depth/shape。
回放中没有逐层CPU读回或文件I/O，测量读取和操作数保存在选定窗口完成后。
NVTX标记同时记录源码范围和稳定decode step，供Nsight关联。
rank0/4的第0/47层QKV/O实际操作数保存，仅作为后续profile/有限kernel验证数据。
部分baseline输出可能被后续inplace规范化改写，不能直接把保存张量当未改动GEMM输出。

带event节点的最后选定完整model forward interval：rank0约30.108ms，rank4约30.685ms。
这是单个instrumented回放，包含依赖/等待及event开销，不是正常P50/P95或纯计算时间。
各source区间为inclusive，attention包含QKV/QK/core/O，MoE包含expert/通信。
父子相加会重复计数，不计算它们总和冒充端到端。

汇总CPU检查6项通过：保留嵌套行及来源，不混成可加的wall partition；缺失/重复
完整forward、负数/NaN、31步不完整记录全部拒绝。

## 处理过的问题

首次smoke完成了32回放和全部组件数据，但只实际重放bucket4，复用完整门禁
校验因此失败。补上计时外3prompt/2token小bucket预热后，第二次全部通过。
首次数据保留诊断目录，没有拿不完整bucket证据发布完整通过结论。

首个Nsight试验在CUDA JIT MaskKernel的cudaLaunchKernelExC处出现segfault；
另查到pynccl解析Condalib NCCL2.30.7而P10为2.28.9。试验进程已确认退出。
后续采集显式固定P10的NCCL路径，不能把库版本现象直接宣称为崩溃根因。
未验证的Nsight采集入口、失败日志留工作区/忽略目录；本文仅保存成功组件记录机制。

## 工具与交接

- `gpu_tools/full_graph_profile_hooks.py`：profile-only源范围和CUDA event节点。
- `gpu_tools/full_graph_profile_benchmark.py`：3预热后32选定回放、tokenizer广播控制。
- `gpu_tools/summarize_full_graph_components.py`：单回放inclusive统计及来源。
- `gpu_tools/test_full_graph_components.py`：6项CPU完整性/嵌套语义检查。
- `gpu_runs/p11-full-profile-smoke-hgx03-20261012-02/`：完整组件、真实workload、
  source SHA、capture/replay、操作数及退出证明。

smoke driver35101与worker已退出；Slurm1031869保留，P11 A/C Nsight重采进行中。
P10已完成且推送ff82e68d；P11–P14目标保持未完成，后续按顺序继续。
