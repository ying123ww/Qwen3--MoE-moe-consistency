# RoutePack 局部与完整 step 耗时分析

2026-09-16 离线分析 2026-09-14 已保存的 GPU 结果，无新 GPU 运行。128 条完整样本、global batch 128、max_tokens_per_gpu=10240、TP1/EP4、8×A100；Native 与 RoutePack 使用相同对齐实现和固定权重受控 advantage。Random GPU 按用户要求未运行。

## 结论

专家调用局部有小幅改善，完整 MoE 未同步改善，尚无稳定 step 加速证据。H 不是完全无效的 proxy，但 excess 的相对降幅显著放大了人对总耗时改善的期待；当前目标未直接覆盖完整执行路径。

正式完整 step：Native 中位数 141.05494s，RoutePack 140.67554s（下降0.269%）；均值反而由141.17832s升到141.48607s（上升0.218%）。每组仅5轮且同一batch，方法分开顺序运行，未随机交错；该差异不足以证明稳定收益。

## 逐层局部对比

完整检查8 ranks×13 slots×48 layers×2 stages×2 kinds，事件无重复、缺失或未配对；按真实EP拓扑[0,1,2,3]/[4,5,6,7]匹配。下表统计每个 collective 内最大 duration，再跨26×48个 collective 累加，只用于描述局部变化，**并非step关键路径或最晚完成rank耗时**。

| 区间 | Native(s) | RoutePack(s) | 降幅 |
|---|---:|---:|---:|
| 专家，forward | 8.15494 | 7.95841 | 2.410% |
| 专家，backward期间重计算的forward | 14.26370 | 14.09302 | 1.197% |
| 完整MoE，forward | 31.76814 | 31.64793 | 0.378% |
| 完整MoE，backward期间重计算的forward | 52.60747 | 54.66177 | -3.905% |

## 占比与抵消项

下表来自每种方法**额外的一轮带事件归因的step**，先在每rank累加，再取8rank均值；不能混用正式5轮中位数做组件分解。MoE含专家，两者不能直接相加。

| 同一归因轮的区间 | Native(s) | RoutePack(s) | RoutePack−Native(s) |
|---|---:|---:|---:|
| 观察到的专家调用（forward+重计算） | 9.14349 | 9.09781 | -0.04567 |
| MoE内其余区间（MoE−专家） | 24.98527 | 25.76937 | +0.78410 |
| F/B内未被上述MoE事件覆盖的区间 | 94.29469 | 93.89170 | -0.40299 |
| 完整step减F/B | 12.01138 | 12.32224 | +0.31086 |
| 完整归因step | 140.43483 | 141.08113 | +0.64630 |

观察到的专家区间约占Native归因step的6.51%；并非全部专家训练计算占比，因为专家autograd backward kernels不在forward hooks内。MoE非专家区间包含router、dispatch/combine、排列/gather、依赖等待和CPU发射间隙，**不能全部标作通信时间**。F/B以外约12秒包括optimizer、梯度处理、清零和metrics等；未单独测optimizer，不可归因为CPU optimizer变慢。

这里可描述为：单次归因中专家节省约46ms，被MoE其他区间增加约784ms等抵消。其因果关系和稳态幅度尚未验证，正式中位数与归因单轮方向不同也提示噪声和测量扰动。

## H/U与专家分布

- excess E：20,597,635→19,381,518，下降5.904%。
- 真正每层最忙rank assignments之和：103,566,211→102,251,790，仅下降1.269%。因为E减去了均值，5.9%是较小的偏斜部分的相对降幅。
- 总assignments仅下降0.118%，来自布局padding变化，完整真实samples不变。
- 每collective最忙目的rank的U之和仅下降0.317%；U excess下降4.733%。U是token/目的rank计数，不是字节或通信时间，更不包含完整source/destination send matrix。
- 每collective最大单expert负载之和下降7.215%，仍不能代表所有expert分块后的GEMM耗时。
- 描述性Pearson相关：Native H与expert区间duration，forward r=0.852，重计算r=0.913；同rank/slot/layer的ΔH与Δexpert duration，r=0.432/0.423。ΔH与Δ完整MoE duration约0.001/0.004。这些相关没有消除层、shape、等待等混杂，不是因果或预测泛化验证。

这支持H对专家执行有一定解释力，但不足以预测包含通信及等待的完整MoE区间。

## 实际执行路径核查

`slime/backends/megatron_utils/alignment/qwen3.py::_wrap_sglang_triton_moe`：SGLang Triton以no_grad算对齐输出；grad-enabled时再运行原生Megatron专家forward并通过bridge提供梯度。因此重计算阶段的expert wrapper可以包含两套forward、权重stack及布局处理，不能视作纯GEMM。DeepEP对齐桥另有route-preserving dispatch、combine和有序gather语义。这是两种schedule共有的成本；本次没有对其做独立消融，不能宣称其导致了多少秒的开销。

## 下一步

先保持算法，针对相同batch做一次有边界的分段测量：dispatch、local permutation、Triton forward、原生专家forward、combine/gather，以及autograd backward和CPU optimizer。需要明确stream/event依赖，不能仅用host调用时长冒充异步CUDA通信时间。正式step再做交错Native/RoutePack重复以估计漂移；不从本次单轮局部数据填写“最慢rank提速”。

本轮离线搜索记录为1.782s一次，完整prepare/search/validation调用44.601s；不是在线调度时延的稳定测量。若每step同步支付该搜索成本，会超过本轮0.379s的中位数差值，因此尚无净吞吐提升证据。

## 可复现证据

- `gpu_runs/autopack-gpu-compare128-20260914/local-attribution-analysis.json`：逐rank分解、proxy、相关及限制。
- `gpu_runs/autopack-gpu-compare128-20260914/gpu-comparison.json`：正式5轮原始时间。
- `gpu_tools/analyze_routepack128_local.py`：离线计算脚本，复用严格事件/topology join工具。
- Native输入：`gpu_runs/autopack-gpu-compare128-20260914/timing_native-01/timing.json`。
- RoutePack输入：`gpu_runs/autopack-gpu-compare128-20260914/runs/timing_routepack-01/timing.json`。
