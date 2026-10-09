# AutoPack 交换调度 GPU 验证（进行中）

2026-09-14，ai_hgx_05，8×A100 80GiB，TP1/EP4。固定 capture02 的 256 条真实回答、max_tokens_per_gpu=20480、global batch=256，使用预先选定的 search seed 0。

## 已完成的验证

RoutePack 和匹配 18 次交换的 Random-Swap 都已实际应用交换后的 schedule，8 ranks 的前向 gate 全部通过。每种 schedule 的 1,599,412 个 response token，在 capture、aligned fallthrough、R3 replay 三方 logprob bitwise 一致；route mismatch 为零。每 rank 12 slots。此结论只覆盖前向和路由，不覆盖梯度或更新一致性。

证据：RoutePack gate（本地产物：`gpu_runs/autopack-gpu-compare-20260914/runs/gate_routepack-01/gate-summary.json`）、Random gate（本地产物：`gpu_runs/autopack-gpu-compare-20260914/runs/gate_random-01/gate-summary.json`）。

## 完整 step 测评状态

尚无有效完整 step 时间，不能声称 GPU 提速。

| 尝试 | 结果 |
|---|---|
| timing_native-01 | 初始化 optimizer 遍历误访问 ChainedOptimizer.optimizer；独立 benchmark 已修并补回归测试。 |
| timing_native-02 | 模型输出转换 FP32 时申请 11.30GiB、仅余 10.02GiB，OOM。expandable allocator 已启用，非明显碎片问题。 |
| timing_native-03 | opt-in 保存 hidden activation 到 CPU 后越过输出转换，在 loss log_softmax 申请 594MiB、仅余 461.75MiB，OOM。 |
| timing_native-04 | 再将 loss 为反向保存的 softmax 精确搬到 CPU，越过损失前向；反向申请 11.59GiB、仅余 7.02GiB，OOM。 |

当前生产损失使用 torch.chunk，反向 SplitBackward 将所有 chunk 梯度拼接成完整词表梯度。CPU profiler 已验证该 cat 会额外分配完整 [T,V] 梯度；20480×151936×4 bytes = 11.5918GiB，与失败申请大小一致，但 GPU 栈仅到 autograd engine，确切 CUDA 分配节点尚未单独 profile。减少 chunk 大小不能消除这个完整缓冲。

原 recipe 使用 TP4/EP8，本次保留数值对齐已验证的 TP1/EP4；相同 20480 token 上限不意味着后者已具备完整训练容量。前向 gate 的成功不能替代反向容量验证。

现有内存实验均限于独立 benchmark 的 opt-in helper，不改生产 PPO 算子。CPU 测试验证已采用的保存张量往返和生产 logprob 算子输出/梯度 exact；不替代 GPU 梯度验证。若最终采用内存选项，三种方法必须设置相同，搬运成本计入完整 step。

## 测量约束

固定权重、lr=0、固定非零受控 advantage，执行真实 backward 和 CPU optimizer；正式轮次只计外层同步边界，逐层事件和梯度检查放在额外轮次。该实验隔离 schedule 对训练 step 的影响，不代表完整在线 RL 吞吐；rollout、路由准备、离线搜索与权重同步另列。

离线 RoutePack excess 的 2.9491%–4.0581% 改善仍是 assignment proxy；本轮 seed0 对应 3.1257%。CPU 搜索中位数 2.913s 必须进一步优化或摊销，不能在净收益计算中忽略。
