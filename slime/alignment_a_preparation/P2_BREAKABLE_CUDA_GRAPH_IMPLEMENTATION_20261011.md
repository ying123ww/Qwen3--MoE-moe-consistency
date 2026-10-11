# Plan 2：Breakable CUDA Graph 的实现与门禁

## 实验范围

Plan 1 已推送到 GitHub main：f139801f 保存三模式复现，f99fd917 保存 Q/K 融合与 QKV/O 调查。Plan 2 在 codex/breakable-cuda-graph-plan2 分支独立测试。

Plan 2 不启用 RMS 融合或联合 Q/K 融合。保持原始确定性算子、TP8 / DP2 / CP2 / EP4 / MoE-DP2、A100、DeepEP normal 和固定 20 SM 通信配置。仅 decode 使用 breakable；prefill 保持 eager。

本地 SGLang 的全局禁用逻辑在 server_args.py，而非最新上游拆分后的 arg_groups/moe_hook.py。Qwen3 forward_deepep 调用 self.experts，进入已存在的 DeepEPMoE.forward 和 eager_on_graph 包装。整个 MoE forward，包括 dispatch、expert compute、combine，均为 eager。

## 必要兼容修正

1. DeepEP normal 只允许显式选择的 breakable backend；默认行为和 full backend 仍禁用 Graph。CPU 测试覆盖默认、显式 full、decode/prefill 显式 breakable 和显式 disabled。
2. 本地 breakable backend 原先只支持 tensor、list/tuple 和 PPProxyTensors。decode 返回 LogitsProcessorOutput，首次启动因此失败。补齐 next_token_logits / hidden_states / full_logits 的静态输出缓冲、前缀切片和复制；拒绝不能重放的其他元数据。CPU 测试覆盖不同形状共享地址和元数据拒绝。
3. 原确定性 unified decode 每层用 torch.any(kv_lens <= 0).item() 检查空 KV，第二次启动因捕获内 D2H 同步失败。捕获时改用设备侧异步断言，每次 replay 仍检查动态 KV 长度；eager 保留原检查。Attention 数值计算未改。

前两次失败的日志和状态分别保存在 gpu_runs/breakable-smoke-hgx03-20261011-01 与 -02。驱动只清理自己的 worker 进程组；Slurm 分配保留，按用户要求不恢复保号。

## 已通过的验证

- 7 项 CPU 测试通过。
- 第三次启动验证完成，8 ranks 实际 replay；每个形状有 49 个 Graph segment 和 48 个 break，所有 break 函数均为 DeepEPMoE._a2a_forward_with_output_impl。
- 完整模型有限 greedy 门禁通过：batch8、batch3、batch1、variable3 四组输入的 tokens、routes、logprob bits 与原确定性 eager 逐 bit 相同；batch8 对 batch3/batch1 检查也通过。
- 8 ranks 均实际 replay 了 batch 2 和 4 的捕获形状。未测试跨框架训练一致性。

门禁证据：gpu_runs/breakable-model-perf-hgx03-20261011-01/exact-comparison.json、exact-breakable/graph-evidence。运行目录按仓库约定保留本地，不纳入 Git。

## 性能对照

结果见 [Plan 2 性能报告](P2_BREAKABLE_CUDA_GRAPH_PERFORMANCE_20261011.md)。四组是 baseline、原确定性 eager、Plan 1（RMS＋联合 Q/K）和 Plan 2（原算子＋breakable）。三轮、12 个独立 Engine，每 Engine 3 预热＋20 正式；固定8 prompts / 1218 输入 tokens / 每请求128输出 tokens。计时不启用 profiler 或逐调用审计；每形状在首次成功 replay 记录一次证据。batch4均在预热记录，但第1、3轮各有一批正式样本首次记录了batch2。原记录保留，剔除这两批后的中位数最大变化0.0225%；不是完全移除观察包装的测试。

当前 baseline 关闭 Graph；没有测 baseline 也开启 breakable 的对照，因此不能将相对该 baseline 的收益解释为确定性计算本身比非确定性计算更快。KV检查的完整调用链与独立断言异常测试见 [KV检查审计](P2_KV_ITEM_AUDIT_20261011.md)。

从已获得的 ai_hgx_03 SSH 分配中复跑：

```bash
/public/home/yingxy2024/.conda/envs/slime/bin/python \
  slime/alignment_a_preparation/gpu_tools/run_breakable_model_and_performance.py \
  --run-id breakable-model-perf-hgx03-新的编号
```

需要已有模型、冻结 rollout、运行环境 JSON 和同等资源分配；脚本检查 GPU 空闲、资源、P2P，持有项目实验锁并保护 worker 内存。源码 SHA 与快照记录在每次实验 manifest 中；正式测量期间不能改变被锁定的源文件。
