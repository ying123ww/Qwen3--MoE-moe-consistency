# Unified decode KV 长度检查审计

## 已确认的根因

本项目确定性 unified decode 路径在每层执行：

```python
kv_lens = kv_indptr[1:batch_size + 1] - kv_indptr[:batch_size]
if bool(torch.any(kv_lens <= 0).item()):
    raise ValueError("Unified decode alignment received an empty KV sequence")
```

用途是拒绝空 KV，避免把负 prefix length 传给 unified Attention。kv_lens 在 GPU 上，any 得到 GPU 布尔标量；Python 分支需要 .item() 把它读回 CPU。

这条检查由本地 SGLang 仓库提交 514ca5a400859d0ea1b4f17963fcad6461fdf777（fix: expose deterministic Qwen3 attention and MoE paths，2026-08-22）加入。其父提交没有该 unified decode 分支。baseline 关闭 SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE，不执行这条检查；它不是 DeepEP 通信或 batch-invariant RMSNorm 自身的要求。

## Trace 调用链证据

rank 0 原始 trace 已完整核对 6144 个事件，全部满足：

```text
P1/attention + step[DECODE]
  aten::item
    aten::_local_scalar_dense
      cudaMemcpyAsync → 1 byte Memcpy DtoH
      cudaStreamSynchronize
```

CPU scalar op 的 External id 与 D2H 事件一致；CUDA API 与 D2H 的 correlation 一致；同线程的 item/scalar/API 包围关系一致。不是根据 kernel 名或总拷贝数推测。

16份原始 trace 的调用链复核已全部通过：baseline 8 ranks 在此调用点均为0次；确定性 RMS 版本8 ranks均为6144次 .item()、6144次1字节D2H和6144次显式同步。全8卡合计49152次。逐rank证据及complete.json保存在 .a-tmp/kv-item-audit/all-ranks；rank 0 的独立证据是 .a-tmp/kv-item-audit/rank0-callchain.json。

每次只复制1字节；主要问题是6144次显式流同步。数量等于48层乘以本次trace中128次decode forward。EagerRunner 在模型 forward 前初始化该 batch 的 Attention metadata；当前 dcp_size=1、无 sliding window、无 speculative 配置中，48层读取同一份 kv_indptr，因而重复检查同一批长度。

## 当前 Graph 修正及验证边界

Plan 2 捕获时使用 torch._assert_async(torch.all(kv_lens > 0), message)，条件检查也被捕获，并随每次 replay 在 GPU 上执行。eager 路径仍保留原 .item() 检查；本轮对照没有同时优化 eager 的检查位置。

正常输入已通过完整模型有限门禁：batch8/batch3/batch1/variable3 的 tokens、routes、logprob bits 均一致，batch划分门禁通过；8 ranks 成功 replay batch2和batch4，各有49个segment和48个MoE eager break。

异常处理并非完全等价：原路径抛同步 Python ValueError；CUDA异步断言失败在后续CUDA调用中报告，且会使该进程的CUDA上下文失效。不能把正向模型门禁说成异常路径已经验证。

独立进程GPU测试已在正式三轮计时结束后完成，持有同一项目锁并确认GPU空闲，没有与8卡计时争抢资源。状态与结果位于 gpu_runs/kv-guard-replay-hgx03-20261011-01/results.json：

- eager-invalid：零长度触发 ValueError，后续CUDA操作正常。
- capture-valid：三组动态正长度（包括全部为1的padding长度）重放通过，输出读取新输入，未冻结捕获时的值。
- capture-invalid：捕获时长度为正、重放时改为零，触发 AcceleratorError / device-side assert；该进程后续CUDA操作继续报错，证实上下文失效。测试仅使用独立进程的CUDA上下文，退出后释放。

这验证的是断言原语的捕获/重放和异常语义，并非完整分布式模型的错误KV恢复流程；不能宣称已经测试了整模型异常恢复。

## 性能结论的边界

能确认这条检查引入了真实D2H和CPU等待，不能仅凭6144次事件推导它单独占多少秒。三轮Plan 2相对原确定性eager的配对延迟下降中位数为59.073%，对Plan 1为52.541%，对当前eager baseline为49.293%；这是整个分段Graph方案的收益，不能全部归因于这一行检查。

最终核对发现，60批Graph正式样本中有2批首次写出了batch2的replay证据；并非所有证据都在预热。保持原记录，剔除这2批的敏感性分析中，各轮中位数最大变化0.0225%。这是观察包装与一次性记录的计时，未测完全移除包装的配置。完整数据与边界见 P2_BREAKABLE_CUDA_GRAPH_PERFORMANCE_20261011.md。

如果继续优化 eager，合理调查方向是在每batch的metadata准备阶段验证一次实际KV长度，并保留当前受支持拓扑及padding语义；需单独进行数值门禁和无profiler消融计时。当前正式实验源文件保持锁定。
