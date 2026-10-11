# KV 检查每 batch 一次，与 baseline Graph 单轮运行

## 已保存的 Plan B

Plan B 实现、工具及最终报告先提交并推送至 codex/breakable-cuda-graph-plan2，提交 a94d24235a31c2ffc7dd5db1f5e8107c5466eb33。本次 KV 修改在独立分支 codex/kv-validation-once-per-batch。

## KV 优化完成，未做性能计时

把空 KV 检查从每层 forward_decode 移到每 batch 的 metadata 准备阶段。eager 和 CUDA Graph 的 capture/replay 准备均检查一次实际 GPU kv_indptr；baseline 和 prefill 跳过这项检查，空 idle batch 不读回标量。仍保留每个非空 decode batch 一次GPU标量读回，并非零同步。

Graph replay 检查本次 bs 对应的 live self.kv_indptr 缓冲，避免误用上次捕获形状或 eager prefill 留下的 metadata view。metadata 记录已校验的行数，模型每层只读取这个 Python 标记；未校验或超出校验范围时明确报错，不再在每层 .item() 或触发设备侧断言。

检查使用 ValueError，并在模型/graph replay 前拒绝零长度。正常 Attention、RMS、RoPE、Gate 和 MoE 数值计算未改。

验证完成：

- 6 项 CPU 检查通过，包括48层forward只读回一次标量、零/负长度拒绝、正padding/空batch、baseline/prefill跳过、不同capture bucket的live缓冲验证。
- 实际 TritonAttnBackend GPU validator 测试通过：三组动态正长度重放输出读取新输入；零/负长度在replay前拒绝，ValueError后CUDA上下文仍可用。此项为最小GPU metadata测试，不是完整分布式模型的异常恢复测试。
- eager 与改动前eager、Graph与改动前Graph，以及新eager与新Graph的三组有限模型对照均通过：batch8、batch3、batch1、variable3 的 tokens、routes、logprob bits 一致，batch划分一致性通过。

按用户要求没有对 KV 优化做性能计时，不报告其加速百分比。证据位于 gpu_runs/kv-once-baseline-graph-hgx03-20261011-01 的 gpu-kv-check.json、compare-*.json 和 optimization-complete.json。

## 非确定性 baseline 开启 Graph：已跑一轮

仍为 ai_hgx_03 / TP8 / DP2 / CP2 / EP4 / MoE-DP2 / A100 / DeepEP normal 固定20 SM。enable_deterministic_inference=False、unified decode关闭、RMS和Q/K实验融合关闭；decode breakable、prefill eager、DeepEPMoE专家主体eager。

3批预热后，1批正式generate耗时 **5.081420秒**。固定8 prompts、1218输入tokens、每请求128输出tokens，共1024输出tokens；约201.52输出tokens/s。所有8 ranks有实际Graph replay证据，49个segment、48个MoE break。

这是单批运行结果，不是多轮统计或重新测量KV优化后的确定性Graph。旧Plan B约6.45秒记录与该单批结果不能直接当作精确、配对的确定性额外开销。

本次4批（含预热）的采样输出哈希与旧eager baseline前4批仅1批相同。该baseline未进行Graph/eager的greedy数值门禁，不能宣称baseline开启Graph后逐bit等价，也未单独定位哈希变化的原因；这不影响前述确定性路径的模型门禁结果。

baseline日志与结果位于 gpu_runs/kv-once-baseline-graph-hgx03-20261011-01/baseline-breakable，实际配置、源码校验和运行核验见同目录父级validation.json。

## 交接

驱动31848已正常结束；8卡均1MiB、0%利用率、无计算进程。按用户要求不恢复保号，Slurm分配1031236保留。
