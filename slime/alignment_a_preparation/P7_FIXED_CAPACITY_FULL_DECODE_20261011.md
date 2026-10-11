# 专家 Graph 反序复测及固定容量 full decode Graph

已保存并推送先前的 KV 检查、RMS/QK 融合与专家 Graph 实现，完成反序复测，
再接入 DeepEP 单机 `num_worst_tokens=16`。当前确定性模型的整个 decode forward
已经可以原生 full CUDA Graph 捕获和回放；prefill、调度、采样及输出处理仍在图外。

## 当前测量结果

| 配置 | 正式批数 | 中位数秒 | P95 秒 | 输出 tokens/s |
|---|---:|---:|---:|---:|
| 反序复测：Breakable＋RMS，专家 eager | 20 | 6.123793 | 6.446355 | 167.22 |
| 反序复测：Breakable＋RMS＋专家 Graph | 20 | 4.333908 | 4.416358 | 236.28 |
| 随后测试：固定容量 dispatch＋RMS，full decode Graph | 20 | 3.268245 | 3.385360 | 313.32 |

full 相对专家 Graph 的中位数延迟下降 **24.589%**，吞吐提升 **32.607%**；相对
此次 Breakable＋RMS 参照的延迟下降 **46.630%**。full 为随后独立启动的一组，
不是反序复测中的配对启动；不把这个百分比当作高精度稳定加速率。

专家 Graph 两轮启动顺序与结果：

| 轮次 | 启动顺序 | 专家 eager 秒 | 专家 Graph 秒 | 配对延迟下降 |
|---|---|---:|---:|---:|
| 原试验 | eager → 专家 Graph | 6.171170 | 4.307625 | 30.198% |
| 反序复测 | 专家 Graph → eager | 6.123793 | 4.333908 | 29.228% |

两轮每组均独立 Engine、3 批预热＋20 批正式。额外的小 bucket 预热在计时之外。
同一 ai_hgx_03，8×A100-SXM4-80GB，Slurm 1031236；Qwen3-30B-A3B BF16，
TP8/DP2/attention CP2/EP4/MoE DP2，Triton attention/MoE，DeepEP normal 固定 20 SM。
确定性开关打开，使用 KV-once 和残差 RMS 候选，Q/K 联合候选关闭。

每批 8 个冻结 prompts、1218 输入 tokens，每请求 128 输出 tokens，总输出 1024。
测量范围为 blocking `Engine.generate`，包含调度、prefill、decode、采样、输出
logprob 和 detokenization/IPC，排除加载、捕获、预热、cache flush、训练及权重同步。
反序复测 40 批正式、full 20 批正式，无 profiler 或逐调用数值审计，零缓存命中。
运行中源码 SHA256 不变，worker 退出码均为 0；首次证据写盘均在预热结束前完成。
反序复测两模式的 23 对采样哈希一致；full 与反序专家 Graph 的 23 对哈希也一致。

这不是非确定性 baseline 的新测量，也不据此估算确定性额外开销。

## 解除此前 full 捕获障碍

此前普通 normal dispatch 在 CPU 上等待 GPU 返回接收 token 数，再动态分配
输出；实际 full 捕获因 `DeepEP error: CPU recv timeout` 失败。本次保持 normal
通信和原 Triton 数学 kernel，改用已安装 DeepEP 的单机固定容量接口：

1. decode/idle 范围内每次 uncached dispatch 传入 `num_worst_tokens=16`，固定
   接收 Tensor 形状，跳过 C++ 的 CPU 接收计数等待。通信 handle 每次新建，不缓存路由。
2. 当前配置每 rank 的 decode 输入至多 4 行，EP4，容量 16 是保守界限。实际
   验证中发送输入为 1/2 行；仍保留上界检查。prefill 的普通动态 dispatch 保持原样。
3. DeepEP 仅把填充 top-k ID 设为 `-1`，填充 X/weights 未初始化。专家计算
   前在设备上识别没有有效路由的行，把这些 X 行及无效路由权重清零，没有 `.item()`。
4. Triton 路径不使用 CPU `num_recv_tokens_per_expert` 列表，接收空列表可以正常
   工作。专家计算使用固定 M=16，包括被屏蔽的填充行；combine 使用本次 GPU 路由布局。
5. 仅实验入口绕过配置层对 DeepEP normal 的 full 后端禁用。使用原生
   `FullCudaGraphBackend`，没有 breakable stub、嵌套专家 Graph 或 eager MoE 回退。

full 模式下，Attention、Gate/TopK、RMS、dispatch、设备侧填充处理、专家计算、
combine、后续层及 LM head 均处于单次 model forward 的 Graph 内。输入 metadata
准备及 KV-once 验证在 replay 之前执行，采样等服务流程在 forward 之后执行。
因此这里的 full 指**整个 decode model forward**，不是整个请求/训练流程。

固定形状会增加填充处理及部分 GPU 工作。本次测量表明减少 CPU 等待和发射的
收益超过了这些成本；未采集新的 profiler launch 计数，不量化它们各自的耗时占比。

## 正确性与捕获证据

先执行固定容量 eager 的无计时 shadow：把有效接收行取出，按实际 M 再运行原
专家 core，对比固定 M=16 的真实行输出。参考输入独立复制，保持原 inplace 语义。

- 8 ranks 合计 21504 次逐 bit 对比，206491648 个 BF16 元素零差异，包含
  343 次没有本地接收 token 的情况；prefill 数据不走固定容量改动。
- eager 固定容量和 full 两种版本分别执行 batch8、batch3、batch1、variable3
  四组 greedy 输入，每请求输出 16 tokens。与既有 Breakable＋RMS 参考相比，
  tokens、专家 routes、logprob bits 全部一致，batch8 对 batch3/batch1 检查通过。
- 每个 rank 的 bucket 2/4 都成功捕获 48 次 dispatch 和全部 48 个专家 core；
  16 份捕获记录及 16 份实际 `FullCudaGraphBackend` 回放记录完整。
- full 正式测量中的 128-token 采样输出哈希与专家 Graph 的 23 组全部一致。

没有运行跨框架训练、参数更新/offload 生命周期或其它模型/拓扑门禁。试验限定
单机 EP4、BF16、H=2048、小 decode bucket、未缓存布局及 expert alignment=1；
超出参数范围会报错。生产 normal 的默认配置仍不自动启用 full，需使用显式实验入口。

## Git 与可复跑入口

先前验证版本已推送到 `origin/codex/kv-validation-once-per-batch`：

- `962a79d4`：KV 每批验证。
- `d5441e77`：普通 normal full 捕获失败探针。
- `fb87cd28`：Graph 融合及专家计算 Graph。

本阶段位于 `codex/fixed-capacity-full-decode`。工具：

- `gpu_tools/run_expert_graph_reverse.py`：字节相同实现的反序计时，复用已通过门禁。
- `gpu_tools/fixed_capacity_experiment.py`：显式配置和 experiment-only full 配置恢复。
- `gpu_tools/fixed_capacity_dispatch_hooks.py`：固定容量、设备侧填充清理及捕获证据。
- `gpu_tools/fixed_capacity_model_gate.py`：固定容量 eager/full 模型门禁。
- `gpu_tools/fixed_capacity_inference_benchmark.py`：相同 generate 协议的 full 计时。
- `gpu_tools/run_fixed_capacity_pipeline.py`：等待复测释放 GPU，资源锁、门禁和计时。

完整证据目录（相对本报告）：

- `gpu_runs/expert-graph-reverse-hgx03-20261011-01/`：反序结果和测量验证。
- `gpu_runs/fixed-capacity-full-hgx03-20261011-01/`：`compare-eager.json`、
  `compare-full.json`、`fixed-shadow-summary.json`、各 bucket 的捕获/回放证据、
  `results.json`、`measurement-validation.json` 和 `source-snapshots/`。

## 交接

反序 driver PID/PGID 21149、固定容量 driver PID/PGID 27594 及所有 worker 已退出。
GPU 均 1 MiB、0% 利用率，无 compute process；Slurm 1031236 保留 RUNNING。
按用户此前要求未恢复保号。Driver 日志在项目 `.a-tmp/`，分别为
`expert-graph-reverse-hgx03-20261011-01-driver.log` 和
`fixed-capacity-full-hgx03-20261011-01-driver.log`。
