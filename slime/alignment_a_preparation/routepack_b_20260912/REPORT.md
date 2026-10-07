# RoutePack B1/B2：2026-09-12 执行与验证结果

> 最新结果：`routepack-b1-20260912-02` 的 8 rank 前向 B1 gate 全部通过，B2 已基于真实 topology/schedule/dispatch 生成并独立复核计数。多批 workload、耗时与调度干预仍未运行；下文早期“未运行”均为当时状态。

本轮完成独立 CPU 数据工具、运行时 topology hook 和独立 route capture 入口，随后实际运行 B2 预备采集。首次采集发现确定性 Qwen3 路径遗漏 route capture，已修复并通过 5 项 CPU spy 回归，第二次 GPU 复验成功，取得并复核 8/8 条合法 routes。**没有更改在线 scheduler，没有运行训练 B1 的实际 dispatch/replay gate，也未作 Go 决定。** Route capture 使用独立进程与运行目录，未混入此前矩阵乘性能归因。

## 实际复核

使用 `torch.load(map_location='cpu', weights_only=True)` 重新读取 v89 固定输入：

- 输入：`/public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt`
- 8 条样本；`rollout_routed_experts` 全部为空，覆盖率 **0/8**。
- `torch.cuda.is_initialized()` 为 false。
- 文件 hash、sample/rollout/reward-group ID、逐样本长度保存在 `legacy_rollout_inventory.json`。

因此，历史 fixed-rollout replay 只能复用 tokens/logprobs，不能作为 Routing Replay trace。不能由历史 exact logprob 推断新 R3 路径已通过。首次不受限 pickle 读取被自动审查拒绝，随后受限 weights-only 读取成功；没有遗留权限阻塞。

源代码审计确认：

1. `actor.train_parallel_config` 仅包含 dense DP、CP、VPP 等调度字段，未包含真实 EP / expert DP / expert owner。不能按 dense DP 数推断独立 EP 副本数。
2. `cp_utils.prepare_routed_experts_for_routing_replay` 对每个样本补一行末端 route，再按本地 microbatch 补齐。每次 `_pad_routed_experts` 都从 `arange(0)` 开始，因此末端和 bin padding 必须分别计数；末端 token 不等于训练 padding。
3. R3 的 `compute_topk` wrapper 和 Qwen ordered-topk capture 存在组合入口，但本轮没有运行实际 router/dispatch 比较；“源码存在入口”不等于兼容 gate 通过。
4. 实际 HF config 是 48 层、128 experts、top-k=8。冻结 SGLang 支持 Engine 级 `enable_return_routed_experts=True` 和请求级 `return_routed_experts=True, routed_experts_start_len=0`。响应 base64 int32 的期望 shape 是 `[prompt+response-1,48,8]`。

## 可用交付

代码和完整命令见 `../../tools/routepack/README.md`：

| 文件 | 作用 | 实测状态 |
| --- | --- | --- |
| `tools/routepack/runtime_topology.py` | 原 Qwen hook 后读取真实 process-group 成员和模型 local_expert_indices，逐 rank 输出 | 已准备；尚未运行时采集 |
| `tools/routepack/profile_data.py` | 合并/校验拓扑、转换实际原生 schedule、H/U/per-expert 摘要、末端/padding 和 EP group/slot 聚合 | 纯 CPU 测试通过 |
| `tools/routepack/inspect_rollout.py` | 受限 CPU 读取并记录已有 rollout 的 trace 覆盖 | 已用于真实 v89 文件 |
| `tools/routepack/capture_routes.py` | 校验并保存真实 Engine 响应中的 routes、token 身份、stable IDs 和 hashes | CPU 测试及真实 8/8 routes 校验通过 |
| `tools/routepack/capture_engine.py` | 复用 aligned inference manifest，独立 Engine 生成一批短 response 并采 routes | 修复后 GPU 采集成功退出 |
| `tests/test_routepack_profile_data.py` | 19 项 CPU 约束/反例测试 | **19 passed** |

测试覆盖 H 与 U 不同、缺失/重复/越界 trace、非法 owner、step 边界、dense DP 到 global rank 映射、同 collective 源 rank 交换不改变接收 H、singleton 跨 EP 副本交换可以改变 collective 负载。还直接调用生产 `cp_utils.prepare_routed_experts_for_routing_replay`，在 CP=TP=1 下验证普通/allgather 两种入口的末端和 padding route 一致。CPU Megatron stub 触发一次预期的 patch-import warning；这不是 GPU Megatron 集成验证。

工具不写虚构的 `topology.json`、`timings.jsonl` 或真实 `route_loads.npz`；只有取得实际输入后才输出这些统计。H/U 为 int64 计数，均不称为通信字节或实测耗时。模型版本/layout 版本必须由 run manifest 冻结；CPU 工具检查声明是否一致，不能自行证明 checkpoint 权重相同。

## 下次最短路径

1. 复用已完成的 `routepack-capture-20260912-02` 原始 routes；需要新增 workload 时使用新的 run ID 采集。当前 8 条样本只有 4 个不同 route hash，不构成 10–20 个独立 global batch 的负载覆盖。
2. 在后续固定真实 rollout 的 aligned training run 中使用 `routepack.runtime_topology.aligned_before_log_prob`（worker `PYTHONPATH` 前置本项目 `slime/tools`），并设置独立 `ROUTEPACK_PROFILE_DIR`；合并全部训练 rank 导出，检验简化拓扑范围。
3. 启用 R3 后，按 sample/token/layer 校验实际 router/dispatch ID，同时复核原 schedule 的 exact logprob；尾端和 padding 单列。该 gate 通过前不运行调度干预实验。
4. 将实际 `build_dp_schedule` 输出转换为 `schedule.json`，再计算样本和 collective H/U。由真实多批 workload 的耗时/干预结果决定 Go/No-Go；当前小型旧 rollout 不具备这个证据。

本轮 gate 状态：legacy trace 可用性 **未通过（0/8）**；CPU 数据契约 **通过**；原始 route 采集与结构/hash 校验 **通过（8/8）**；B1 runtime topology、alignment+R3、实际 dispatch、多批 B2 workload/timing **未运行**。这表示先补数据和集成验证，不能解读为 RoutePack 已证明无收益。

## GPU 采集发现的集成缺陷与修复

`routepack-capture-20260912-01` 已实际执行生成，但在保存前被严格校验拒绝：`Duplicate logical expert IDs in topk`，进程返回 1；未输出可被误用的 `captured` 数据目录。运行目录保留日志、参数和失败状态。

根因经源码和 CPU spy 双重确认：Qwen3 的确定性 `_select_experts` 手工计算 top-k/softmax 后直接返回 `StandardTopKOutput`，绕过普通 TopK 路径的 `capture_routed_experts_if_allowed`。因此开启 server/request 的 routes 开关仍不会在该模型分支写入 route cache。缓存初始化为零，这与实测重复 expert 的拒绝原因一致；未保存失败响应，故不把“全部输出均为零”表述为已逐元素验证的结果。

在确认 01 完全退出后，最小补丁增加通用 capture helper 的调用：将返回 IDs 转为 int32，交给 helper，再原样返回同一 Tensor。未修改 top-k 顺序、softmax 数学或 expert dispatch。沿用通用 helper 的 draft opt-out 与无 capturer 时 no-op 行为。可审阅补丁：`qwen3_capture_fix.patch`。

修复前 CPU spy：4 passed + 1 strict xfailed，缺失的 capture 回调被复现。修复后：**5 passed**，验证每层只捕获一次、捕获与返回 IDs 是同一 int32 Tensor、FP32 权重与旧公式逐元素相同、draft opt-out、无 capturer 和非确定性分支。这些测试验证 CPU 方法行为；GPU 原始 route 采集现已通过，后续训练 B1 仍未运行。

Launcher 已增加严格校验的 `--run-id`，便于保留失败 01 并用 `routepack-capture-20260912-02` 复验。新 source hash 覆盖 Qwen 模型、TopK、route capturer/base cache 与采集脚本。01 运行期间 launcher 源码增加了参数入口，故不依据其 traceback 的当前行号解释根因；使用被冻结的源 hash、Qwen 补丁和函数调用链作为依据。


## 修复后的 GPU 复验完成

`routepack-capture-20260912-02` 成功退出（returncode=0），[实际复核](capture02_verified/verification.json)确认：

- 8/8 条样本，48 层、top-k=8；expert ID 在 0–127 内，每 token/layer 的 top-k ID 唯一。
- 1218 prompt tokens + 128 response tokens = 1346 input tokens，原始 trace 共 1338 行；每条缺最后一行符合 tokens−1 契约。
- 原始 trace 共 2,055,168 bytes、513,792 expert assignments，所有 shape/hash 校验通过。
- 保存了真实 `per_expert_counts.npz`，axes 为 sample/layer/logical_expert；未假设 EP owner，未混入合成末端或布局 padding。
- `samples_verified.jsonl` 用明确 run ID 限定采集局部样本 ID，保留原始身份映射；未生成训练 schedule 或奖励。

该结果证明对齐推理可以实际采集合法原始 routes，不能替代训练端实际 dispatch ID 比较、模型权重版本匹配或 alignment+Routing Replay exact gate。未测调度收益，也未实现在线 RoutePack。

最后确认节点无 GPU compute 进程，本轮专用临时目录/缓存已清理，日志和失败证据保留，见 `../gpu_runs/p1-investigation-cleanup-20260912.json`。


## 后续 B1 → B2 CPU 汇总入口（已准备）

新增 `tools/routepack/summarize_b1.py`，等待实际 B1 成功后，读取全部 rank 的真实 topology/schedule/gate 和 router/dispatch 数组，以及该 B1 实际读取的 prepared rollout `.pt` 与 `router-exact-off-20260912-02` 的 batch8 capture。源码只做 CPU 汇总，不改 `b1_gate.py` 或 router worker，也不启动 GPU。

汇总会重新校验样本/token/step/rollout 关系、原始路由哈希、运行时末端/padding；逐个复核两阶段的实际 router/dispatch 文件。再按真实 EP owner 计算样本和 collective H/U/per-expert，并与实际 replay dispatch 文件独立统计的结果逐项比较。只有所有条件成立才创建输出目录；不生成虚构耗时或 Go 决定。

`test_summarize_b1.py` **5 passed**，覆盖有效两 rank 反例和 gate/dispatch/token/step 证据被破坏时的拒绝。命令与依赖产物见工具 README 最后一节。这里记录的是分析入口已就绪，实际 B1 运行和汇总结果应在后续实验完成后另行记录。

## 后续并行执行：模型基线与 B1/B2 入口已就绪

实际 `router-exact-off-20260912-02` 完成四组模型采集：batch8、batch3、batch1、variable3，包含 response logprobs。原实现的 batch8 对应样本与 batch3/batch1 的 tokens、routes、FP32 logprob bytes 全部相同，证据为该 run 的 `capture/baseline-batch-comparison.json`。

已将首批 8 条样本构建为 `router-off02-b1-rollout.pt`，完成 token/trace hash、shape、top-k 唯一性、logprob 有限性及 weights-only 读回校验。该文件 reward=0 仅为前向验证占位，不能作为训练奖励。

B1 新增 `b1_gate.py` / `prepare_b1.py`：在真实 Megatron 原生 schedule 上观测有序 router 与实际 DeepEP Buffer.dispatch IDs，分别执行 aligned fallthrough 和 R3 replay forward，要求 logprob bits 完全相同。启动审阅修复了 Ray tracing 子类可能遮蔽方法的问题，改为直接绑定当前 actor；6 项 B1 CPU 回归及实际 Ray tracing CPU 绑定检查通过。统一 launcher 另外要求 8 rank gate.json 全部通过，不能仅由 Ray job 成功判定。

B2 新增 `summarize_b1.py`，5 项 CPU 回归通过。它依赖 B1 实际通过后导出的全部 topology/schedule/router/dispatch 文件，校验固定输入身份并重新计算 collective H/U 与 per-expert counts；不虚构耗时，不给出 Go 结论。

**当前 GPU 阻塞**：另一个 Qwen3-4B 实验分别在 off01、shadow01 启动期间使用该节点，触发模型加载前的显存不均衡检查。off02 利用中间空闲窗口成功；shadow01 未执行到候选矩阵乘，不能解释为数值 exact 失败。snapshot 显示外部服务位于 GPU6/7，详见 `../gpu_runs/router-exact-contention-20260912.json`。已清理本任务失败运行的临时目录与缓存，未停止其他任务。需要连续 8 卡窗口完成 shadow/candidate/B1，再运行 B2 实际汇总。B1 GPU gate 和多批 B2 workload/timing 均尚未完成。


## B1 prepared rollout 的语义边界

当前 B1 输入是为前向集成 gate 构造的 fixture：`prepare_b1.py` 显式生成 `index=i`、`rollout_id=i`、`group_index=i//2`，response loss mask 全为 1，`reward=0` 为未使用占位。它保存真实 Engine 生成的 tokens/routes/logprobs，并在 metadata 中保留 source capture ID，但这些新建的编号、分组和 reward 不能称作原始训练奖励语义。

后续 B2 若从成功 B1 得到 topology/schedule/H/U，能够表述为“这份小型前向 fixture 在原生 scheduler 下的真实执行与路由计数”；不能据此宣称保留了生产 RL/GRPO 分组奖励、覆盖多样生产 workload、通过梯度/参数更新等价或取得性能收益。输出的 `step` 是 native schedule 的分区编号；B1 不执行 optimizer update。

B1 首次启动因 `tools.routepack` 与 Megatron 的 `tools` 包命名冲突，在 init 阶段失败，尚未进入前向；后续使用独立 `routepack` 模块入口重跑。CPU 汇总必须指向实际成功重跑的目录，不能使用失败 01 的不完整产物；目前等待后续 gate 全部通过，不生成任何实际 B2 结论。

## 最终实际验收：B1 通过，B2 计数链通过

`../gpu_runs/routepack-b1-20260912-02/gate-complete.json` 确认 8 rank 全部通过，进程返回 0。普通 aligned 前向、R3 replay 前向、原始 aligned Engine capture 的 128 个 response token logprobs 逐 bit 相同，最大绝对误差 0；所有 router/实际 DeepEP dispatch 路由比较均无差异。R3 覆盖 48 层并消费全部原生 microbatch，optimizer_steps=0。本结论仅涵盖原生 schedule 上的两次前向，不涵盖 backward 或参数更新。

B2 独立汇总见 [validation.json](b1_02_verified/validation.json)：重新读取并校验 1,536 份实际 route 数组、521,664 行 token/layer 观测（含两阶段、两边界），另复核 384 个 logprob 数组值（128 个 response token × aligned/replay/rollout 三份）。真实 dispatch 的 H、U、per-expert 计数与原始 capture 加生产 terminal/padding 规则重建完全一致。产物包括 `topology.json`、`schedule.json`、`samples.jsonl`、`collectives.jsonl`、`route_loads.npz` 和源文件哈希。

运行时真实 EP groups 为 `[0,1,2,3]` 与 `[4,5,6,7]`，expert DP 分别连接对应 owner rank；48 层各 128 experts，EP 内每 rank 持有 32 experts。一个 global batch、每 rank 一个原生 microbatch，形成两个 group/slot collective。

### 小 fixture 的 padding 必须单独解释

实际训练输入 1,346 tokens，其中原始 route 1,338 行、合成 terminal 8 行；pad multiplier=512 导致布局 padding 2,750 行，占 4,096 行总布局的 **67.14%**。统计见 [load-interpretation.json](b1_02_verified/load-interpretation.json)。

| EP group | 输入 tokens | 布局 padding | 仅真实 route 的最大 H peak/mean | 含 terminal/padding 的最大 H peak/mean |
| --- | ---: | ---: | ---: | ---: |
| 0–3 | 725 | 1,323 | 1.7843 | 3.1580 |
| 4–7 | 621 | 1,427 | 1.6070 | 3.2451 |

这里 peak/mean 是每层接收 rank 的 H 最大值除以均值，再取层间最大值；H 是 expert assignments，U 是到目的 rank 的唯一 token 数，均不是通信字节或耗时。生产 padding 的 arange 规则在固定层上集中到部分 owners，显著放大小 fixture 偏斜；不能将它解读为代表性 workload 的天然路由偏斜或 RoutePack 已有收益。

### 下一步

1. 推理侧：router M=1/2 小 tile 已通过独立模型 exact gate；在去掉审计同步/文件输出后做成对计时，再扩展其它 M 和 QKV/output tile。当前仍默认关闭，不宣称整模型提速。
2. RoutePack：使用 10–20 个代表性独立 global batch，采集分层 H/U 与实际耗时，分离真实 token、terminal、padding；之后才做合法 group/slot 重排与配对 pilot。

成功 run 的 launcher 已清理专用缓存，并记录 `Stopped all 7 Ray processes`。最终复核时节点 SSH 返回 `No route to host`，因此不宣称已复核最新 GPU 占用。失败 01 的工作区临时文件已清理，节点 `/dev/shm/routepack-b1-20260912-01` 缓存可能仍留存，需节点恢复后检查；详见 `../gpu_runs/routepack-final-resource-status-20260912.json`。
