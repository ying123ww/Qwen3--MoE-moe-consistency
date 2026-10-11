# 非确定性 baseline 的完整 decode CUDA Graph

baseline 的整个 decode model forward 已成功使用原生 full CUDA Graph 捕获和回放。
本轮 20 批正式 generate 的中位数为 **1.890481 秒**，比重新测量的最佳 baseline
Breakable＋固定容量＋专家 Graph 的 **3.093301 秒**减少 **38.885%**；输出吞吐
331.04→541.66 tokens/s，增加 **63.625%**。Prefill 保持 eager。

## 测量

| baseline decode 配置 | 正式批数 | 中位数秒 | P95 秒 | 输出 tokens/s |
|---|---:|---:|---:|---:|
| Breakable＋固定容量 dispatch＋专家 Graph | 20 | 3.093301 | 3.215569 | 331.04 |
| 固定容量 dispatch＋原生 full Graph | 20 | 1.890481 | 1.984975 | 541.66 |

同一 ai_hgx_03，8×A100-SXM4-80GB，Slurm 1031236，SSH step_extern；模型
Qwen3-30B-A3B BF16，TP8/DP2/attention CP2/EP4/MoE DP2，Triton attention/MoE，
DeepEP normal。两组均 `enable_deterministic_inference=False`、unified decode=0、
RMS/QK/router 候选关闭，使用 baseline 原有 DeepEP 配置选择。

每组独立 Engine，先额外进行不计时的 3-prompt、2-token 小 bucket 预热，再
3 批标准预热＋20 批正式。固定 8 prompts、1218 输入 tokens、每请求128输出
tokens，共1024输出 tokens；temperature=0.6、ignore_eos=True，返回 logprob。
范围为 blocking `Engine.generate`，包含调度、prefill、decode、采样、logprob、
detokenization/IPC，排除模型加载、编译/捕获、预热、cache flush、训练和权重同步。

总计40正式批、46标准生成样本；全部零缓存命中，无profiler、数值shadow或
逐调用文件I/O。源码SHA256保持不变，所有worker退出码0。两组resolved配置
仅 `cuda_graph_config` 和 `forward_hooks` 不同；timing的prefill请求分组均保留默认。
捕获和首次回放证据均在正式测量前写完；Breakable的部分bucket4记录在第一批
标准预热产生，full记录在额外预热结束前完成，未宣称全部记录都在Engine返回前。

这是每组一次顺序启动、20个批内样本，不是多轮交错独立重复。38.885%是本轮
观察值，未采集新的profiler，不能把收益精确拆成通信等待、图切换或算子发射占比。
原P8组合3.110243秒与本轮3.093301秒接近；主比较使用本轮参照，不混用旧结果。

## Graph 覆盖范围

| 部分 | 本次 full |
|---|---|
| Decode Attention、原生 RMS/QK、RoPE/KV写入 | 捕获 |
| Gate/TopK、DeepEP normal dispatch | 捕获 |
| Padding识别及无效X/weights清零 | 捕获 |
| Triton expert core、本地专家归约、combine | 捕获 |
| 后续decoder层及LM head | 同一个forward图内 |
| Metadata/输入准备、调度、sampling、输出处理 | 图外 |
| Prefill | Eager |

每rank/bucket用 `FullCudaGraphBackend` 的一个原生图执行整个decode forward。
没有49段/48个MoE eager break，也没有嵌套独立专家Graph或breakable fallback。
“full”不表示整个请求或在线训练iteration都是一个图。

## 实现

复用已经验证的 `fixed_capacity_dispatch_hooks.py`，仍调用安装DeepEP的
Normal/intranode `Buffer.dispatch(num_worst_tokens=16)`。它跳过CPU动态接收数量
等待，并固定接收Tensor形状；真实路由、GPU接收布局与通信handle每次动态产生。
未修改DeepEP底层通信，也未切换low-latency。填充ID为−1，X/weights设备侧清零。
容量16对应EP4、每rank decode输入≤4行的保守上界，超出配置范围报错，prefill
继续普通动态dispatch。当前仍有限定模型/dtype/拓扑的实验实现。

`fixed_capacity_experiment.allow_full_configuration`新增明确的deterministic参数，
默认True保留此前确定性实验约束；baseline adapter显式传False。仅实验入口恢复
normal配置层主动禁用的full后端，生产默认没有自动开启full。

新增 `baseline_full_experiment.py`、`baseline_full_model_gate.py`、
`baseline_full_inference_benchmark.py`、`run_baseline_full_pipeline.py`，保留baseline
原生norm和通信配置选择，不启用旧确定性融合。所有需要替换算子的hook在捕获前安装。

## 正确性与回放证据

先重新生成原baseline Breakable参考，再验证fixed eager shadow和full，均在
正确性gate统一 `prefill_max_requests=1`，避免非确定性baseline的prefill grouping
变化混入同输入比较。正式计时没有使用该限制，保持 `prefill_max_requests=None`。

- Fixed eager shadow：8 ranks合计21504次、171687936个有效BF16元素，零bit差异；
  包含1085次本地零接收。对照把有效行取出、按原actual-M core独立计算。
- Fixed eager和full分别与原参考比较batch8、batch3、batch1、variable3，每请求
  greedy生成16tokens；完整tokens、专家routes、response logprob bits均一致。
- Full门禁及正式Engine各有16份捕获记录和16份实际成功回放记录，覆盖
  8 ranks×bucket2/4；每次捕获48个dispatch和全部48个expert layer。
- 实际backend均为 `FullCudaGraphBackend`；不是只核对一个配置开关。

baseline没有跨batch不变性承诺，本轮不要求batch8与batch3/1的同行bit一致。
自然调度两种计时模式的23对采样输出哈希有17对一致，因此不宣称正式生成输出
全部bit相同。受控gate验证数值路径；自然计时固定输入/输出长度，token身份和
MoE路由不完全固定。没有进行跨框架训练、权重更新/offload生命周期或其它拓扑验证。

此前确定性full为3.268245秒，属于旧一轮。不要把它与本轮1.890481秒直接计算
“已确认的确定性额外开销”；需要同轮同协议重复两种full配置才能给出该结论。

## 证据、Git 与交接

运行目录：`gpu_runs/baseline-full-hgx03-20261012-01/`，相对本报告。

- `results.json`、两模式 `summary.json`/`measurements.jsonl`：结果和逐批记录。
- `measurement-validation.json`：40正式批、缓存、source锁、输出哈希、配置及证据时间检查。
- `server-arg-differences.json`：完整resolved配置差异，仅图配置和hooks。
- `compare-eager.json`、`compare-full.json`、`fixed-shadow-summary.json`：正确性。
- `exact-full/graph-evidence/` 与 `full/graph-evidence/`：每rank/bucket捕获及回放。
- `source-snapshots/`、`manifest.json`：本次实际源码及SHA256。

上一步baseline Breakable工具/P8原提交为 `0a3a91cd`，本次full实现/P9/总览原提交
为 `195fb228`；均已推送。按用户最新要求，成功内容直接选择合入main，分别为
`3788f6fe` 和 `adafc97a`。失败的普通Normal全图探针及P4独立报告未合入main。
原始运行结果按约定留本地、不纳入Git；后续工作直接使用main。

Driver实际PID/PGID127310及全部worker已退出，最后GPU compute列表为空，
8卡均1MiB、0%利用率。Slurm1031236保留RUNNING，按用户此前要求不恢复保号。
日志：项目 `.a-tmp/baseline-full-hgx03-20261012-01-driver.log`。
整个目录总结见 [WORK_SUMMARY_20261012.md](/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/WORK_SUMMARY_20261012.md)。
