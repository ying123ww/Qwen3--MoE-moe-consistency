# P13：真实更新后的 Full Graph 生命周期回归

**P13 完成。** 真实 Slime/Megatron 在线生成与训练的 version 0、1、2 各
1024 个 target logprob 全部逐 bit 一致。8 个 rank 都完成两轮非零
StatelessAdam 更新；全量同步后保留的旧图读取新值，与重新捕获的结果一致。
sleep/wake 后参数、KV 和 graph pool 身份经核验，地址替换负例拒绝旧图，
重建后输出恢复一致。这是既定模型/拓扑的有限回归，不是长期训练验证。

## 实际配置和奖励协议

运行：`gpu_runs/p13-full-online-hgx03-20261012-06`，ai_hgx_03，8×A100，
Slurm1031869。Qwen3-30B-A3B BF16；训练 TP1/EP4，推理 TP8/DP2/CP2/EP4/
MoE DP2，DeepEP normal，固定接收容量16，decode native full、buckets2/4，
prefill eager。采用 P12 成功组合：QKV/O 小tile、残差RMS、联合Q/K、Router小tile。

保留 v89 的 lr=1e-6、StatelessAdam 和每步 reset states，不称为标准有状态
Adam。实际执行3次自然生成、前后向和优化器更新；version2 是前两次更新后的
权重。最后还有第三次更新，但未生成 version3，不能宣称第三次更新后的训练/
推理 logprob 也已验收。每次8个样本、各128 response tokens，强制 ignore_eos；
没有teacher-forcing生成或用synthetic参数修改代替真实训练。

原random reward曾使四组奖励分别为[1,1]、[0,0]、[1,1]、[0,0]，GRPO组内
advantage全为0，实际梯度和master delta均0，严格非零门禁拒绝该次运行。
成功回归使用明确的诊断奖励：每两样本组为0/1，原GRPO流程产生非零advantage。
这改变了奖励负载，不是原随机reward复现结果，也不是任务学习或收敛实验。

## 同步路径以运行和源码为准

CLI保留 `--update-weight-mode full --update-weight-transport nccl --colocate`。
实际actor分支选择 `UpdateWeightFromTensor`，包含训练侧NCCL参数收集、
Gloo/IPC及flattened-bucket向SGLang投递；不是最终向推理worker纯NCCL广播。
源码见 `slime/backends/megatron_utils/actor.py` 的colocate分支，以及
`update_weight/update_weight_from_tensor.py`。保持原v89拓扑，没有为使测试通过
更换为另一种训练/同步路径。

每次完整同步有2个非专家bucket和48个专家bucket；每rank共200次成功worker
更新调用（初始+三次更新后同步）。供本回归使用的旧图回放分别发生在50、100、
150次更新调用之后。地址替换负例另外递增一次ledger版本，因此最后恢复后的
同步计数记录201，并非多了一次权重同步bucket。

## 实测数值门禁与非零更新

| 推理权重版本 | 对应训练target tokens | logprob bit mismatch | 更新前后旧图/重捕获 |
|---|---:|---:|---|
| v0，初始全量同步 | 1024 | 0 | tokens和128个greedy probe LP bits一致 |
| v1，真实更新0之后 | 1024 | 0 | 同上；旧图使用新权重 |
| v2，真实更新1之后 | 1024 | 0 | 同上；地址替换恢复后也一致 |

独立gate检查实际训练使用对应自然rollout的完整token序列，8个唯一训练位置，
每样本128个有限FP32 logprob，再以int32视图比较；不是只比较浮点最大差异。
训练数据和rollout数据均直接保存，三版本共3072 targets逐bit一致。

| 真实optimizer step | 所有rank的global gradient norm | 每rank抽样FP32 master变化元素范围 |
|---|---:|---:|
| rollout0 / step0 | 10.8996054622 | 47250–69715 |
| rollout1 / step0 | 11.3917059994 | 44490–65662 |
| rollout2 / step0，额外观察 | 15.0439693700 | 54895–76473 |

变化元素来自每个实际CPU master参数的前64个系数，是非零更新见证，不能
当作所有训练权重的完整checksum。实际调用原optimizer.step，检查成功、有限
非零梯度与master变化；没有手动写入训练参数。v1/v2的greedy tokens/LP记录
均与前一版本不同，进一步证明新权重对实际推理输出产生影响。

## 地址、图身份、sleep/wake及负例

每rank记录全部435个named weight parameters（含LM head）、96个KV张量和
request-to-token张量的data pointer、storage pointer、offset、shape、stride、
dtype、device。捕获和更新记录完整地址digest；最后保存的两bucket快照与
最终捕获digest一致。实际所有同步和wake后的地址变化均为空。

独立验收器逐事件重建各bucket的活动graph：更新后first replay的graph object
ID与pool ID仍等于更新前捕获的对象，而不是先重捕获再冒称旧图读取新值。
之后才请求重新捕获，同权重下8请求×16 token的greedy tokens/LP bits相同。
这既有图身份的机械证据，也有真实输出改变和参考比较。

实际weights/KV/cuda_graph release和分阶段resume都经过记录。默认不启用
weights CPU backup，weights wake之后先完成全量权重恢复，再wake KV；未恢复
值或处于offload时的replay被门禁拒绝。每个完整wake独立核验50个成功同步
bucket。这是本实验完整同步协议的验证，不保证任意partial-update调用恢复了
全模型值。

v2负例逐rank将layer0 input norm的storage替换为数值相同的clone，并保留旧
tensor，实际检测1个参数地址改变。调用与replay入口相同的指针门禁拒绝旧图，
避免执行已失效图；随后清理backend、重新捕获两bucket，恢复输出逐bit相同。
每rank共10次bucket捕获：初始、v0/v1/v2参考及负例恢复，各2个bucket。
不得由此外推任意量化、跨节点、所有offload机制或标准有状态Adam。

## 启动问题和资源边界

checkpoint并发读取曾超过900GiB guard；成功运行对原DCP FileSystemReader
读取加任务局部flock，等待原Future完成并释放CPU临时分配，不改变checkpoint
张量或训练数学。8个真实reader进程记录读取；成功运行cgroup观测峰值
886.210GiB，阈值900GiB。这个值包含任务cgroup的内存及cache，不是GPU显存。

另一次运行在初始同步后因router `503 no_available_workers`失败，未记为成功。
注册接口返回accepted后激活仍异步；成功入口等待实际目标worker出现且healthy、
router readiness=200后才允许offload。本轮28次观测、27.263秒，从空worker/503
到1个目标worker/200。它证明等待覆盖实际异步间隙；没有把前次失败的所有根因
假定为已独立定位。正式成功进程及私有Ray cluster已清理，GPU空闲已核验。

## 保存证据、发布源码与复跑

成功运行目录保存 `manifest.json`、argv/runtime env、source SHA、原始job日志、
checkpoint内存、实际rollout/train三版本、optimizer witnesses、逐rank lifecycle、
greedy比较、`target-logprob-bit-gate.json` 和
`independent-online-lifecycle-validation.json`。原始权重/数据和失败探针不进入Git。

发布到 `gpu_tools/p13_online/`：14个runtime/测试文件与成功源码字节相同；
两份配置/launcher只调整P12依赖到已发布目录和更正同步文字。实际argv与全部
forward hooks逐项相同，映射SHA记录在 `published-source-validation.json`。
发布版独立生命周期验收器重验实际GPU数据通过，13项CPU机制测试通过；
这些CPU测试不替代上述GPU回归。仅启动器路径适配没有额外重复GPU计时。

从仓库根目录，在已分配8GPU、GPU空闲的ai_hgx_03和相同slime环境执行：

```bash
python slime/alignment_a_preparation/gpu_tools/p13_online/run_online.py \
  --run-id NEW_UNIQUE_RUN_ID \
  --timing-run slime/alignment_a_preparation/gpu_runs/p12-full-timing-hgx03-20261012-01
python slime/alignment_a_preparation/gpu_tools/p13_online/validate_lifecycle.py \
  --run slime/alignment_a_preparation/gpu_runs/NEW_UNIQUE_RUN_ID
```

入口依赖本地checkpoint、原v89 argv/runtime快照和P12已验收结果，不是无数据
通用benchmark。新run ID不得覆盖已有证据；私有Ray集群仅停止本任务进程。
用户取消保号的指示继续有效。本阶段不新增端到端性能结论；下一阶段P14另行
验证容量、显存、多batch、零接收/idle、捕获成本及边界行为。
