# 在线 RL 端到端计时：短程实测（2026-09-10）

状态：两组在线运行及汇总校验均完成。**本轮总耗时中位数46.36→52.40秒（+13.03%），均值比+20.75%；仅短程pilot，不能作为稳态开销结论。**沿用已有在线恢复路径，先完成一组短程 aligned/configured-baseline 对照，每种配置独立启动一次，首轮预热，随后五轮计时；lr=1e-6、StatelessAdam、reset optimizer states、random reward。不是任务学习或标准有状态 Adam 长期稳定性实验。

## 范围

驱动端单调时钟从 `rollout_manager.generate` 调用前开始，到训练更新、全量权重同步以及阻塞 `onload_kv` 返回后结束。各 RPC 显式等待，`actor_model.update_weights()` 内部 ray.get 全部 actor 的同步返回。首次模型加载、初始权重同步和最终进程销毁不计入；禁止 evaluation/checkpoint 与离线 replay 模式混入此计时。

分段：rollout/reward/data 转换、rollout offload、training RPC、训练后清理、rollout weights onload、weight sync、rollout KV onload。training RPC 内部包含数据加载、模型 onload/offload、优化器与 CPU 权重备份；不能称为纯 forward/backward，也不能把它与之前 train_one_step 的约 18 秒直接等同。阶段是连续 wall-clock 区间，总和应与总时间一致。

## 对照

- 两组相同初始 checkpoint、seed/数据源、模型结构、8×A100、并行拓扑、Slime 在线 HTTP rollout 路径和全量 NCCL 权重同步。
- aligned：保留历史 Qwen3 对齐 hooks、Megatron deterministic mode、SGLang deterministic inference、unified decode、固定 DeepEP 参数。
- configured baseline：移除 Qwen3 hooks/layers、关闭 MEGATRON_USE_SGLANG_*、移除 Megatron/SGLang deterministic 开关、unified decode=0、SGLang DeepEP 默认配置；移除 CUBLAS_WORKSPACE_CONFIG 与 NVTE_ALLOW_NONDETERMINISTIC_ALGO 显式约束。
- 两组仍共享修改过的依赖栈、BF16、Triton/DeepEP normal、CPU offload、StatelessAdam、NCCL_ALGO 初值 Ring 等条件。不是 pristine upstream 或生产峰值 baseline，也不同于此前保留 deterministic mode 的训练-only baseline。
- 正常在线采样，最大响应 128 tokens，不强制两模型生成相同内容。每轮记录 prompt 哈希、样本数、实际 response/training tokens；真实更新后两组权重/轨迹可能分叉，因此是短程在线系统对照，不是严格固定工作量的 kernel 因果测量。
- 不做大 tensor dump；只保留小型计时与 workload JSON。有限 logprob 检查不等于本轮新做的训推 bitwise exact gate；此前两轮正确性记录独立引用。

## 资源与验证

继续使用 ai_hgx_03、Slurm 999484（1T/64CPU/8GPU）及900 GiB内存守护。此前 Wan21 任务已由用户释放，重新确认空 GPU 后才提交。TVM-FFI JIT 缓存置于节点 /dev/shm。

新增计时单测验证连续时间相加、写记录在计时外、离线模式拒绝误标为 e2e；连同既有 launcher/资源/测量测试，共14项CPU测试通过。GPU 结果已验收：两组均 SUCCEEDED，12 条完整迭代记录和 workload 记录齐全；各分段时间之和等于整轮时间。

## 结果：计时已补齐，稳定开销仍需更长重复

真实在线循环已完成：每种配置独立启动一次，运行 6 轮，排除首轮后比较 5 轮。

- 正式迭代中位数：baseline **46.360 秒**，aligned **52.399 秒**，本轮比值增幅 **13.03%**。
- 正式迭代范围：baseline **43.643–54.026 秒**，aligned **51.401–70.678 秒**。
- 正式迭代均值：baseline **47.197 秒**，aligned **56.988 秒**，均值比增幅 **20.75%**。与中位数比不同，反映早期较慢轮次的影响，不能择优混用。
- 每种配置正式部分均为 **5120 response tokens、10590 training input tokens**；所有对应轮次 prompt SHA256 和 response 长度一致。模型更新轨迹和生成内容不要求一致；相同 token 数不等于逐 kernel 相同负载。
- 按正式部分累计输出 token/累计在线时间计算：baseline **21.696**、aligned **17.969 response tokens/s**。这是含训练和同步的系统速率，不能当作纯 SGLang 生成吞吐。

**不能把 13.03% 宣称为已确认的稳态总开销。** 当前每种配置只有 1 个独立运行，formal 早期仍偏慢，特别是 aligned rollout 2/3。新的 prompt 长度、执行形状、编译缓存或系统波动的贡献没有单独定位。此处按预先定义排除首轮，没有事后删掉较慢正式数据。需要更长预热、更多在线轮次与交错独立重复后，才适合用单一百分比概括稳定开销。

### 逐轮时间

| rollout ID | 用途 | baseline（秒） | aligned（秒） |
|---|---|---:|---:|
| 1 | 预热 | 72.757 | 74.923 |
| 2 | 正式 | 54.026 | 70.678 |
| 3 | 正式 | 44.918 | 58.980 |
| 4 | 正式 | 47.040 | 51.485 |
| 5 | 正式 | 46.360 | 52.399 |
| 6 | 正式 | 43.643 | 51.401 |

### 分段平均时间（只统计五轮正式数据）

| 阶段 | baseline（秒） | aligned（秒） |
|---|---:|---:|
| 生成、reward、数据处理 | 14.030 | 21.446 |
| 训练 RPC（含内部搬运与备份） | 26.016 | 28.536 |
| 全量权重同步 | 5.923 | 6.042 |
| 其他切换与清理 | 1.228 | 0.965 |
| 总计均值 | 47.197 | 56.988 |

分段采用均值，保证可以相加；不要拿这些均值去凑总中位数。当前均值差主要体现在生成/reward/data 区间，其次是 training RPC；尚不能据此归因到某个 kernel。training RPC 包含 onload/offload 和 CPU 权重备份，口径大于原有 train_one_step。

### 更新与正确性边界

两组每轮均调用 optimizer 并完成全量权重同步，lr=1e-6。aligned 六轮中 5 轮梯度非零、native 六轮全部非零。aligned rollout 5 的四个 prompt 组内随机奖励均相同（zero_std count_0=1、count_1=3），advantage/loss/grad_norm 均为零，所以不能写成六次非零参数更新。

本轮使用 random reward 与 StatelessAdam/reset optimizer states；它验证在线系统执行和计时，不验证任务学习收益或标准 Adam 长期稳定性。未保存逐 token train dump，也未重新做全量 bitwise comparator，不能把有限 rollout logprob 检查或聚合训练指标当作新的六轮精确一致性验收。

## 验收与产物

- `gpu_runs/a-online-aligned-20260910-01/`、`gpu_runs/a-online-native-20260910-01/`：命令、环境、状态、日志、`iterations.jsonl`、`workload.jsonl`、`online-summary.json`。
- `gpu_runs/online-comparison-20260910.json` 与 `.csv`：正式统计和原始逐轮对应关系。
- `gpu_runs/online-source-hashes-20260910.json`：计时代码版本哈希。
- `gpu_tools/summarize_online.py` 核验成功：两任务成功、每次 6 轮、首轮 warmup、所有耗时有限、分段和正确、在线参数和学习率正确、梯度有限且存在非零更新、对应输入哈希和长度一致。
- GPU 结束检查无计算进程；900 GiB guard 未触发。Slurm 999484 与守护进程保留。采样内存约 790–795 GiB，不宣称这是完整历史精确峰值。

## 如何衔接既有报告

此前训练 +2.34%、独立推理 +23.48%，与本次在线短程中位数比 +13.03% 使用不同计时边界及 baseline，不应加和或声称相互复现。现在已经完成端到端计时接入和一组实际测量；剩余缺口是**稳态及独立重复证据**，不是再实现一次计时。

下一步如继续完善性能结论，先固定这份协议，延长在线预热与测量并交错重复；保留所有逐轮数据。若研究优化，再对生成和训练内部做少量 profiler 归因，正式性能数字仍使用未开启 profiler 的运行。
