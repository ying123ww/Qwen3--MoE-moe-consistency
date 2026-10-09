# Qwen3 MoE Consistency

Qwen3 MoE 训练与推理数值一致性实验项目，包含训练端对齐、推理端算子融合、逐 bit 门禁及性能测量工具。

## 阅读入口

- [第 0 层数值对齐审计](docs/qwen3-layer0-alignment-audit.md)：实现范围与默认开关。
- [Residual + RMSNorm 融合](slime/alignment_a_preparation/P1_RESIDUAL_RMS_PERFORMANCE_20261008.md)：融合实现、归约布局修复、有限输入数值门禁及性能对照。
- [Router 小 M tile 对照](slime/alignment_a_preparation/P1_ROUTER_TILE_PERFORMANCE_20261007.md)：数值门禁与完整推理计时。
- [仓库组织与 Git 约定](docs/GIT_WORKFLOW.md)：版本控制范围、提交与检查流程。
- [依赖来源](docs/dependencies.json)：保留的依赖源码与用途。

融合候选默认关闭。启用条件、CUDA graph 状态和已验证范围以对应报告为准；单算子收益与完整推理收益分别记录。

## 仓库结构

| 路径 | 用途 |
|---|---|
| `slime/slime/` | 训练、rollout、跨框架数值对齐集成 |
| `slime/tools/routepack/` | 路由捕获、排列、校验及 profiling 工具 |
| `slime/alignment_a_preparation/gpu_tools/` | 可复跑的门禁、算子基准、性能与测量控制测试 |
| `experiments/sglang-main-sm80/` | 实验版 SGLang；当前 Qwen3 router 与 RMSNorm 融合运行时 |
| `sglang/` | 另一版本的推理依赖源码及项目修改 |
| `Megatron-LM/` | 训练依赖源码及项目修改 |
| `DeepEP/`、`experiments/torch_memory_saver/` | 通信与显存管理依赖源码 |
| `experiments/Qwen3-30B-A3B-2layer/` | 两层模型配置和来源说明 |
| `docs/` | 项目审计、依赖说明及 Git 约定 |
| `scripts/check_repo_hygiene.py` | 检查暂存区的文件范围、对象大小及意外 gitlink |

依赖以普通源码文件保存，保留许可证和必要的上游文件；安装方式见各依赖 README。两个 SGLang 目录服务于不同代码版本，运行融合实验时请使用 `experiments/sglang-main-sm80/python` 中的实现，并按启动脚本配置环境。

## 数据与实验产物

模型权重、tokenizer、数据集、原始运行输出、trace、逐调用审计、临时源码快照、PID/排队状态和本机工具配置在本地维护，不随项目源码提交。克隆后需要按依赖 README 和实验脚本准备模型与数据路径。

技术报告中的原始证据路径指向本地实验产物；完成的报告、实验设计、可复跑脚本和测试纳入版本控制。自动更新的排队报告不纳入 Git。

## 提交前检查

先按功能选择文件暂存，再运行：

```bash
python3 scripts/check_repo_hygiene.py
git diff --cached --stat
git -c core.whitespace=-blank-at-eof diff --cached --check
```

GitHub 的 `Repository hygiene` 工作流会检查已提交文件。详细提交规则与本次融合的 CPU 测试命令见 [Git 约定](docs/GIT_WORKFLOW.md)。
