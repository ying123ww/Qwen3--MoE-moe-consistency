# Qwen3 MoE Consistency

Qwen3 MoE 训练与推理数值一致性实验项目，包含 Megatron、SGLang、slime 和 DeepEP 的当前源码及本地修改。

建议从 [Qwen3 第 0 层数值对齐与默认开关审计](docs/qwen3-layer0-alignment-audit.md) 开始阅读。该文档说明对齐实现、启用条件、默认开关以及历史实验结果的适用范围。

## 目录

- `slime/`：训练与 rollout 集成、Qwen3 对齐适配、RoutePack 工具及实验记录。
- `Megatron-LM/`：训练端源码和本地修改。
- `sglang/`：推理端源码和本地修改。
- `DeepEP/`：专家并行通信依赖源码。
- `experiments/`：两层 Qwen3 配置、启动脚本、实验版 SGLang 和 torch_memory_saver。
- `docs/`：项目审计和 [源码快照来源记录](docs/source-snapshot.json)。

## 使用说明

本仓库将依赖目录保存为普通源码文件，克隆即可获得本地修改，无需初始化 Git 子模块。各依赖的环境配置和安装方式参见相应目录的 README。

该快照保留源码、配置、文档、tokenizer 文件及小型实验结果；模型权重、分布式检查点、tensor dump、构建产物、缓存、运行日志和原始性能 trace 在本地保留，不随源码上传。运行实验前，需要准备模型权重，并按实际环境调整脚本中的模型、数据和依赖路径。

普通运行 Qwen3 不会自动启用完整的数值对齐路径；具体开关和验证范围见上述审计文档。

## 依赖来源与许可证

各依赖目录保留其原有许可证。源码快照来源、基础提交和本地修改清单见 [source-snapshot.json](docs/source-snapshot.json)。本次上传保存当前文件状态，不包含这些依赖各自的 Git 历史。
