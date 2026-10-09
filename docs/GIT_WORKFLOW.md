# 仓库组织与 Git 约定

本仓库管理项目源码、可复跑的实验工具、测试、配置、许可证和完成的技术报告。源码目录保持现有路径，便于已验证的启动脚本与正在排队的实验继续使用。

## 文件范围

| 文件类别 | 管理方式 |
|---|---|
| 对齐与融合实现、门禁、基准、CPU 测试 | 按功能提交 |
| 修改过的依赖源码、许可证、安装配置及必要上游文件 | 在现有目录中保留；用途见 dependencies.json |
| 已完成的技术报告、实验设计与配置模板 | 纳入 Git，说明证据范围和复跑入口 |
| gpu_runs、cpu_runs、cpu_validation、routepack 运行结果及源码快照 | 本地保留，整个目录忽略 |
| 数据集、tokenizer、模型权重和检查点索引 | 从外部准备，Git 只保留模型配置及来源说明 |
| 代理工具目录、临时脚本、本机 Conda 备份、PID、节点和队列状态 | 本地保留，不纳入 Git |

顶层 `AGENTS.md` 是当前集群的保号约定，在本地保留并继续生效；公共仓库不携带这一机器专属配置。依赖原有的文档和功能测试属于其源码分发内容，不能按文件大小或扩展名随意删除。

## 提交方法

使用明确路径暂存关联文件，并检查范围：

```bash
git add -- path/to/implementation.py path/to/test.py
python3 scripts/check_repo_hygiene.py
git diff --cached --name-status
git diff --cached --stat
git -c core.whitespace=-blank-at-eof diff --cached --check
```

融合代码、对应测试与验证说明作为一个功能提交；仓库清理使用独立的 `chore(repo)` 提交。建议使用 `feat(alignment)`、`test(alignment)`、`docs` 等具体前缀，提交说明记录相关验证和仍未覆盖的范围。

已有 GPU 实验按源码 SHA 锁定运行时和测量实现。此时保持源文件字节，避免为了格式调整而使已排队实验的校验失败；本次只取消无关文件的跟踪并整理文档。

`.repo-policy.json` 定义禁止跟踪的路径、文件名和单文件 10 MiB 上限；检查脚本检查 Git 索引中的实际 blob，而非仅依赖 `.gitignore`。即使文件被强制暂存，CI 仍会检查。确有必要的资源应在提交前明确调整策略并解释用途。

## 本次融合的 CPU 验证

在包含 PyTorch、Triton 和 pytest 的环境中，从仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/slime/alignment_a_preparation/gpu_tools${PYTHONPATH:+:$PYTHONPATH}" \
python -m pytest -q -p no:cacheprovider \
  slime/alignment_a_preparation/gpu_tools/test_router_tile_exact.py \
  slime/alignment_a_preparation/gpu_tools/test_residual_rms.py \
  slime/alignment_a_preparation/gpu_tools/test_router_tile_performance.py \
  slime/alignment_a_preparation/gpu_tools/test_router_cache_flush.py \
  slime/alignment_a_preparation/gpu_tools/test_remaining_inference_profile.py
```

2026-10-09 整理时上述 44 项测试通过，覆盖 CPU 分派、审计失败、P2P 检查解析、cache-flush 控制与 profiling 测量控制。本次未启动额外 GPU 回归；GPU 数值与性能范围见已完成报告。

## 历史与本地文件

从 Git 移除运行产物时使用 `git rm --cached`，保留本地文件。忽略规则只阻止后续误提交，不会删除已存在于旧提交中的 blob。

本次通过普通提交整理当前分支，保留已有历史。若之后需要减少历史克隆体积，应先确定需要保留的历史与引用，再单独规划历史重写。
