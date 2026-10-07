# 启动工具修复（2026-09-09）

本次只修改我们新增的 gpu_tools，不改模型、依赖仓库或历史 v89。未启动新 GPU 作业，未清理或重启远端集群。

- 启动前读取 CPU affinity、cgroup v1/v2 本级及父级 CPU/内存限制、主机 MemAvailable、PSI、临时盘及 /dev/shm 容量，写入 resource-preflight.json。CPU 数取 affinity 与可读 quota 的较小值。
- 默认要求可用主机/cgroup 内存至少 512 GiB、临时盘至少 20 GiB、/dev/shm 至少 8 GiB、有效 CPU 至少 8；内存限制无法读取则停止。这些是保守启动门槛，并非实测内存峰值或不会 OOM 的保证。参数允许显式调整，需先依据节点配额及后续采样校准，不能为了绕过失败盲目降低。
- Ray 新集群显式设置 num-cpus，object store 固定为 4 GiB，避免默认按整机资源扩张。4 GiB 是小样本 A 实验的起点，尚未 GPU 验证；若需要更大对象空间，先测量再调整。
- 删除捕获任意异常后自动 ray start 的行为。新建必须带 --start-cluster，已有临时目录或端口占用则拒绝。旧集群没有新版资源记录则拒绝复用，CPU allocation 不同也拒绝复用。不会自动删除旧目录或停止集群。
- 启动器、Ray 新进程和 worker 均清空继承代理；status.py 直接禁用 HTTP 代理。
- 节点内文件锁覆盖检查至提交，避免我们两个启动器并发提交；保留 GPU 占用与活动作业检查。
- nvidia-smi 超时 15 秒；Ray HTTP 连接/读取超时 5/15 秒；ray start 超时 120 秒。启动命令超时不能证明其子进程均退出，仍必须检查残留。
- 提交前写 submission-uncertain.json，只有提交响应及本地记录均成功后移除。遇到超时或异常，后续提交会被阻止；必须先查询该 submission_id 的实际状态。
- --preflight-only 只做资源检查、保存结果，不初始化 Ray 或提交 GPU 作业。

## 验证与剩余事项

CPU unittest 覆盖低磁盘中止、未知内存限制、父级 cgroup 限额、v1 配额，以及现有 probe/汇总逻辑。测试设置 CUDA_VISIBLE_DEVICES 为空。启动脚本和状态工具通过语法检查。

这些修复不会改变已经运行的 a-restore-20260909-02 或旧 Ray。恢复节点后仍需确认该作业状态，按所属范围结束旧作业/集群，再处理临时盘容量并使用新版启动流程。当前未确认 GPU 释放。运行中的分阶段资源采样、共享存储性能与 NUMA affinity 仍需后续现场检查；没有宣称这些问题已经修好。
