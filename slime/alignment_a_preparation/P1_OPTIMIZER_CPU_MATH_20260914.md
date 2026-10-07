# CPU Adam 数学与 fill_ 诊断（2026-09-14）

## 结论

本轮只运行 CPU 数学/分配实验，没有提交正式 GPU 计算。现有训练 optimizer 未修改。

1. **当前 CPU 复现中 fill_ 的机制已确认**：实际 StatelessAdam 的 abs 输出在确定性算法与未初始化内存填充均开启时发生填充。原实现单参数 profile 有 5 次 fill_，较大的 fill_ 的父调用是 abs；其余父调用为 empty / empty_strided。单独关闭 fill_uninitialized_memory 或 deterministic_algorithms 后均为 0 次。与历史 trace 的 abs 父调用线索吻合，但不能宣称历史 1.5 秒全部来自本次已观测机制。with_stack=True 未取得可用 Python 栈（stack 为空），证据是 profiler 父子算子关系和开关干预。
2. **找到保留确定性设置的局部候选**：将新分配的 abs 输出上的 mul 改为 mul_，保留运算顺序和标量值，减少一个参数大小的临时输出分配及对应写入。它仍有 5 次 fill_，没有消除 abs 的填充。没有用去掉乘以 1、改公式或全局关闭确定性来实现这个候选。
3. **不能由单进程扫描推荐生产线程数**：8 线程原实现出现严重异常，其他配置也有轮间波动。未采集调度/off-CPU、allocator、NUMA 带宽等证据，不能将异常归因为线程超配或内存带宽饱和。

## 协议

- 节点 ai_hgx_05，Intel Xeon Platinum 8358P，torch 2.11.0+cu129；CPU affinity 24–31，interop threads=1，intraop 扫描 1/2/4/8。
- 实际 stateless_adam.py 通过文件导入，候选通过内存中的单行替换建立独立类。每参数一个子 optimizer，模拟 HybridDeviceOptimizer 的 CPU 数学部分。
- experiment01：8 个 FP32 参数，共 72 MiB；experiment02：32 个参数，共 288 MiB。尺寸交替 6/12 MiB，固定 seed 的合成梯度，lr=1e-3。
- 每配置 3 步预热、5 步计时，3 轮交替配置及线程顺序。每个规模为独立 Python 进程，各自三轮在同进程运行。初始化不计时，step 内部的分配计时；profiler 独立于正式计时。
- Wan21 保号同机运行，CPU 核心没有排他预留。没有八 rank CPU 竞争或 H2D，没有历史完整 14.218 GiB/rank 布局和超大参数。

## experiment02 计时

下表为三轮各自 step 中位数再取中位数，单位 ms；保留全部异常，不删除离群值。

| CPU 线程 | 原实现：det+fill | 原实现：det、不填充 | 原实现：非det | mul_候选：det+fill |
|---|---:|---:|---:|---:|
| 1 | 180.187 | 168.720 | 168.297 | 114.278 |
| 2 | 73.756 | 67.488 | 95.464 | 60.411 |
| 4 | 63.589 | 60.277 | 59.883 | 33.646 |
| 8 | 1307.743 | 43.195 | 43.068 | 20.153 |

4 线程原实现三轮为 41.322, 65.525, 63.589 ms，候选为 33.937, 33.646, 32.590 ms；配对耗时下降分别为 17.87%, 48.65%, 48.75%。三轮中位数再取中位数对应下降 47.09%。范围较宽，不能只发布单个比例掩盖波动。

8 线程原实现三轮为 1363.518, 45.362, 1307.743 ms，有秒级异常。不能据此宣称候选获得几十倍普遍加速。关闭填充的因果结论是 fill_ 事件消失；其净耗时收益受规模、分配与同机波动影响，不能将所有配置差值解释成纯 fill_ 时间。

experiment01 4线程原实现 14.689 ms、候选 8.395 ms（同一汇总口径），方向一致；8线程同样有异常。所有逐步时间均保留在原始 JSON。

## 正确性及限制

- 每个规模的 48 个配置完成 8 次更新后的完整 FP32 参数字节哈希全部相同。
- 候选与原实现额外比较 16 种 bias_correction/maximize/AdamW或耦合衰减/weight_decay 分支，每组5次更新；梯度含正负零、极小值、极大值、NaN、正负无穷，均按 int32 位模式比较通过。
- 上述验证针对本次输入和现有 StatelessAdam；不是独立标准 Adam 参考，也不覆盖所有 dtype/layout、Hybrid stream 依赖和完整模型。
- CUDA_VISIBLE_DEVICES 为空，张量均在 CPU，结果记录 cuda_initialized=False。CPU profiler 导出尝试查询 GPU 元数据产生 gpuGetDeviceCount code100，未使用 GPU 计算。

## 下一步

CPU 侧已完成 fill_ 开关因果验证、线程扫描和保持现有数学顺序的候选筛选。候选尚未部署。优先后续在真实 HybridDeviceOptimizer 上验证 mul_ 的参数正确性与完整 optimizer 时间，再验证训练 step；这些需要 GPU。线程调优仍须结合八 rank 实际 CPU affinity/NUMA 和竞争，不应用本单进程结果全局替换 OMP_NUM_THREADS。不要把本表比例当作训练或 RL 加速，也不要按历史6.7/1.5秒线性外推。

## 复现与资源交接

- 脚本：gpu_tools/benchmark_adam_cpu.py。
- 命令与限制：cpu_runs/adam-cpu-20260914/REPRODUCE.md。
- 原始结果：experiment01/result.json、experiment02/result.json，各含 profile 父子事件、算子汇总、哈希与逐步时间；同目录各四份 CPU trace。
- 源码哈希：cpu_runs/adam-cpu-20260914/source-hashes.txt。
- CPU 实验均已完成。保号恢复后保持运行：ai_hgx_05 PID/PGID **79791**，workers 79898–79905，日志 cpu_runs/adam-cpu-20260914/keepalive.log 已确认 first BP done，8卡利用率93–95%。本轮没有停止其他任务。
