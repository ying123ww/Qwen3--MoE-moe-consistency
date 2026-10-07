# CPU optimizer 修复 GPU 验证（ai_hgx_05，2026-09-14）

## 结论

在单张 A100-SXM4-80GB、实际 HybridDeviceOptimizer + StatelessAdam 的隔离测试中，两项补丁保留 FP32 master pinning，并在正常异步环境下恢复部分 H2D/CPU 工作重叠。128 参数测试三轮配对中，完整 optimizer 调用耗时下降 40.49%–42.56%（三轮配对降幅平均约 41.28%）。这不是完整模型训练 step 或在线 RL 加速。

## 配置与正确性

实际依赖：`/public/home/yingxy2024/slime-deps/Megatron-LM`，修复提交 `73a68e7`。读取该文件并在内存中构造 old、pin_only、event_only、both 四个隔离类；没有修改工作区依赖以切换版本。

- GPU0，CPU threads=4，确定性算法开启，非零 lr=1e-3。
- BF16 GPU parameters、FP32 decoupled gradients 和 CPU masters，全部 CPU offload，overlap=True。
- 128 个参数：64 对 6/12 MiB FP32 master，总计 1.125 GiB；梯度为固定 seed 的合成数据，没有前后向。
- 每种模式每轮 4 步预热、10 步正式测量，3 轮交替顺序，均在同一 Python 进程内重新初始化。不是3次独立进程实验。
- 每步测量前后同步 device；profile 与正式计时分开，只有第一轮额外采一份 trace。
- 正确性：15次更新后，四路径GPU参数字节哈希相同；current-stream consumer读到的参数与CPU master转换值无差异。此处没有独立标准Adam数学参考，也没有证明所有竞态均不可能发生；不能据pin_only本次通过而去掉event修复。

## 关键环境发现

远端默认环境含 `CUDA_LAUNCH_BLOCKING=1`；当前共享 `~/.bashrc:22` 明确导出该值（本轮只读检查，未修改）。最初01（16参数）和02（128参数）继承该设置，虽然pinning改善耗时，但GPU dtype cast的host kernel launch阻塞覆盖H2D，未形成CPU math重叠。这两轮保留为环境诊断证据。

03显式设置 `CUDA_LAUNCH_BLOCKING=0`，只影响测试进程，没有改用户全局环境。旧9月10日训练trace是否继承该变量尚未确认，不能由本节点当前环境倒推历史。

## 03 异步环境结果

| 轮次 | Old 中位数 ms | Both 中位数 ms | 耗时下降 |
|---|---:|---:|---:|
| 0 | 375.237 | 215.540 | 42.56% |
| 1 | 372.765 | 220.663 | 40.80% |
| 2 | 371.734 | 221.213 | 40.49% |

三轮中位数再取中位数为372.765→220.663ms，下降40.80%；这与三轮配对降幅的均值41.28%是不同统计量。

独立profile中的指标：

| 指标 | Old | Both |
|---|---:|---:|
| H2D API 主线程时间 ms | 140.495 | 1.766 |
| H2D GPU interval union ms | 90.954 | 49.559 |
| H2D ∩ 互斥CPU数学 ms | 0 | 13.922 |
| H2D ∩ 互斥CPU fill_ ms | 0 | 17.549 |

Both的H2D与狭义CPU数学重叠占H2D GPU union的28.09%；计入fill_则为63.50%。数学、fill_互斥，API与GPU时间不能直接相加。Pinned master数量为128/128，Old为0/128。

## 产物与后续边界

- `gpu_tools/benchmark_optimizer_backport.py`：四组独立类、正确性、无profiler计时及trace入口。
- `gpu_runs/optimizer-backport-hgx05-20260914-03/result.json`：环境、实际源码hash、逐步时间及参数hash。
- 同目录 `exclusive_trace_analysis.json`、`TRACE_ANALYSIS.md` 与四份原始trace。
- `gpu_runs/optimizer-backport-hgx05-20260914-01/analyze_traces.py`：CPU复算脚本。

本轮已经回答“补丁是否能改善搬运并恢复部分重叠”。完整训练step、14GiB级实际参数布局、真实梯度和在线RL净收益仍需单独测量；不要用本微基准比例填入训练吞吐字段。
