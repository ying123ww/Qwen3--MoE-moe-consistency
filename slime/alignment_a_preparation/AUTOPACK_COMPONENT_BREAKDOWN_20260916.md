# Native / RoutePack 分段 GPU 归因

2026-09-16，ai_hgx_03，8×A100 80GB，TP1/EP4；原128条完整样本，max_tokens_per_gpu=10240。恢复后的rollout文件SHA256与9月14日原文件完全相同。只跑Native与RoutePack，没有Random或重复forward gate。

## 结论

两组真实训练与分段采集均完成。被观察的首microbatch里，两套专家前向成本几乎不变，明显差别落在重计算的dispatch及DeepEP相关kernel上。但该差别不能外推为整step加速：本轮无profiler的单次pilot完整step为Native **140.06787s**、RoutePack **140.07860s**，基本相同；原5轮测评也未证明稳定收益。

## 测量方法与限制

1次预热、1次无监测pilot、1次归因、1次梯度检查。归因轮分别为首microbatch model.forward、对应backward_step和整个optimizer.step创建独立CPU+CUDA profiler context；每个窗口开始/结束同步，保留异步尾部。各rank独立Chrome trace记录kernel、memcpy和多stream活动，通过CUDA API correlation关联CPU组件范围；48个有效trace（2方法×8ranks×3窗口）的GPU事件均有CUDA launch关联。

首microbatch不是全step；loss forward不在model.forward窗口，backward包含loss反向与重计算。窗口同步和profiler扰动会改变rank到达及等待，不能据单次结果证明因果或稳态改善。下表为**8 ranks的中位数，不是8次独立重复**。GPU component数字为关联活动的inclusive duration sum；嵌套行不能相加，跨stream重叠需看union。DeepEP kernel时间包含等待，不是纯数据传输时间。

## 首microbatch：GPU活动时间

| 分段（秒） | Native | RoutePack |
|---|---:|---:|
| forward dispatch | 0.39967 | 0.39990 |
| forward expert排列 | 0.07228 | 0.07227 |
| forward SGLang Triton experts | 0.27888 | 0.27452 |
| forward expert恢复排列 | 0.01858 | 0.01856 |
| forward combine（含gather路径） | 0.15554 | 0.15433 |
| 重计算 dispatch | 1.84845 | 0.77748 |
| 重计算 SGLang Triton experts | 0.27858 | 0.27613 |
| 重计算原生TE fc1 | 0.15266 | 0.15263 |
| 重计算原生TE fc2 | 0.08799 | 0.08830 |
| 重计算 combine | 0.21110 | 0.21140 |

原生TE fc1/fc2只覆盖原生专家中的线性层，不包括激活、其他处理或全部原生backward，不能称为完整原生专家forward成本。combine中的有序gather未进一步单独标记；本次可定位到包含它的combine区间，不能单独报gather时间。

## 首microbatch：真正backward与多stream

| 指标（秒） | Native | RoutePack |
|---|---:|---:|
| forward CUDA活动union | 1.92532 | 1.90415 |
| backward CUDA活动union | 11.07345 | 9.90892 |
| backward DeepEP命名kernel duration sum | 3.76932 | 2.52809 |
| backward GEMM命名kernel duration sum | 1.08711 | 1.08436 |
| backward未细分类kernel duration sum | 6.63911 | 6.63382 |

这里backward窗口包含真实autograd backward，而非仅重算forward。GEMM/DeepEP基于kernel名字分类，未识别的保留unknown。大量真正backward操作位于forward组件范围以外：组件未归属不等于CUDA关联失败，更不能将这些时间归到通信。

被采集slot0的最忙rank H之和9,777,840→9,626,503，约下降1.55%；此slot总E下降约5.34%。多数计算区间几乎不变，而dispatch差异远大于H变化，支持继续检查到达/等待与执行时序，不能简单归因于计算量下降。

## Optimizer：整step一次调用

| 指标（秒） | Native | RoutePack |
|---|---:|---:|
| optimizer CPU调用区间wall | 11.29073 | 11.76094 |
| optimizer CUDA活动union | 3.46176 | 3.31018 |
| memcpy活动duration sum | 3.46926 | 3.46196 |
| NCCL命名kernel duration sum | 0.56122 | 0.43108 |

CPU调用wall包含计算、拷贝及等待，不能把约11秒全部当作CPU数学。Memcpy与NCCL可重叠，sum不能相加当wall；CPU wall减GPU union也不是精确CPU数学时间。两种schedule没有明显改变需要搬运的参数量，本次memcpy活动时间近似。

## 采集修复与后续

第一次使用dynamic collection toggle的trace只有CPU事件，已标为invalid，未用于结果。小矩阵验证普通profiler有CUDA而toggle无CUDA后，改用独立窗口，三个窗口的GPU smoke均通过再重跑。有效结果目录为`gpu_runs/autopack-components-hgx03-20260916-02/`。

当前已经把大部分所需区间拆开，但没有单独隔离完整原生专家forward、gather、每种autograd算子和optimizer CPU数学。下一项有价值的验证是用较轻的CUDA事件/明确通信依赖检查dispatch等待，并跨多个slots复现；不是继续增加swap搜索预算。不能用这里backward union下降约10.5%填写“训练提速10.5%”。

证据：`component-comparison-summary.json`，`analysis-native/`，`analysis-routepack/`及两组`runs/timing_*-02/timing.json`。恢复保号情况见同目录`keeper-retry.json`与`keepalive-retry.log`。

补充rank0真实kernel名统计：`indexing_backward_kernel`占Native 2.068s、RoutePack 2.093s（13,576/13,604次）；kernel名字本身不足以唯一归到某个Python算子，本报告不猜测来源。Top15见结果目录`top-backward-kernels.json`。

保号已验证恢复：PGID 7247、torchrun PID 7258，日志出现first BP done，8卡利用率95%–98%。首次恢复继承实验allocator后OOM，清除allocator配置后重试通过；不将首次提交成功当作恢复成功。
