# 真实模型残差 RMSNorm、Q/K norm/layout：手写与 torch.compile 对照

日期：2026-10-15。节点 ai_hgx_01，GPU 0，NVIDIA A100-SXM4-80GB。
Python 3.12.13、PyTorch 2.11.0+cu129、Triton 3.6.0，使用原模型 slime 环境及项目 SGLang 源码。

本轮支持继续保留现有手写融合。在已测输入中，手写融合逐 bit 通过且耗时最低；
直接 compile 残差原函数能加速，但仍比手写慢。直接 compile Q/K 原函数和两项纯数学表达式
compile 均出现严格一致性失败。失败候选未接入模型运行路径。

这里比较真实模型算子及历史真实激活回放，不是此前独立构造的 Add + ReLU。
本轮没有完整模型 forward/generate、训练或跨框架回归，不能将算子收益当成整模型加速。

## 比较定义与输入

四个候选均计算相同接口的结果：

- 原实现：复现已启用 batch-invariant 的实际分支，且先与历史模型保存输出逐 bit 核对。
- 手写融合：调用现有生产残差 `_candidate(..., allow_cuda_graph=True)`，或实际 Q/K `joint` prototype。
- compile 原函数：`torch.compile(..., backend="inductor", fullgraph=True, dynamic=False)`，保留现有自定义归约/Triton 调用。
- compile 数学式：同样编译，但用纯 PyTorch 表达式描述计算，让 Inductor 选择归约和融合。
  这是独立候选，数学等价不代表逐 bit 等价。

残差路径保留 FP32 加法、未舍入 FP32 和的平方/归约、rsqrt、乘权重后转 BF16，另返回 BF16 residual。
Q/K 路径保留 FP32 平方/均值、sqrt 后 reciprocal、乘权重后转 BF16；输出 contiguous。

残差使用 `rms-repro-hgx05-20261008-01/exact-shadow/audit` 的 rank4/6、layer39/input 捕获。
两文件的 x/residual/weight/reference 逐 bit 相同，因此只有 **1 组独立 M2 激活**，重复回放两次。
M1 为该输入前缀，M4 为两行重复，均明确记录来源。H=2048，BF16 输入/权重，epsilon=1e-6。

Q/K 使用 `remaining-profile-hgx03-qk-20261011-01/profile-candidate/qk-evidence` 的
384 文件（8 rank × 48 layer），原始 M4，另取 M1/M2 前缀，共 1152 case。
Q=[M,2048]、K=[M,256]、head_dim=128；恢复共享 QKV storage、原行 stride=2560、Q/K offset=0/2048。
另在 CPU 上核对了全部 1152 case 的输入 bits、stride、offset 和共享 storage。
运行 metadata 必须记录每层 Q/K 为 `rms_norm_batch_invariant`。

首轮误选 `profile-baseline`，layer1 的历史输出核对失败，运行立即停止。该轮不计入正式结果；
脚本已增加逐 rank/layer 的运行路径校验，正确轮次是 `qk-corrected`。

## 逐 bit 与跨 batch 结果

以下统计比较 BF16 位模式，不使用 allclose；包含正负零区分。
分母为比较元素数，包含 rank 副本、不同 M 的前缀及残差重复回放，不代表独立数值样本数。

| 算子 / 候选 | 输出元素差异 | 最大绝对误差 | 跨 batch 前缀失败 |
|---|---:|---:|---:|
| 残差 / 原实现 | 0 / 57,344 | 0 | 0 |
| 残差 / 手写 | 0 / 57,344 | 0 | 0 |
| 残差 / compile 原函数 | 0 / 57,344 | 0 | 0 |
| 残差 / compile 数学式 | **6 / 57,344** | 0.00048828125 | 0 |
| Q/K / 原实现 | 0 / 6,193,152 | 0 | 0 / 768 |
| Q/K / 手写 | 0 / 6,193,152 | 0 | 0 / 768 |
| Q/K / compile 原函数 | **40 / 6,193,152** | 0.0078125 | 0 / 768 |
| Q/K / compile 数学式 | **8 / 6,193,152** | 0.0078125 | **8 / 768** |

残差共 6 case；数学式编译在每份 M2 的 `[1,446]` 失配，M4 重复后有两处，M1 通过。
归一化输出失配，可见 residual 全部一致；Graph 输出相对原实现的失配数量相同。
四候选共 24 项跨 batch 前缀比较全部通过，说明“自身 batch 一致”不等于“与原实现逐 bit 一致”。

Q/K 手写全 1152 case 通过；compile 原函数有 40 case 失败（M2 8、M4 32），
compile 数学式有 8 case 失败（均 M2）。四候选输出布局全部吻合。
用于计时的层0/24/47共9组 Graph 输出，与同候选 direct 输出的 bits/布局全部一致；
这些计时样本本身也与原实现一致，不能取代全部1152 case的严格检查结果。

另用独立新进程复验两个实际失配样本，direct、再次调用和单次 CUDA Graph 都稳定复现，输入未变化：

| 候选 | 样本 | Q坐标 | 原实现 | compile |
|---|---|---|---:|---:|
| compile 原函数 | rank0/layer3/M4 | [3,460] | 0.48828125 | 0.486328125 |
| compile 数学式 | rank0/layer10/M2 | [1,1268] | 1.4609375 | 1.453125 |

证据为 `qk-failure-recheck.json`；复验入口为 [compile_qk_failure_recheck.py](gpu_tools/compile_qk_failure_recheck.py)。

## CUDA Graph 耗时

单位为每次算子调用的微秒。每个图捕获100次调用，保留每次输出；每轮重放20次，
用 CUDA Event 测量并除以2000次调用，摊薄 Python 提交开销。候选顺序逐轮轮换8轮；
编译、预热、捕获、profiler均在计时区间外。每项测试串行使用同一张卡。
残差表合并两次同输入重放共16轮取中位数；Q/K合并层0/24/47共24轮取中位数。
以下不是包含网络、完整模型或跨层依赖的端到端延迟。

| 算子 | M | 原实现 | 手写融合 | compile 原函数 | compile 数学式 |
|---|---:|---:|---:|---:|---:|
| 残差 RMS | 1 | 22.882 | **2.626** | 4.816 | 1.964 |
| 残差 RMS | 2 | 24.107 | **2.734** | 5.029 | 2.573 |
| 残差 RMS | 4 | 24.472 | **2.730** | 5.331 | 2.633 |
| Q/K norm/layout | 1 | 3.441 | **1.839** | 4.193 | 3.227 |
| Q/K norm/layout | 2 | 8.057 | **1.847** | 6.969 | 3.311 |
| Q/K norm/layout | 4 | 8.326 | **1.931** | 7.139 | 3.309 |

加粗表示完整本轮一致性检查通过的较快候选。残差 compile 数学式只在 M1 通过，
不能因 M1 更快而将其作为覆盖全部 M 的替代实现。Q/K 两个 compile 候选均被全量一致性检查拒绝。

## 直接调用耗时

单位为微秒；包含 Python、分配、发射和等待 GPU 完成。每轮100次调用，
前后 synchronize，host wall 计时；聚合方式同上。

| 算子 | M | 原实现 | 手写融合 | compile 原函数 | compile 数学式 |
|---|---:|---:|---:|---:|---:|
| 残差 RMS | 1 | 263.875 | 41.476 | 122.394 | 72.188 |
| 残差 RMS | 2 | 267.570 | 42.010 | 122.616 | 72.797 |
| 残差 RMS | 4 | 266.914 | 42.047 | 122.770 | 73.577 |
| Q/K norm/layout | 1 | 97.273 | 55.955 | 101.818 | 95.153 |
| Q/K norm/layout | 2 | 154.843 | 55.967 | 142.423 | 94.806 |
| Q/K norm/layout | 4 | 155.540 | 56.164 | 142.684 | 95.854 |

## 实际 kernel 数与结论

| 算子 / M | 原实现 | 手写融合 | compile 原函数 | compile 数学式 |
|---|---:|---:|---:|---:|
| 残差 / 1,2,4 | 11 | 1 | 3 | 1 |
| Q/K / 1 | 2 | 1 | 2 | 2 |
| Q/K / 2,4 | 4 | 1 | 4 | 2 |

数量来自独立 profiler trace，而非依据 `fullgraph=True` 推测。
残差直接 compile 成功融合两侧逐元素计算，但原 mean_kernel 仍是边界，合计3个 kernel。
Q/K 直接 compile 在 M2/M4 仍有Q复制、Q norm、K复制、K norm四个 kernel；
数学式 compile 仍将Q/K分为两个 kernel；手写 joint 在一个 kernel 内完成两者。
`fullgraph=True` 只保证编译图不分段，不保证单 GPU kernel 或逐 bit 一致。

**Q/K直接compile的精度变化已观察到。** 本次生成wrapper把 `eps` 声明为 `fp64`；
实际Triton缓存中，eager核TTIR的eps为f32，编译路径为f64。
原PTX使用 `sqrt.approx.ftz.f32`、`div.full.f32`、`mul.f32`；编译路径出现
`sqrt.rn.f64`、`div.rn.f64`、`mul.f64`。两者num_warps=4、num_stages=3、enable_fp_fusion=true。
这证明直接compile虽保留相同Triton源码，却改变了标量类型及后续计算精度，符合舍入差异的来源；
本轮未做强制dtype干预消融，因此不宣称这是全部差异的唯一原因。
原始IR/metadata/PTX在 `qk-cache-audit`，关键IR行及SHA保存在数值汇总中。

现有手写融合在本轮严格一致性与耗时上仍有优势。此次结果不支持用默认 compile 直接替换它们。
若继续改进 compile，需要先解决数值差异，再重新通过真实输入与整模型验证。

## 复现与证据

脚本：[残差测试](gpu_tools/compile_residual_rms_probe.py)、[Q/K测试](gpu_tools/compile_qk_layout_probe.py)。
可审阅数值汇总：[COMPILE_NORM_RESULTS_20261015.json](COMPILE_NORM_RESULTS_20261015.json)，
包含环境、逐bit统计、每batch中位数、kernel数及原始证据SHA256；已核对运行时源码未改动。

原始结果目录：`gpu_runs/compile-norm-hgx01-20261015-01/`。
残差为 `rms/results.json`、生成代码与trace；Q/K为 `qk-corrected/` 下的
`manifest.json`、`correctness.json`、`cross-batch-prefix.json`、`summary.json`、`timings.json` 与trace。
Q/K编译输出保留在 `qk-corrected.log`，缓存IR额外导出到 `qk-cache-audit/`。

在已确认空闲的 GPU 上，用现有 slime Python，设置 `CUDA_VISIBLE_DEVICES=0`、
`TORCHINDUCTOR_COMPILE_THREADS=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，
PYTHONPATH 包含 `experiments/sglang-main-sm80/python` 及本目录 `gpu_tools`，分别运行：

```bash
python gpu_tools/compile_residual_rms_probe.py --output-dir <new-run>/rms
TORCH_LOGS=output_code python gpu_tools/compile_qk_layout_probe.py --output-dir <new-run>/qk --rounds 8 --graph-replays 20
python gpu_tools/compile_qk_failure_recheck.py --run-dir <new-run>/qk --output <new-run>/qk-failure-recheck.json
```
