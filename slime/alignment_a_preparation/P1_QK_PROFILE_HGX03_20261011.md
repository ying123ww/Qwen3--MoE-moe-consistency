# ai_hgx_03 Q/K 热点调查（2026-10-11）

同一 SSH/extern 启动器；baseline 与当前 RMS 融合版本均 router off。每组 3 批预热；5 批无 profiler 检查；1 批完整 128-token profile。

- baseline 无 profiler 中位数：12.654737 秒。
- candidate 无 profiler 中位数：14.055948 秒。

| rank | baseline GPU 事件 | RMS GPU 事件 | 增量 | baseline GPU 区间并集秒 | RMS GPU 区间并集秒 |
|---|---:|---:|---:|---:|---:|
| 0 | 190028 | 242238 | 52210 | 3.242848 | 5.061437 |
| 1 | 190048 | 242260 | 52212 | 3.023936 | 4.392074 |
| 2 | 190048 | 242252 | 52204 | 3.194649 | 4.929398 |
| 3 | 190028 | 242240 | 52212 | 3.496179 | 5.352465 |
| 4 | 190028 | 242238 | 52210 | 2.938102 | 4.637424 |
| 5 | 190048 | 242260 | 52212 | 2.904795 | 4.707674 |
| 6 | 190048 | 242252 | 52204 | 3.225748 | 5.079279 |
| 7 | 190028 | 242240 | 52212 | 3.068441 | 5.169726 |

GPU 事件包含 kernel、memcpy、memset，不等于 kernel 发射次数。CPU、GPU 累计时间与运行时等待不可相加解释无 profiler 延迟。Q/K 细分标记仅用于 profile Engines，张量与布局证据只在预热采集。全部 8 ranks 及细分结果见 JSON 原始数据。

## Q/K 细分结果

rank 0 kernel: 188422 → 234782（+46360）；memcpy: 1085 → 7326（+6241）；memset: 521 → 130（-391），净增 52210 个 GPU 事件。decode 的 Q/K 布局复制 12288 次，独立 Q/K norm 各 6144 次；baseline fused QK norm 为 6144 次。联合小 M 候选若通过门禁，理论上可减少 decode 的 18432 个 GPU 事件；这个数量不是延迟收益比例。

真实 Q/K 来自 packed QKV，decode M=4，Q[4,2048]、K[4,256]，原行 stride=2560；两次 reshape 都产生实际复制。Q/K 均走 rms_norm_batch_invariant，不是 forward_native。保留 48 层 × 8 ranks 的 384 份真实张量样本。

3 份 gzip 导出被 Engine 关闭过程截断，已从完整原始 JSON 无损恢复：完整 JSON 解析和 Record Window End 验证、唯一 1 MiB 前缀匹配、重新压缩后全量解压 SHA256 相同。损坏原件和恢复记录保留在 trace-recovery.json。一个 rank 3 O 投影事件缺 External ID，通过 kernel/API correlation 与唯一同线程包围 mm 关联，详见 component-summary 的 recovered_matrix_links。

独立候选初版错误使用 rsqrt，真实样本发现一个 BF16 相邻值差异，未进入模型。修复为上游 sqrt 后取倒数后，两种原型都通过 384 真实样本＋48 合成案例的 7335936 BF16 元素逐 bit 比较。单 GPU独立算子 wall：原路径94.970 µs、stride两次发射49.165 µs、联合一次发射37.670 µs；CUDA event 区间包含发射空隙，不能当纯 kernel 时间，也不能直接预测整轮收益。证据见 qk-micro-hgx03-20261011-01 与 -02。
