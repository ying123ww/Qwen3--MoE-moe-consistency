# Q/K 归一化、布局复制与 QKV/O 调查（2026-10-11）

## 结论

Q/K 联合候选通过独立张量、完整模型、不同 batch 划分及三轮完整 generate 测量。配对延迟下降中位数 3.480%，当前剩余相对 baseline 的额外开销中位数 6.463%。

| 轮次 | baseline 秒 | RMS 秒 | RMS＋Q/K 秒 | Q/K 延迟下降 | 剩余额外开销 |
|---|---:|---:|---:|---:|---:|
| 1 | 12.750854 | 14.097863 | 13.607312 | 3.480% | 6.717% |
| 2 | 12.748009 | 14.076789 | 13.571872 | 3.587% | 6.463% |
| 3 | 12.753549 | 13.999291 | 13.532129 | 3.337% | 6.105% |

每轮三组启动顺序轮换，同一 ai_hgx_03 / SSH step_extern / Slurm 1031236；9 个独立 Engine，每组 3 预热＋20 正式，共 180 批正式、207 批总样本；固定 8 prompts、1218 输入 tokens、每请求 128 输出 tokens。无 profiler/逐调用审计；实际 CUDA graph backend 均 disabled；所有计时 worker 退出码 0，全部源码锁不变。RMS 与联合候选的 69 对采样输出哈希一致。

## 已定位的路径

Q/K 都走 SGLang 上游 rms_norm_batch_invariant，而非 forward_native。decode 的 packed QKV 行 stride=2560，Q[4,2048]、K[4,256]；两次 reshape(-1,128) 都发生真实复制。原路径每次 Q/K 调用有两个复制 kernel 和两个 norm kernel。候选直接读取原 stride，一次 kernel 计算 Q/K 并写出与原路径相同的连续输出，保留 RoPE 与 KV 写入路径。

rank 0 原 RMS 版本比 baseline 多 52210 个 GPU 事件：kernel +46360、memcpy +6241、memset -391。优化后 8 个 rank 均减少 18432 个 kernel（decode 6144 次调用，每次 4→1）；rank 0 GPU 事件 242238→223806。memcpy 仍为 7326 次，说明额外 memcpy 来自其他路径。净事件减少约占旧事件增量 35.3%，不是端到端延迟占比。

GPU/CPU 累计区间会重叠，profile 受观测扰动；部分 rank 的 GPU 区间并集没有下降，不能把其差值与 CPU/API 时间相加解释收益。端到端收益仅引用上面的无 profiler 配对计时。

## 数值验证

初版把上游 sqrt 后取倒数误写成 rsqrt，rank 0 / layer 10 的一个 Q 元素出现 BF16 相邻值差异；该失败原型未进入模型。已恢复上游 sqrt＋倒数顺序，并保留 1024-wide masked reduction、FP32 运算与末尾 BF16 舍入。

修正后：384 份真实张量、48 个合成 mode 案例、7335936 个 BF16 元素逐 bit 一致；独立 wall 中位数原路径94.970 µs、stride读取49.165 µs、联合37.670 µs。CUDA event 区间含 host 发射空隙，不是纯 kernel 时间。

完整模型 shadow：8 ranks、21504 次候选调用、127401984 个元素零 bit 差异。无审计 candidate 与 off/shadow 四组有限 greedy 输入的 tokens/routes/logprob bits 全部一致；batch8 对 batch3/batch1 检查通过。未运行跨框架训练 gate，不宣称任意输入或训练一致。

候选通过显式 forward hook 工厂 qk_runtime_hooks:install 安装，mode=joint；默认不启用。限定 BF16、小 M=1..4、head_dim=128、同类型 BF16/FP32 权重、受支持的 batch-invariant 分支；其他路径回退。上游模型和数值 RMSNorm 文件未改动。

## QKV/O 后续调查

真实输入/权重/输出来自第 0、47 层和 rank 0、4，共 8 份样本。QKV 的 M/K/N=4/2048/2560，O 投影=4/2048/2048，均 BF16、无 bias。原 tile=128×128、BK=64、8 warps；比较 M16×N128、M16×N64、M16×N32、M32×N64，保持 BK=64 与原累加源码。40 项真实矩阵与 M1/M2/M4 batch 划分检查均零 bit 差异，原 kernel 与捕获的模型输出一致。

| 真实矩阵类型 | 原 tile wall µs | M16×N64 wall µs | 配对下降中位数 |
|---|---:|---:|---:|
| QKV | 47.225 | 34.285 | 27.503% |
| O | 44.266 | 34.295 | 22.470% |

各小 tile 的 wall 时间都接近34 µs，当前单算子数据不足以证明哪个变体最优。这是有限真实矩阵的单 GPU 微基准，不是整轮收益；QKV/O tile 候选尚未接入模型、未通过完整模型或跨框架 gate，注册的 matmul 路径保持原配置。

## 证据与恢复

初次采样3份 gzip 被 Engine 关闭截断；完整原始 JSON 经解析、Record Window End、唯一 1MiB 前缀匹配及重新压缩后的全量解压 SHA256 校验恢复，原损坏文件保留。后续联合采样在 shutdown 前完整校验8份 gzip，未发生截断。一个缺 External ID 的 O 投影事件以唯一同线程 mm 包围相关 CUDA API 关联，恢复方式显式记录，歧义不作归因。

- [原路径与细分 profile](gpu_runs/remaining-profile-hgx03-qk-20261011-01/REPORT.md)
- [独立 Q/K gate 与微基准](gpu_runs/qk-micro-hgx03-20261011-02/exact-comparison.json)
- [完整模型与整轮计时](gpu_runs/qk-model-perf-hgx03-20261011-01/REPORT.md)
- [优化后的事件对照](gpu_runs/qk-joint-profile-hgx03-20261011-01/event-comparison.json)
- [QKV/O 真实矩阵结果](gpu_runs/dense-tiles-hgx03-20261011-01/results.json)

## 交接

正式采样、模型门禁、180批计时、优化后采样及 QKV/O 微基准全部结束；驱动进程48182、73442、11006已退出。8卡均1MiB显存、0%利用率、无计算进程。按用户要求，保号和恢复监控保持停止。Slurm分配1031236仍保留。
