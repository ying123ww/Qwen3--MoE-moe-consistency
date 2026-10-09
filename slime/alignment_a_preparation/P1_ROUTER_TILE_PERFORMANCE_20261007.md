# Router 小 M 计时路径与完整推理对照（2026-10-07）

## 当前状态

完整三轮性能对照已完成。单算子约30%的收益仅转化为整轮generate约0.38%的延迟下降，默认仍off。数值gate通过，未复跑跨框架训练gate。

实验目录：router-perf-hgx05-20261007-01（本地产物：`gpu_runs/router-perf-hgx05-20261007-01/`）。原始 driver PID/进程组 65728、续跑 driver 49944 均已退出；当时的 Slurm allocation 为 1028339，现已结束。阶段、子任务 PID/进程组、退出状态与日志分别记录在 state.json、*-launch.json、*-exit.json、*.log 中。无需保号（用户本轮明确取消）。

## 实现

- [router helper](../../experiments/sglang-main-sm80/python/sglang/srt/models/qwen3_router_tile.py)：默认 off；候选限 FP32 CUDA、M=1/2、K=2048、N=128。保持原 persistent kernel、K32、3 stages，仅使用 16×64/4-warps。范围外原 mm；范围内错误直接失败。
- SLIME_QWEN3_ROUTER_TILE_AUDIT=0 显式关闭 candidate 的逐调用 rank 查询、计数、JSON 与文件写入。shadow 仍强制审计；CUDA graph 仍拒绝。
- [模型 gate](gpu_tools/router_tile_exact.py) 新增 --disable-audit，独立验证实际计时路径的 tokens/routes/logprob bits。
- [性能启动器](gpu_tools/router_tile_performance.py) 在加载模型前核对八卡 P2P，并仅对本轮 worker 进程组做异常/900 GiB 内存保护清理。
- [推理计时](gpu_tools/inference_benchmark.py) 新增 --router-tile-mode，保存模式和源码哈希，计时自动关闭审计。

## 已完成的单卡验证

原始结果（本地产物：`gpu_runs/router-perf-hgx05-20261007-01/router-microbenchmark.json`）；[可重跑脚本](gpu_tools/benchmark_router_timing.py)。

| M | 原版 wall µs | candidate 无审计 wall µs | 延迟下降 |
|---|---:|---:|---:|
| 1 | 73.65 | 51.05 | 30.69% |
| 2 | 73.55 | 51.37 | 30.16% |

BF16-origin FP32 合成输入，3 seeds，共 1152 个输出元素；原版逐 bit 比较与追加一行后同行比较均零差异，审计计数器为空。12 轮交错顺序、每配置每轮 250 次调用。wall time 包含分配、Python/Triton 发射与 GPU 完成；不是纯 kernel 时间、真实模型 gate、跨框架 gate 或整模型收益。

## 完整协议

同一冻结 8 prompts（1218 input tokens）；每请求 128 response tokens，共 1024。每 Engine 3 批预热、20 批正式计时，缓存每批清空。三轮顺序分别为 baseline/aligned/candidate、aligned/candidate/baseline、candidate/baseline/aligned。沿用原 SGLang/DeepEP normal 的 TP8/DP2/CP2/EP4/MoE-DP2 拓扑。正式计时无 profiler、逐调用审计或路由捕获；加载与预热排除。

模型 gate 单独使用 greedy、16 response tokens，batch8/3/1 与变长3，独立 off/shadow/candidate Engines；候选 gate 的 auditing 为 0。通过后才进入性能计时。跨框架训练 gate 本轮尚未复跑。

## ai_hgx_04 的失败记录

首次失败实验（本地产物：`gpu_runs/router-perf-hgx04-20261007-01/`） 在原版首次 DeepEP dispatch 的 cudaIpcOpenMemHandle 处报 peer access unsupported，尚未进入 router 候选。实查 GPU2 所有 NVLink inactive、与其他卡 P2P 读写全部 NS；清理后 GPU 空闲。该失败不计入数值 gate 或性能样本。用户随后指定 ai_hgx_05；其八卡 P2P 读写预检查通过。

## 本轮真实模型 gate 结果

三模式比较（本地产物：`gpu_runs/router-perf-hgx05-20261007-01/exact-comparison.json`）确认 model_exact=true、batch_exact=true，两类 failures 均为空。shadow 覆盖 8 ranks × 48 层（384 组合），23,232 次调用，比较 3,538,944 个 FP32 输出元素，零 bit 差异；真实 eligible M 为 1/2。candidate 实际关闭逐调用审计，其完整 tokens/routes/response logprob bits 与独立 off、audited shadow 一致；三个模式自身的 batch8/3/1 同行结果一致。范围是四个有限 greedy case、16 response tokens、关闭图；未据此宣称跨框架或任意样本一致。

## 完整性能结果

| 轮次 | baseline s | aligned s | candidate s | 延迟下降 |
|---|---:|---:|---:|---:|
| 1 | 12.694660 | 15.784588 | 15.742720 | 0.265% |
| 2 | 12.738622 | 15.795575 | 15.725812 | 0.442% |
| 3 | 12.682391 | 15.755255 | 15.695439 | 0.380% |

三轮配对延迟下降的中位数为 0.380%。候选仍比baseline慢23.45%–24.01%。共180批正式计时，每种模式60批。

完整结果（本地产物：`gpu_runs/router-perf-hgx05-20261007-01/comparison.json`）。这不是纯kernel时间或CPU主因的直接证明。

第二轮baseline在第二批预热前cache flush被拒绝：仅1批预热，零正式样本，失败记录独立保留。增加最多10秒、仅在计时区间外等待成功的flush重试后续跑。核验数值源码和冻结输入未变，保留已完成的5组，其余4组完成。控制变化与核验记录在resume-manifest.json、resume-validation.json；第一版控制脚本保存在项目.a-tmp/router-perf-20261007/。加入该修复后27项CPU检查通过。

首次driver65728、续跑driver49944均已退出，正式对照完成。旧数值源码在source-snapshots-before-rms（本地产物：`gpu_runs/router-perf-hgx05-20261007-01/source-snapshots-before-rms/`）保存，后续融合实验不改写本轮结果。

下一项融合pilot已通过688,128个BF16输出/残差元素bit检查，单次wall约130→24µs（约81.5%下降）。pilot结果（本地产物：`gpu_runs/residual-rms-hgx05-20261007-01/microbenchmark.json`）。这不是模型证明；初版在真实模型 layer39 input RMSNorm 上出现1bit差异，已阻止进入完整计时；修复后重新验证，另见[P1_RESIDUAL_RMS_PERFORMANCE_20261008.md](P1_RESIDUAL_RMS_PERFORMANCE_20261008.md)。

后续residual RMSNorm融合已完成真实模型逐bit门禁与三轮180批完整计时，配对延迟下降中位数10.926%，优先级高于本项小tile。详见[P1_RESIDUAL_RMS_PERFORMANCE_20261008.md](P1_RESIDUAL_RMS_PERFORMANCE_20261008.md)。
