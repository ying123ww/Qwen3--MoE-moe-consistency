# Plan 2：DeepEP normal 与 Breakable CUDA Graph

状态：完成；3轮、12个独立Engine、240批正式样本，全部worker退出码0，源码SHA保持不变。

同一 ai_hgx_03 / SSH extern / Slurm1031236，TP8 / DP2 / CP2 / EP4，固定DeepEP normal 20 SM。每组3预热＋20正式，固定8 prompts / 1218输入tokens / 1024输出tokens。Plan 2使用原始确定性算子，decode breakable、prefill eager；DeepEPMoE专家主体eager，Gate/Top-K捕获。Plan 1是RMS＋联合Q/K、Graph关闭。baseline也是Graph关闭的原配置，未测baseline开启Graph的对照。

| 轮次 | baseline eager 秒 | 原确定性 eager 秒 | Plan 1 秒 | Plan 2 秒 | 相对原确定性延迟下降 | 相对Plan 1延迟下降 | 相对baseline延迟下降 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.708850 | 15.798449 | 13.712629 | 6.450765 | 59.168% | 52.957% | 49.242% |
| 2 | 12.721644 | 15.761854 | 13.563693 | 6.450801 | 59.073% | 52.441% | 49.293% |
| 3 | 12.708907 | 15.697274 | 13.554164 | 6.432667 | 59.020% | 52.541% | 49.385% |

配对延迟下降中位数：相对原确定性59.073%，相对Plan 152.541%，相对当前eager baseline49.293%（吞吐约1.97倍）。

有限greedy四组输入的tokens/routes/logprob bits及batch划分门禁通过；69组三模式采样输出哈希一致。8 ranks均实际replay，49个segment、48个DeepEPMoE break。独立校验见measurement-validation.json；启动、cache flush、模型加载不计入generate。未测试跨框架训练一致性。

计时无profiler或逐调用张量审计，但观察包装在每个形状首次replay时记录一次证据。第1轮batch15、第3轮batch7在正式区间首次记录batch2，不能称作完全无记录的计时。原样本保留，剔除这2/60批Graph样本后的各轮中位数变化为+0.0122%、0、-0.0225%，未改变收益判断；详见evidence-write-sensitivity.json。未测完全移除观察包装的配置。

KV长度检查的D2H/同步调用链与独立断言异常行为见 [KV检查审计](P2_KV_ITEM_AUDIT_20261011.md)。Graph整体收益不能全部归因于这一个检查。

正式任务及独立断言检查均结束，GPU已空闲。按用户要求不恢复保号；Slurm分配1031236保留。
