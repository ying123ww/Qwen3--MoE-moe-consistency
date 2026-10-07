# A 阶段 GPU 实验进展

用户已于 2026-09-09 授权使用 ai_hgx_03。节点为 8 × A100-SXM4-80GB，首次检查全卡空闲。

## 实验环境

- 独立 Ray：127.0.0.1:8267，GCS 6387；任务临时目录 /tmp/ray-alignment-a-20260909。
- 独立 CUDA 构建路径：/tmp/slime-cuda-alignment-a-20260909。
- 启动与测量工具位于相邻 gpu_tools；不改模型数值实现。
- Probe CPU 测试 2 项、跨 rank 汇总测试 1 项通过。

## 尝试记录

1. a-restore-20260909-01：已主动停止。模型加载和直连 health_generate 正常，但继承的失效代理使启动器等待失败。未运行训练，不作为数值或性能结果。
2. a-restore-20260909-02：在 worker 环境中清空代理并设置 NO_PROXY=* 后重试；模型、并行、优化器和对齐设置未变。

## 后续 gate

先完成两轮在线复现和严格 logprob 验收，再运行固定数据 aligned/native correctness，之后用相同优化器和 lr=0 做完整 step 时间/显存对照。模型初始化、首次未计时 step 和后续预热不计入稳态训练吞吐。

## 节点通信异常

第二次尝试已通过生成健康检查和八卡训练初始化，随后 SSH 与 Ray agent HTTP 持续超时。最后确认的状态是训练模型初始化，不能据此推断当前仍在运行或已经停止。已尝试仅停止 a-restore-20260909-02，但尚未获得确认；GPU 是否释放也未确认。没有新的 numerical gate 或性能结果。详细状态见 node-communication-incident.json。

## 用户授权强制清理

已向 UID 2772 名下的 Python/Ray/SGLang 计算进程发送 SIGKILL，包含全部八个训练 actor 和八个推理 GPU 进程。复查训练进程仍为 R，SIGKILL pending；raylet 为 Z。八卡显存仍约 73–74 GB/卡，尚未释放；没有新作业。证据见 force_cleanup_result.json。不能将发送信号成功表述为进程已退出。

## 节点恢复后预检

15:06 节点 GPU 无计算进程，用户旧 Ray/Python 已清除，1 分钟 load 为 0.09。当前作业 999296 的 cgroup 内存上限为 320 GiB，CPU affinity 32；根分区仍仅 1.7 GB 可用。新版预检正确阻止启动。Slurm 拒绝对运行中的作业修改 MinMemoryNode。已准备 ALIGN_A_RAY_TMP 环境参数用于新实验迁移临时目录；尚未创建新集群或提交 GPU 作业。重新申请资源需先确认释放当前分配。

## 新分配上已提交第三次复现

Slurm 999346，ai_hgx_03，64 CPU / 640 GiB / 8 GPU。资源和空闲 GPU 预检通过。独立 Ray 临时目录 /dev/shm/ray-a-2772-restore03，object store 4 GiB。已提交 a-restore-20260909-03，Ray 返回 RUNNING（15:35:55 开始设置 runtime env）。此状态仅证明任务已提交并开始初始化，不代表 correctness 或训练更新完成。回执和状态位于同名子目录。

## 第三次失败原因及第四次提交

03 已 FAILED：SGLang numa_utils._create_numactl_executable 硬编码 /tmp，在写启动脚本时 ENOSPC，未进入 GPU 模型加载。修正该函数尊重 tempfile.gettempdir，并为 launcher 的 worker 设置独立 TMPDIR/TMP/TEMP。远端可执行文件验证通过、9 项 CPU 回归通过，已提交 a-restore-20260909-04。此补丁修改了 SGLang 源码的启动路径，数值实现未改；后续源码清单需要包含此变更。

## 第四次持续监督结果

SGLang 已在八卡加载，健康检查通过；Megatron 已加载 checkpoint，在线 rollout 已保存 rollout/1.pt，进入首步 backward。随后 cgroup 用量达到 640 GiB 上限，failcnt 56，观察到 cache 约 188 GiB、RSS 约 452 GiB。首步尚无完成记录或 train dump。SSH/Ray 再次超时，任务专属 stop POST 未确认；独立 SSH 认证后超时。旧 shell 追加 driver 24019 的 TERM 命令，执行未确认。尚未确认释放 GPU，不得追加实验。详细快照 supervision.json。640 GiB 建议已被实测证明缺乏余量，后续需重新核算并降低 CPU backup/offload 峰值。

## 1 TiB 分配与900 GiB保护

新 Slurm 999484，64 CPU、1 TiB、8 GPU。独立监控 PID 49233 每秒读取 job-level memory.usage_in_bytes（含 cache/shmem），阈值 966367641600 bytes。达到阈值或读数失败后，持续 SIGKILL 同 UID 且属于该 job cgroup 的 Python/Ray/SGLang 计算进程，排除监控自身和 SSH。本地脚本 /dev/shm/a-memory-guard-999484.py，状态 /dev/shm/a-memory-guard-999484.json。内核失去响应时不能保证信号立即生效。启动器在启动前和提交前强制检查监控心跳与job匹配。12 项 CPU 测试通过。

05 在监控未就绪时被正确阻止，没有提交。06 已成功提交，19:55:40 返回 RUNNING/runtime env setup，复查监控活跃，使用约10.5 GiB，未触发。guard-start.json保存快照。独立Ray目录 /dev/shm/ray-a-2772-job999484。

## 2026-09-10：在线复现验收通过

06 Ray状态SUCCEEDED。CPU独立比较step1/2：各8samples、1024target tokens，max_abs=0、mismatch=0、finite。日志训练step1/2 grad_norm分别8.390384493032364和8.007492371649894，lr=1e-6；权重同步完成。使用StatelessAdam；不是标准有状态Adam长期验证。900GiB监控仍正常且未触发。开始固定rollout aligned/native正确性比较，再进入计时。

## 固定rollout正确性对照

aligned/native均成功完成且八rank审计通过。aligned:1024 targets、0 mismatch、max_abs=0。native:344/1024 mismatch、max_abs=0.7916750907897949、mean_abs=0.014129203162156045。两者lr=0、同一历史rollout1与初始checkpoint。计时自动衔接首次被GPU残留保护拦截，未提交；已改为最多300秒等待实际GPU计算进程退出。aligned-timing-20260910-02已提交，计划随后native-timing-20260910-02，每种26步（首步不计时，5步预热，20步统计）。

## 首组完整性能对照已完成

两组timing-20260910-02均SUCCEEDED；每组八rank、第2–26轮记录齐全，排除5步预热后20步。native median17.45372460s，aligned17.95016807s，增加2.84434%；输入吞吐下降2.76568%；allocated峰值增加0.132115GiB/rank。非多次独立重复，不能宣称统计显著或生产最大吞吐。详见 ../A_STAGE_RESULTS_20260910.md。顺序流程PAIR_COMPLETE，nvidia-smi确认无计算进程，900GiB guard未触发、仍活跃。

## 三组独立重复已完成

额外四次03/04运行均成功，八rank验收通过。配对开销2.84434%、2.18512%、1.99784%，平均2.34243%。原始记录/CSV/JSON与重复结果报告已写入。已确认GPU无计算进程，900GiB guard仍活跃且未触发。下一阶段为扩负载，尚未执行。

## 2026-09-10 推理补测完成

六次独立 SGLang 运行成功，每次 3 warmup+20 formal batches，8×128 output tokens。三组耗时增幅 23.5637%、23.8586%、23.0218%，均值 23.4814%。summarize_inference.py 校验通过；GPU 已空，900 GiB guard 未触发，Slurm 保留。明确与 training-phase +2.34% 及 PR #2262 e2e +25% 的口径区别。新报告 INFERENCE_AND_PR2262_20260910.md。

## 2026-09-10 在线端到端 pilot 完成

两次 Ray SUCCEEDED，完整六轮/每种，首轮预热+五轮正式。中位数native46.359768s、aligned52.398774s，比值+13.0264%；明确单次短程非稳态结果。工作量/分段/学习率/有限梯度校验通过；aligned一次随机奖励组内全相同造成零梯度，如实保留。GPU已空，900GiB未触发，Slurm保留。报告 ONLINE_RL_TIMING_20260910.md。

## 2026-09-10 P1 profiling partial / node root disk full

User skipped P0; implemented bounded third-iteration profiling. aligned-01 failed on plain-text start_profile JSON parsing, fixed with regression test. aligned-02 saved 16 SGLang and 2 Megatron GPU traces, then SGLang aborted in CUDA IPC deserialization during weight sync. Stopped failed Ray job; controller exited; no native job submitted. GPU workers released. Root filesystem has 0 available space; /var/crash holds ~133 GiB root-owned historical vmcores; no system files deleted. 900 GiB guard remains active and untriggered. Report: ../P1_PROFILING_20260910.md. Do not resume original full-chain profiler blindly; isolate profiler/IPC after root space restored.

## P1 independent inference recovery, 2026-09-10 22:30 UTC
Both a-p1-inference-{baseline,aligned}-20260910-01 succeeded. 8 GPU traces each, 3 warmup + 1 full-generate capture, 1024 output tokens per batch, finite logprobs. Dedicated workspace temp and devshm compiler caches write-tested and removed after success. GPU workers released; guard preserved. Root-full does not block this migrated independent path. Online CUDA IPC failure unresolved. P1 report updated with paired kernel candidates, not performance claims.

## 2026-09-11 P1 frozen training pair complete
Native/aligned training-only runs both SUCCEEDED, 3 RPCs each, rollout 3 captured on ranks 0/4, all 8-rank audit/finite nonzero gradients verified. GPU released; dedicated scratch cleaned; guard untriggered. Fixed summary CPU/GPU user annotation double-counting, reprocessed old online and new training traces, corrected prior report; normal benchmarks unaffected. New report: ../P1_TRAIN_PROFILING_20260911.md. CPU optimizer ~15.4s common cost; sampled F/B ~2.23 vs 3.30s candidate, not benchmark overhead. IPC unresolved.
