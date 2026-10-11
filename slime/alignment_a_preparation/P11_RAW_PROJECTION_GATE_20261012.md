# P11 原始投影操作数与输出门禁

原始快照采集和独立数值门禁已完成。它补齐原始GEMM参考数据，
**不代表NCU计数器权限已解决，也不代表P11整阶段完成**。

采集：`gpu_runs/p11-raw-projection-profile-hgx03-20261012-05/`。
门禁：`gpu_runs/p11-raw-projection-gate-hgx03-20261012-05/`。
同Qwen3 BF16、8×A100、TP8/DP2/CP2/EP4、C alignment、cap16、full buckets2/4。
8rank各32次选定full回放；原始数据覆盖rank0/4、第0/47层、两个bucket，
QKV/O/FP32 Router共24份实际输入、权重和原始输出。

## 为什么需要单独采集

模块返回的tensor可能被后续通信或原地计算改写；保留Python引用并不等于
保存了原始GEMM输出。跨bucket延迟转存也会失效：先前运行中，bucket4回放后
rank0的bucket2快照不再匹配，但当前bucket4的12份快照全部精确。
两个graph共用pool，不能据旧引用的存在假设旧输出内容仍有效。

新采集器在原始 `_matmul_persistent_triton` 调用前复制输入，调用后立即
复制输出。记录最近一次实际full replay的bucket，**每个bucket回放后立即
转存该bucket数据**：3prompt小请求之后转存bucket2，再完成预热及32步
选定回放后转存bucket4。没有把其它bucket的旧引用混入当前快照。
保存操作不在正常性能测量内；graph中的复制仅用于这项数值诊断。

## 实际形状与验证范围

| 来源 | bucket2 M/K/N | bucket4 M/K/N | dtype |
|---|---|---|---|
| QKV | 2/2048/2560 | 4/2048/2560 | BF16 |
| O | 2/2048/2048 | 4/2048/2048 | BF16 |
| Router | 1/2048/128 | 2/2048/128 | FP32 |

Router行数受CP分片影响，不能把capture bucket直接当作它的M。
独立重算使用原始persistent kernel：BF16 BM128/BN128/BK64，
FP32 BM128/BN128/BK32，均8warps、3stages、GROUP_M8。

24份快照共**112,128个原始输出元素**，全部有限；
原始输出逐bit差异**0**，M前缀1/2/4中适用的批次检查差异**0**。
FP32用int32 bit表示比较，BF16用int16表示比较，不使用容差。
只证明这些有限实际矩阵和前缀，不外推所有输入或训练参数更新。

P12随后在同一批24份原始数据上测试小tile，结果另行记录。
本文不将单算子结果称为完整模型加速。

## 入口与交接

```bash
python gpu_tools/run_raw_profile.py \
  --p10-run gpu_runs/p10-full-ab-timing-hgx03-20261012-01 \
  --run-id NEW_UNIQUE_RAW_RUN
python gpu_tools/validate_raw_projections.py \
  --evidence gpu_runs/NEW_UNIQUE_RAW_RUN/C/raw-projections \
  --run-id NEW_UNIQUE_GATE_RUN
```

入口仅支持C smoke数值诊断，不支持A/C Nsight模式；普通时间线入口仍是
`run_full_graph_profile.py`。在ai_hgx_03、slime环境及已分配GPU中执行。
采集使用资源锁、P2P/内存保护、source SHA与原生full捕获/回放校验。
检查门禁实际完成后才使用输出；不得把提交命令或24个文件的存在当作通过。

验证过的collector/hook/validator字节保持与成功运行一致。
发布入口仅将临时目录定位改成自身目录，并限制为已验证的smoke模式；
模块解析、CLI帮助、语法和仓库hygiene已检查。
raw数据、权重、失败快照与临时脚本保持忽略，不纳入Git。
