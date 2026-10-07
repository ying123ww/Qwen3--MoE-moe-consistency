# Real-workload evidence

`capture02-evidence.tar.gz` contains all 256 captured token arrays, route arrays, FP32 logprobs/bits, sampling seeds and finish reasons, plus capture manifest/log/argv/environment. Extract under `/tmp` to restore the absolute paths used by `../autopack-real-hgx05-b1-20260914-02/b2-verified/samples.jsonl`:

```bash
tar -xzf capture02-evidence.tar.gz -C /tmp
```

The GPU run used ai_hgx_05. CPU timing used debug02 and psamp Python. Seed 0 ran three measured repeats after warm-up without concurrent benchmark processes; seeds 1–4 ran concurrently for outcome robustness only, and their wall times must not be used as performance estimates.

Native schedule B1/B2 gates passed at global batch 256, max_tokens_per_gpu 20480, data padding multiplier 512, TP1/EP4. The checked-in standard workload recipe uses TP4/EP8; this run retained the previously validated alignment topology. Detailed original GPU arrays remain on ai_hgx_05 at `/tmp/ap-real-b1-arrays-20260914-02`; the portable archive contains capture evidence, not all 18,432 GPU dispatch/replay audit arrays. Source hashes and verified B2 summaries are preserved in the B1-02 directory.

The first real capture was stopped and excluded because it omitted per-response sampling seeds. Capture02 uses Slime's group seeds42–49. The first real B1 hit a logits FP32 allocation OOM; B1-02 passed with expandable allocator segments, unchanged dtype/shape/token budget.

The GPU task finished. The repository-authorized Wan21 keeper was restored; status.json and keepalive02.log record its process identity and first-backward verification. Keeper output is unrelated to RoutePack results.
