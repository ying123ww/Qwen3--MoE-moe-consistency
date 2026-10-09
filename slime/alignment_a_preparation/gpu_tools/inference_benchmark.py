"""SGLang-only matched-workload benchmark; deliberately not an RL iteration."""
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

FROZEN = Path('/public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt')


def flush_cache_until_idle(engine, timeout=10.0):
    """Require a successful flush; wait only outside the measured generate."""
    deadline = time.monotonic() + timeout
    retries = 0
    while True:
        flushed = engine.flush_cache()
        if flushed if isinstance(flushed, bool) else flushed.success:
            return retries
        retries += 1
        if time.monotonic() >= deadline:
            raise RuntimeError('Could not flush cache before batch within idle timeout')
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--variant', choices=['baseline', 'aligned'], required=True)
    ap.add_argument('--residual-rms-mode', choices=['off','candidate'], default='off')
    ap.add_argument('--router-tile-mode', choices=['off', 'candidate'], default='off')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--profile', action='store_true', help='One full generate capture after warmup; not a benchmark')
    ap.add_argument('--profile-shapes', action='store_true', help='Record CPU op shapes for attribution')
    ap.add_argument('--profile-attribution', action='store_true', help='Install CPU component scopes only in profiling Engines')
    ap.add_argument('--response-tokens', type=int, default=128)
    args = ap.parse_args()
    if args.profile and args.steps != 1:
        raise ValueError('Profiling requires --steps 1')
    if args.profile_shapes and not args.profile:
        raise ValueError('Shape recording requires --profile')
    if args.profile_attribution and not args.profile:
        raise ValueError('Component scopes require --profile')
    if args.response_tokens < 1:
        raise ValueError('Response token count must be positive')
    if args.router_tile_mode != 'off' and args.variant != 'aligned':
        raise ValueError('Router tile candidate requires the aligned variant')
    if args.residual_rms_mode != 'off' and args.variant != 'aligned':
        raise ValueError('Residual RMSNorm candidate requires aligned mode')
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_MODE'] = args.residual_rms_mode
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_AUDIT'] = '0'
    os.environ.pop('SLIME_QWEN3_RESIDUAL_RMS_AUDIT_DIR', None)
    os.environ['SLIME_QWEN3_ROUTER_TILE_MODE'] = args.router_tile_mode
    os.environ['SLIME_QWEN3_ROUTER_TILE_AUDIT'] = '0'
    os.environ.pop('SLIME_QWEN3_ROUTER_TILE_AUDIT_DIR', None)
    import torch
    samples = torch.load(FROZEN, map_location='cpu', weights_only=False)['samples']
    prompts = [list(map(int, s['tokens'][:-int(s['response_length'])])) for s in samples]
    assert len(prompts) == 8 and sum(map(len, prompts)) == 1218
    import sglang as sgl
    aligned = args.variant == 'aligned'
    config = dict(model_path='/storage/data/yingxy2024/Models/Qwen3-30B-A3B',
                  trust_remote_code=True, random_seed=1234, dtype='bfloat16',
                  tp_size=8, dp_size=2, attn_cp_size=2, ep_size=4, moe_dp_size=2,
                  enable_dp_attention=True, enable_dp_lm_head=True, moe_dense_tp_size=1,
                  moe_a2a_backend='deepep', deepep_mode='normal', moe_runner_backend='triton',
                  attention_backend='triton', kv_cache_dtype='bfloat16', page_size=64,
                  mem_fraction_static=0.65, context_length=1024, max_prefill_tokens=512,
                  disable_prefill_cuda_graph=True, enable_memory_saver=True,
                  enable_deterministic_inference=aligned,
                  skip_server_warmup=True, log_level='info', port=15800, nccl_port=15801)
    if aligned:
        config['deepep_config'] = os.environ['SLIME_QWEN3_FIXED_DEEPEP_CONFIG']
    if args.profile_attribution:
        config['forward_hooks'] = [dict(name='p1-attribution', target_modules=[''],
                                       hook_factory='profiling_annotations:install', config={})]
    sampling = dict(temperature=0.6, max_new_tokens=args.response_tokens, ignore_eos=True)
    manifest = dict(cache_flush_policy='require_success_with_bounded_idle_retry', variant=args.variant, residual_rms_mode=args.residual_rms_mode, router_tile_mode=args.router_tile_mode,
                    router_tile_audit=False, source_sha256={
                        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                            Path(__file__),
                            Path(sgl.__file__).parent / 'srt/models/qwen3_router_tile.py',
                            Path(sgl.__file__).parent / 'srt/batch_invariant_ops/fused_rmsnorm_candidate.py',
                            Path(sgl.__file__).parent / 'srt/layers/layernorm.py',
                            Path(sgl.__file__).parent / 'srt/models/qwen3_moe.py',
                            Path(sgl.__file__).parent / 'srt/batch_invariant_ops/batch_invariant_ops.py',
                        )}, requested_server_args=config, sampling=sampling,
                    prompt_lengths=list(map(len, prompts)), frozen_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
                    scope='blocking Engine.generate for 8 prompts: scheduling + prefill + decode + sampling + output logprobs + detokenization/IPC; excludes load, warmup, cache flush, reward, training and weight sync',
                    environment={k: v for k,v in os.environ.items() if k.startswith(('SGLANG_', 'NCCL_', 'SLIME_QWEN3', 'TVM_FFI_'))},
                    note='Same patched software and Triton/DeepEP-normal topology. Baseline is configured baseline, not pristine upstream or tuned peak. Fixed lengths, generated token identities may differ.')
    (args.run/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    started = time.perf_counter()
    if args.profile_attribution:
        annotation_source = Path(__file__).with_name('profiling_annotations.py')
        manifest['source_sha256'][str(annotation_source)] = hashlib.sha256(annotation_source.read_bytes()).hexdigest()
        (args.run/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    engine = sgl.Engine(**config)
    try:
        (args.run/'resolved-server-args.json').write_text(json.dumps(dataclasses.asdict(engine.server_args), indent=2, default=str)+'\n')
        (args.run/'startup.json').write_text(json.dumps({'startup_seconds': time.perf_counter()-started})+'\n')
        timings = []
        for i in range(args.warmup + args.steps):
            flush_retries = flush_cache_until_idle(engine)
            capture = args.profile and i == args.warmup
            if capture:
                engine.start_profile(output_dir=str(args.run/'profiles'/'rollout'),
                                     activities=['CPU','GPU'], with_stack=False, record_shapes=args.profile_shapes)
            started = time.perf_counter()
            outputs = engine.generate(input_ids=prompts, sampling_params=sampling, return_logprob=True, logprob_start_len=-1)
            elapsed = time.perf_counter()-started
            if capture:
                engine.stop_profile()
            assert len(outputs) == 8
            completion = [o['meta_info']['completion_tokens'] for o in outputs]
            assert completion == [args.response_tokens]*8, completion
            for out in outputs:
                lp = out['meta_info']['output_token_logprobs']
                assert len(lp) == args.response_tokens and all(math.isfinite(float(x[0])) for x in lp)
            row = dict(cache_flush_retries=flush_retries, batch=i, profiled=capture, warmup=i<args.warmup, elapsed_s=elapsed, output_tokens=sum(completion),
                       output_tokens_per_s=sum(completion)/elapsed,
                       cached_tokens=[o['meta_info'].get('cached_tokens') for o in outputs],
                       output_sha256=hashlib.sha256(json.dumps([o.get('output_ids', o['text']) for o in outputs]).encode()).hexdigest())
            with (args.run/'measurements.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            if i >= args.warmup:
                timings.append(elapsed)
            print(json.dumps(row), flush=True)
        median = statistics.median(timings)
        ordered = sorted(timings)
        idx = (len(ordered)-1)*0.95
        p95 = ordered[int(idx)] + (ordered[math.ceil(idx)]-ordered[int(idx)])*(idx-int(idx))
        summary = dict(residual_rms_mode=args.residual_rms_mode, router_tile_mode=args.router_tile_mode, router_tile_audit=False, profiled=args.profile, timing_use='attribution only' if args.profile else 'benchmark', variant=args.variant, formal_batches=len(timings), warmup_batches=args.warmup,
                       median_s=median, p95_s=p95, output_tokens_per_s=(8*args.response_tokens)/median,
                       input_tokens=1218, output_tokens=8*args.response_tokens, scope=manifest['scope'])
        (args.run/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()
