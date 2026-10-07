"""Bounded multi-slot input capture on ai_hgx_05; not a timing benchmark."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts-json', type=Path, required=True)
    p.add_argument('--reference-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--response-tokens', type=int)
    p.add_argument('--real-training', action='store_true', help='256 records, natural EOS, temperature 1, up to 8192 response tokens')
    p.add_argument('--temperature', type=float)
    p.add_argument('--context-length', type=int)
    p.add_argument('--capture-batch-size', type=int, help='Bound Engine response/base64 routes retained in host memory')
    args = p.parse_args()
    args.response_tokens = args.response_tokens if args.response_tokens is not None else (8192 if args.real_training else 128)
    args.temperature = args.temperature if args.temperature is not None else (1.0 if args.real_training else 0.0)
    args.capture_batch_size = args.capture_batch_size if args.capture_batch_size is not None else 32
    if args.capture_batch_size < 1 or args.temperature not in (0.0, 1.0):
        raise ValueError('Positive batch size and temperature 0 or 1 required for B1 logprobs')
    if args.real_training and args.temperature != 1.0:
        raise ValueError('Real training capture requires temperature 1')
    if socket.gethostname().split('.')[0] != 'ai_hgx_05':
        raise RuntimeError('This entrypoint is for ai_hgx_05')
    if not 1 <= args.response_tokens <= (8192 if args.real_training else 128):
        raise ValueError('response token limit exceeds capture mode')
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if busy.strip():
        raise RuntimeError('GPU processes exist; serialize capture after optimizer tests')
    fixture = json.loads(args.prompts_json.read_text())
    sources = fixture['samples']
    prompts = [list(map(int, s['prompt_tokens'])) for s in sources]
    if args.real_training:
        if len(prompts) != 256 or len({tuple(x) for x in prompts}) != 32:
            raise ValueError('Real fixture requires 32 distinct prompts with 8 responses each')
        for i, source in enumerate(sources):
            if prompts[i] != prompts[(i // 8) * 8]:
                raise ValueError('Real fixture must keep each eight-response prompt group contiguous')
            source['group_index'] = i // 8
    elif len(prompts) != 32 or len({tuple(x) for x in prompts}) != 32:
        raise ValueError('Expected exactly 32 distinct prompts')
    if any(not x or (not args.real_training and len(x) + args.response_tokens > 512) for x in prompts):
        raise ValueError('Prompt plus response must fit 512-token bin')
    reference = json.loads(args.reference_manifest.read_text())
    config = dict(reference['requested_server_args'])
    if not config.get('enable_deterministic_inference'):
        raise ValueError('Aligned reference required')
    config.update(enable_return_routed_experts=True, skip_server_warmup=True,
                  disable_cuda_graph=True, disable_prefill_cuda_graph=True)
    required_context = max(map(len, prompts)) + args.response_tokens
    if args.context_length is not None:
        if args.context_length < required_context:
            raise ValueError(f'Context length must be at least {required_context}')
        config['context_length'] = args.context_length
    elif args.real_training:
        config['context_length'] = max(config.get('context_length') or 0, required_context)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['SLIME_QWEN3_ROUTER_TILE_MODE'] = 'off'
    os.environ.pop('SLIME_QWEN3_ROUTER_TILE_AUDIT_DIR', None)
    args.output.mkdir(parents=True, exist_ok=False)
    sampling = dict(temperature=args.temperature, max_new_tokens=args.response_tokens, ignore_eos=not args.real_training)
    if args.real_training:
        sampling.update(top_p=1.0, top_k=-1, no_stop_trim=True)
        seed_base = int(fixture['rollout_seed'])
        per_request_sampling = [dict(sampling, sampling_seed=seed_base + i % 8) for i in range(len(prompts))]
    else:
        per_request_sampling = None
    manifest = dict(requested_server_args=config, sampling=sampling,
        sampling_seed_policy='fixture.rollout_seed + within_group_index; reused across prompt groups' if args.real_training else 'legacy engine default',
        per_request_sampling_seeds=[x['sampling_seed'] for x in per_request_sampling] if per_request_sampling else None,
        input_path=str(args.prompts_json.resolve()), input_sha256=hashlib.sha256(args.prompts_json.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capture_batch_size=args.capture_batch_size, real_training=args.real_training,
        scope='32 prompt groups x 8 responses, natural EOS; capture is not a timing benchmark' if args.real_training else '32 unique short prompts; correctness/routes only, no representative timing claim')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    import numpy as np
    import sglang
    from routepack.capture_routes import save_engine_routes
    from routepack.prepare_b1 import prepare
    model = json.loads((Path(config['model_path']) / 'config.json').read_text())
    engine = sglang.Engine(**config)
    try:
        (args.output / 'resolved-server-args.json').write_text(json.dumps(dataclasses.asdict(engine.server_args), indent=2, default=str))
        directory = args.output / 'captured'
        rows = []
        inventory = None
        for start in range(0, len(prompts), args.capture_batch_size):
            end = min(start + args.capture_batch_size, len(prompts))
            outputs = engine.generate(input_ids=prompts[start:end], sampling_params=per_request_sampling[start:end] if per_request_sampling else sampling, return_logprob=True,
                logprob_start_len=-1, return_routed_experts=True, routed_experts_start_len=0)
            # Keep the old single-batch file layout; multi-batch paths are relative to captured/.
            batch_dir = directory if len(prompts) <= args.capture_batch_size else directory / f'batch-{start:04d}'
            batch_inventory = save_engine_routes(outputs, prompts[start:end], sources[start:end], batch_dir,
                model_version=config['model_path'], model_config=model)
            batch_rows = [json.loads(x) for x in (batch_dir / 'samples_captured.jsonl').read_text().splitlines()]
            for local, (out, row) in enumerate(zip(outputs, batch_rows)):
                i = start + local
                values = np.asarray([x[0] for x in out['meta_info']['output_token_logprobs']], dtype='<f4')
                if not 0 < len(values) <= args.response_tokens or values.shape != (row['response_length'],) or not np.isfinite(values).all():
                    raise ValueError('Missing, inconsistent or nonfinite response logprobs')
                if not args.real_training and len(values) != args.response_tokens:
                    raise ValueError('Short fixed-length capture ended unexpectedly')
                row.update(sample_id=f'captured:{i}', response_logprob_path=f'logprobs-{local}.npy',
                    response_logprob_bits_path=f'logprob-bits-{local}.npy', effective_logprob_temperature=1.0,
                    sampling_temperature=args.temperature, sampling_seed=per_request_sampling[i]['sampling_seed'] if per_request_sampling else None, finish_reason=out['meta_info'].get('finish_reason'), source_fixture_record=sources[i])
                np.save(batch_dir / row['response_logprob_path'], values, allow_pickle=False)
                np.save(batch_dir / row['response_logprob_bits_path'], values.view('<u4'), allow_pickle=False)
                prefix = batch_dir.relative_to(directory)
                for key in ('route_path', 'token_path', 'response_logprob_path', 'response_logprob_bits_path'):
                    row[key] = str(prefix / row[key])
                rows.append(row)
            if inventory is None:
                inventory = dict(batch_inventory)
            else:
                for key in ('sample_count', 'trace_coverage_count', 'raw_trace_bytes'):
                    inventory[key] += batch_inventory[key]
            (directory / 'samples_captured.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
            (directory / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
            print(json.dumps({'captured': end, 'total': len(prompts), 'raw_trace_bytes': inventory['raw_trace_bytes']}), flush=True)
            del outputs, batch_rows
        result = prepare(directory, args.output / 'rollout.pt')
        (args.output / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()
