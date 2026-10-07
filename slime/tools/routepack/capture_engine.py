"""One short SGLang route-capture run using a frozen A-stage Engine config.

Run on the allocated GPU node in the same runtime environment as the chosen
reference manifest. No training, optimizer, weight synchronization or profiling.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

from capture_routes import save_engine_routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-manifest', type=Path, required=True)
    parser.add_argument('--frozen-rollout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--response-tokens', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.response_tokens <= 128:
        raise ValueError('Preflight capture supports 1..128 response tokens')
    reference = json.loads(args.reference_manifest.read_text())
    config = dict(reference['requested_server_args'])
    if not config.get('enable_deterministic_inference'):
        raise ValueError('Use an aligned reference manifest')
    config['enable_return_routed_experts'] = True
    config['skip_server_warmup'] = True
    sampling = {**reference['sampling'], 'max_new_tokens': args.response_tokens, 'ignore_eos': True}
    import torch
    sources = torch.load(args.frozen_rollout, map_location='cpu', weights_only=True)['samples']
    prompts = [list(map(int, sample['tokens'][:-int(sample['response_length'])])) for sample in sources]
    model_config_path = Path(config['model_path']) / 'config.json'
    model_config = json.loads(model_config_path.read_text())
    if (model_config['num_hidden_layers'], model_config['num_experts_per_tok'], model_config['num_experts']) != (48, 8, 128):
        raise ValueError('Expected Qwen3-30B-A3B 48-layer, top8, 128-expert config')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'purpose': 'RoutePack B2 preflight capture, not a benchmark or training replay gate',
                'reference_manifest': str(args.reference_manifest.resolve()), 'requested_server_args': config,
                'sampling': sampling, 'source_rollout': str(args.frozen_rollout.resolve()),
                'source_rollout_sha256': hashlib.sha256(args.frozen_rollout.read_bytes()).hexdigest(),
                'model_config_sha256': hashlib.sha256(model_config_path.read_bytes()).hexdigest(),
                'model_weight_version_note': 'HF path only; checkpoint weight content hash and training match still pending',
                'prompt_lengths': list(map(len, prompts))}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    import sglang
    engine = sglang.Engine(**config)
    try:
        (args.output / 'resolved-server-args.json').write_text(json.dumps(dataclasses.asdict(engine.server_args), indent=2, default=str) + '\n')
        started = time.perf_counter()
        outputs = engine.generate(input_ids=prompts, sampling_params=sampling, return_logprob=True,
                                  logprob_start_len=-1, return_routed_experts=True, routed_experts_start_len=0)
        elapsed = time.perf_counter() - started
        inventory = save_engine_routes(outputs, prompts, sources, args.output / 'captured',
                                       model_version=config['model_path'], model_config=model_config)
        (args.output / 'capture-wall-time.json').write_text(json.dumps({'elapsed_s': elapsed,
            'use': 'capture diagnostic only; no warmup, includes route collection and serialization'}) + '\n')
        print(json.dumps(inventory), flush=True)
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()
