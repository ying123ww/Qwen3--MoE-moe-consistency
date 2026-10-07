"""Persist Engine.generate responses as B2 preflight data, not training evidence."""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np


def save_engine_routes(outputs, prompts, source_samples, directory, *, model_version, model_config):
    """Call after generate(return_logprob=True, return_routed_experts=True).

    Engine must be initialized with enable_return_routed_experts=True. This
    function has no engine lifecycle, profiler, or training responsibilities.
    """
    if not (len(outputs) == len(prompts) == len(source_samples)) or not outputs:
        raise ValueError('Mismatched/empty response, prompt and identity lists')
    layers = int(model_config['num_hidden_layers'])
    topk = int(model_config['num_experts_per_tok'])
    experts = int(model_config['num_experts'])
    prepared = []
    for i, (output, prompt, source) in enumerate(zip(outputs, prompts, source_samples)):
        meta = output['meta_info']
        encoded = meta.get('routed_experts')
        if not isinstance(encoded, str):
            raise ValueError(f'Missing base64 route trace for response {i}')
        token_logprobs = meta['output_token_logprobs']
        response = [int(row[1]) for row in token_logprobs]
        if len(response) != meta['completion_tokens']:
            raise ValueError('Output token identity coverage mismatch')
        if meta.get('prompt_tokens', len(prompt)) != len(prompt):
            raise ValueError('Prompt token coverage mismatch')
        tokens = list(map(int, prompt)) + response
        route = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.int32)
        expected_shape = (len(tokens) - 1, layers, topk)
        if route.size != np.prod(expected_shape):
            raise ValueError(f'Route coverage mismatch: {route.size} != {expected_shape}')
        route = route.reshape(expected_shape)
        if route.min() < 0 or route.max() >= experts:
            raise ValueError('Route IDs out of logical expert range')
        if np.any(np.diff(np.sort(route, axis=-1), axis=-1) == 0):
            raise ValueError('Duplicate logical expert IDs in topk')
        record = {'sample_id': f'{Path(directory).name}:{i}', 'source_sample_index': source['index'],
                  'rollout_id': source.get('rollout_id'), 'reward_group_id': source.get('group_index'),
                  'token_count': len(tokens), 'response_length': len(response), 'layer_ids': list(range(layers)),
                  'model_version': model_version, 'traced_token_count': len(tokens) - 1,
                  'route_path': f'routes-{i}.npy', 'token_path': f'tokens-{i}.npy',
                  'trace_hash': hashlib.sha256(route.astype('<i8').tobytes()).hexdigest(),
                  'token_hash': hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes()).hexdigest(),
                  'source': 'independent_engine_generate',
                  'grouping_note': 'source rollout/reward IDs retained as provenance; no RL reward or training schedule computed'}
        prepared.append((record, route, tokens))
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    for record, route, tokens in prepared:
        np.save(root / record['route_path'], route, allow_pickle=False)
        np.save(root / record['token_path'], np.asarray(tokens, dtype=np.int64), allow_pickle=False)
    (root / 'samples_captured.jsonl').write_text(''.join(json.dumps(item[0]) + '\n' for item in prepared))
    inventory = {'sample_count': len(prepared), 'trace_coverage_count': len(prepared),
                 'trace_shape_suffix': [layers, topk], 'raw_trace_bytes': sum(item[1].nbytes for item in prepared),
                 'training_topology_gate': 'not_run', 'alignment_replay_gate': 'not_run',
                 'actual_dispatch_route_gate': 'not_run', 'timing_use': 'capture overhead included; not performance benchmark'}
    (root / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return inventory
