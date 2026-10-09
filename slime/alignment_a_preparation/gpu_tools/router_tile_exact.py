"""Independent Engine processes for the opt-in Qwen3 router tile exact gate."""
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

PREP = Path(__file__).resolve().parents[1]
ROOT = PREP.parent
sys.path.insert(0, str(ROOT / 'tools/routepack'))
from capture_routes import save_engine_routes


def cases(prompts):
    if len(prompts) != 8 or any(not p for p in prompts):
        raise ValueError('Expected the frozen eight nonempty prompts')
    return [('batch8', list(range(8)), prompts),
            ('batch3', [0, 1, 2], prompts[:3]),
            ('batch1', [0], prompts[:1]),
            ('variable3', [0, 1, 2], [p[:n] for p, n in zip(prompts[:3], [17, 63, 129])])]


def capture(args):
    # Set before importing SGLang so spawned ranks get the same mode.
    if args.disable_audit and args.mode != 'candidate':
        raise ValueError('Only candidate capture may disable audit')
    os.environ['SLIME_QWEN3_ROUTER_TILE_MODE'] = 'off'
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_MODE'] = 'off'
    prefix = 'SLIME_QWEN3_ROUTER_TILE' if args.component == 'router' else 'SLIME_QWEN3_RESIDUAL_RMS'
    os.environ[prefix + '_AUDIT'] = '0' if args.disable_audit else '1'
    os.environ[prefix + '_MODE'] = args.mode
    os.environ[prefix + '_AUDIT_DIR'] = str((args.output / 'audit').resolve())
    reference = json.loads(args.reference_manifest.read_text())
    config = dict(reference['requested_server_args'])
    if not config.get('enable_deterministic_inference'):
        raise ValueError('Reference must enable deterministic inference')
    config.update(enable_return_routed_experts=True, skip_server_warmup=True,
                  disable_cuda_graph=True, disable_prefill_cuda_graph=True)
    sampling = dict(temperature=0, max_new_tokens=args.response_tokens, ignore_eos=True)
    import torch
    sources = torch.load(args.frozen_rollout, map_location='cpu', weights_only=True)['samples']
    prompts = [list(map(int, s['tokens'][:-int(s['response_length'])])) for s in sources]
    workloads = cases(prompts)
    model_config_path = Path(config['model_path']) / 'config.json'
    model_config = json.loads(model_config_path.read_text())
    if tuple(model_config[k] for k in ('num_hidden_layers', 'num_experts_per_tok', 'num_experts', 'hidden_size')) != (48, 8, 128, 2048):
        raise ValueError('Expected the measured Qwen3-30B-A3B config')
    args.output.mkdir(parents=True, exist_ok=False)
    srt = ROOT.parent / 'experiments/sglang-main-sm80/python/sglang/srt'
    paths = [Path(__file__), ROOT / 'tools/routepack/capture_routes.py', model_config_path,
             srt / 'models/qwen3_moe.py', srt / 'models/qwen3_router_tile.py',
             srt / 'batch_invariant_ops/batch_invariant_ops.py', srt / 'layers/sampler.py',
             srt / 'batch_invariant_ops/fused_rmsnorm_candidate.py', srt / 'layers/layernorm.py']
    manifest = dict(component=args.component, mode=args.mode, audit_enabled=not args.disable_audit, requested_server_args=config, sampling=sampling,
                    effective_logprob_temperature=1.0,
                    logprob_rule='sampler greedy branch: log_softmax(unscaled logits)',
                    source_rollout=str(args.frozen_rollout.resolve()),
                    source_rollout_sha256=hashlib.sha256(args.frozen_rollout.read_bytes()).hexdigest(),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    cases=[dict(name=n, source_indices=i, prompt_lengths=list(map(len, p))) for n, i, p in workloads],
                    scope='Finite model exact gate. CUDA graphs disabled; audit overhead included; not a performance measurement.',
                    checkpoint_note='Checkpoint path/config frozen; weight shard hashes not computed.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    import sglang
    engine = sglang.Engine(**config)
    try:
        (args.output / 'resolved-server-args.json').write_text(json.dumps(dataclasses.asdict(engine.server_args), indent=2, default=str) + '\n')
        for name, indices, inputs in workloads:
            flushed = engine.flush_cache()
            if not (flushed if isinstance(flushed, bool) else flushed.success):
                raise RuntimeError(f'Cache flush failed before {name}')
            output = engine.generate(input_ids=inputs, sampling_params=sampling, return_logprob=True,
                                     logprob_start_len=-1, return_routed_experts=True, routed_experts_start_len=0)
            if len(output) != len(inputs):
                raise RuntimeError('Engine output count mismatch')
            directory = args.output / f'case-{name}' / 'captured'
            save_engine_routes(output, inputs, [sources[i] for i in indices], directory,
                               model_version=config['model_path'], model_config=model_config)
            metadata_path = directory / 'samples_captured.jsonl'
            records = [json.loads(line) for line in metadata_path.read_text().splitlines()]
            for i, (out, record) in enumerate(zip(output, records)):
                raw = out['meta_info']['output_token_logprobs']
                if len(raw) != args.response_tokens or not all(math.isfinite(float(x[0])) for x in raw):
                    raise RuntimeError(f'Invalid returned logprobs for {name}:{i}')
                values = np.asarray([x[0] for x in raw], dtype='<f4')
                record.update(response_logprob_path=f'logprobs-{i}.npy',
                              response_logprob_bits_path=f'logprob-bits-{i}.npy',
                              effective_logprob_temperature=1.0, sampling_temperature=0,
                              input_prompt_length=len(inputs[i]))
                np.save(directory / record['response_logprob_path'], values, allow_pickle=False)
                np.save(directory / record['response_logprob_bits_path'], values.view('<u4'), allow_pickle=False)
            metadata_path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            print(json.dumps(dict(case=name, samples=len(records), status='captured')), flush=True)
    finally:
        engine.shutdown()
    (args.output / 'complete.json').write_text(json.dumps(dict(mode=args.mode, cases=len(workloads))) + '\n')


def load_case(directory, name):
    root = directory / f'case-{name}' / 'captured'
    rows = [json.loads(s) for s in (root / 'samples_captured.jsonl').read_text().splitlines()]
    return [(r, {k: np.load(root / r[k], allow_pickle=False) for k in
                 ('route_path', 'token_path', 'response_logprob_path', 'response_logprob_bits_path')}) for r in rows]


def compare_samples(left, right, label):
    if len(left) != len(right):
        raise ValueError(f'{label}: sample count differs')
    failures = []
    for index, ((_, a), (_, b)) in enumerate(zip(left, right)):
        for key in a:
            equal = a[key].shape == b[key].shape and a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes()
            if not equal:
                failures.append(dict(label=label, sample=index, field=key,
                                     left_shape=list(a[key].shape), right_shape=list(b[key].shape)))
    return failures


def audit_summary(directory, mode, expected_ranks=8, expected_layers=48, component='router'):
    coverage = set()
    rows = 0
    compared = 0
    shapes = set()
    for path in sorted((directory / 'audit').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows += 1
            if row['mode'] != mode or row['mismatches'] or row.get('nonfinite_elements', 0):
                raise ValueError(f'Failed/mixed audit row in {path}: {row}')
            if row['eligible']:
                if mode == 'shadow' and row['compared_elements'] <= 0:
                    raise ValueError('Shadow row has no actual bit comparisons')
                coverage.add((row['rank'], row['layer'], row['kind']) if component == 'residual-rms' else (row['rank'], row['layer']))
                compared += row['compared_elements']
                shapes.add(tuple(row['a_shape']))
    expected = ({(r,l,k) for r in range(expected_ranks) for l in range(expected_layers)
                 for k in ('input','pre_mlp') if k != 'input' or l > 0}
                if component == 'residual-rms' else
                {(r,l) for r in range(expected_ranks) for l in range(expected_layers)})
    if coverage != expected:
        raise ValueError(f'{mode} rank/layer coverage mismatch: missing={sorted(expected - coverage)}, extra={sorted(coverage - expected)}')
    return dict(rows=rows, covered_rank_layers=len(coverage), compared_elements=compared,
                eligible_a_shapes=[list(x) for x in sorted(shapes)])


def compare(args):
    directories = dict(off=args.baseline, shadow=args.shadow, candidate=args.candidate)
    manifests = {}
    for mode, directory in directories.items():
        complete = json.loads((directory / 'complete.json').read_text())
        manifests[mode] = json.loads((directory / 'manifest.json').read_text())
        if complete != dict(mode=mode, cases=4) or manifests[mode]['mode'] != mode:
            raise ValueError(f'Incomplete or wrong-mode capture: {directory}')
    component = manifests['off'].get('component', 'router')
    if any(m.get('component', 'router') != component for m in manifests.values()):
        raise ValueError('Mixed exact gate components')
    # Paths for audit/mode are environment-only; actual Engine args must match.
    for mode in ('shadow', 'candidate'):
        for key in ('requested_server_args', 'sampling', 'source_rollout_sha256', 'source_sha256', 'cases'):
            if manifests[mode][key] != manifests['off'][key]:
                raise ValueError(f'Experiment manifest differs: {mode} {key}')
    failures = []
    for name in ('batch8', 'batch3', 'batch1', 'variable3'):
        reference = load_case(args.baseline, name)
        for mode in ('shadow', 'candidate'):
            failures.extend(compare_samples(reference, load_case(directories[mode], name), f'off-vs-{mode}:{name}'))
    batching_failures = []
    for mode, directory in directories.items():
        original = load_case(directory, 'batch8')
        batching_failures.extend(compare_samples(original[:3], load_case(directory, 'batch3'), f'{mode}:batch8-vs-batch3'))
        batching_failures.extend(compare_samples(original[:1], load_case(directory, 'batch1'), f'{mode}:batch8-vs-batch1'))
    result = dict(model_exact=not failures, batch_exact=not batching_failures,
                  failures=failures, batching_failures=batching_failures,
                  shadow_audit=audit_summary(args.shadow, 'shadow', component=component),
                  candidate_audit=(audit_summary(args.candidate, 'candidate', component=component)
                      if manifests['candidate'].get('audit_enabled', True)
                      else dict(disabled=True, validation='Compared full tokens/routes/logprob bits with off and audited shadow')),
                  limitation='Finite greedy workload, graph-disabled Engine; not cross-framework or performance validation.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if failures or batching_failures:
        raise RuntimeError('Router tile model/batch exact gate FAILED; inspect comparison JSON')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--component', choices=['router', 'residual-rms'], default='router')
    cap.add_argument('--disable-audit', action='store_true', help='Verify the candidate timing path using complete model artifacts')
    cap.add_argument('--mode', choices=['off', 'shadow', 'candidate'], required=True)
    cap.add_argument('--reference-manifest', type=Path, required=True)
    cap.add_argument('--frozen-rollout', type=Path, required=True)
    cap.add_argument('--output', type=Path, required=True)
    cap.add_argument('--response-tokens', type=int, default=16)
    cmp = sub.add_parser('compare')
    cmp.add_argument('--baseline', type=Path, required=True)
    cmp.add_argument('--shadow', type=Path, required=True)
    cmp.add_argument('--candidate', type=Path, required=True)
    cmp.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'capture':
        if not 1 <= args.response_tokens <= 128:
            parser.error('response tokens must be in 1..128')
        capture(args)
    else:
        compare(args)


if __name__ == '__main__':
    main()
