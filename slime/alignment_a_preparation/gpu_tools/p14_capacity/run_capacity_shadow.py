"""Eager padded-expert checks against the already completed cap8/16 graph run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

sys.path.insert(0, str(Path.cwd() / 'slime/alignment_a_preparation/gpu_tools'))
import residual_rms_performance as guard
from resource_preflight import snapshot, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model-run', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'ai_hgx_03':
        raise RuntimeError('Use ai_hgx_03')
    if not json.loads((args.model_run / 'results.json').read_text())['complete']:
        raise RuntimeError('Cap8/16 full-model prerequisite incomplete')
    run = guard.PREP / 'gpu_runs' / args.run_id
    run.mkdir(exist_ok=False)
    lock = open(guard.PROJECT / '.a-tmp/router-tile-performance.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, guard.interrupted)
    signal.signal(signal.SIGINT, guard.interrupted)
    try:
        if guard.gpu_pids():
            raise RuntimeError('Other GPU task active')
        resources = snapshot(str(guard.PROJECT / '.a-tmp'))
        validate(resources, 512, 20)
        groups = [Path(x['path']) for x in resources['cgroups']
                  if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups) != 1:
            raise RuntimeError('Allocation memory cgroup missing')
        folder = Path(__file__).parent.resolve()
        env = guard.environment(run)
        env['PYTHONPATH'] = ':'.join([str(folder), str(guard.PREP / 'gpu_tools'),
                                    str(guard.PREP / 'gpu_tools'), env.get('PYTHONPATH', '')])
        nccl = guard.ENVROOT / 'lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2'
        env['SGLANG_NCCL_SO_PATH'] = str(nccl)
        env['LD_LIBRARY_PATH'] = str(nccl.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
        paths = sorted(folder.glob('*.py'))
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        guard.save(run / 'manifest.json', dict(source_sha256=hashes,
                   model_run=str(args.model_run), restore_keepalive=False,
                   scope='Eager padding expert real-row shadow and full-graph output comparison'))
        guard.p2p_preflight(run)
        summaries = {}
        for cap in (16, 8):
            reference_dir = args.model_run / f'cap-{cap}'
            reference = json.loads((reference_dir / 'results.json').read_text())
            output = run / f'cap-{cap}'
            output.mkdir()
            guard.command(run, f'shadow-{cap}', [str(guard.PYTHON),
                          str(folder / 'capacity_shadow_probe.py'), '--run', str(output),
                          '--capacity', str(cap), '--variant', reference['variant']],
                          env, groups[0] / 'memory.usage_in_bytes', timeout=2400)
            actual = json.loads((output / 'results.json').read_text())
            if not actual['complete'] or len(actual['cases']) != 10:
                raise RuntimeError('Incomplete shadow workload')
            for left, right in zip(reference['cases'], actual['cases'], strict=True):
                if left['case'] != right['case']:
                    raise RuntimeError('Workload identity mismatch')
                for a, b in zip(left['records'], right['records'], strict=True):
                    for field in ('prompt', 'response_length', 'tokens', 'logprob_bits',
                                  'route_sha256', 'route_shape'):
                        if a[field] != b[field]:
                            raise RuntimeError(f'Eager/full mismatch {left["case"]} {field}')
            proofs = [json.loads(p.read_text()) for p in
                      (output / 'graph-evidence').glob('fixed-shadow-rank-*.json')]
            if {p['rank'] for p in proofs} != set(range(8)):
                raise RuntimeError('Missing rank padding shadow evidence')
            if any(p['counts'].get('shadow_calls', 0) <= 0 or
                   p['counts'].get('shadow_elements', 0) <= 0 for p in proofs):
                raise RuntimeError('No actual expert shadow comparisons')
            summaries[cap] = proofs
        if not all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                   for p, h in hashes.items()):
            raise RuntimeError('Source changed during shadow run')
        guard.save(run / 'results.json', dict(complete=True, capacities=summaries,
                   cases_per_capacity=10, all_full_eager_tokens_routes_logprob_bits_equal=True))
        guard.save(run / 'state.json', dict(status='complete', pid=os.getpid(), time=time.time()))
    except BaseException as exc:
        guard.save(run / 'state.json', dict(status='failed', error=str(exc),
                                         pid=os.getpid(), time=time.time()))
        raise
    finally:
        guard.stop_worker()
        print('P14_SHADOW_GPU_PIDS=' + guard.gpu_pids(), flush=True)


if __name__ == '__main__':
    main()
