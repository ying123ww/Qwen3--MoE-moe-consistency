"""Isolated full-graph startup attempt; clean up only this worker group."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

import residual_rms_performance as guard
from resource_preflight import snapshot, validate


def main():
    if socket.gethostname() != 'ai_hgx_03':
        raise RuntimeError('Use ai_hgx_03')
    signal.signal(signal.SIGTERM, guard.interrupted)
    signal.signal(signal.SIGINT, guard.interrupted)
    run = guard.PREP / 'gpu_runs/full-graph-baseline-hgx03-20261011-01'
    run.mkdir(exist_ok=False)
    lock = open(guard.PROJECT / '.a-tmp/router-tile-performance.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if guard.gpu_pids():
            raise RuntimeError('Other GPU task exists')
        resources = snapshot(str(guard.PROJECT / '.a-tmp'))
        validate(resources, 512, 20)
        groups = [Path(x['path']) for x in resources['cgroups']
                  if Path(x['path']).name.startswith('job_') and 'memory' in x['path']]
        if len(groups) != 1:
            raise RuntimeError('Allocation memory cgroup missing')
        guard.save(run / 'resources.json', resources)
        env = guard.environment(run)
        env['PYTHONPATH'] = str(guard.PREP / 'gpu_tools') + ':' + env.get('PYTHONPATH', '')
        env['SGLANG_MEMORY_SAVER_CUDA_GRAPH'] = '0'
        env['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE'] = '0'
        source = guard.PREP / 'gpu_tools/full_graph_baseline_probe.py'
        guard.save(run / 'manifest.json', dict(host=socket.gethostname(), pid=os.getpid(),
            job_id=os.environ.get('SLURM_JOB_ID'), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            backend='full', segmented_graph=False, deepep_mode='normal', deterministic=False,
            graph_scope='Full model decode forward, prefill eager', restore_keepalive=False,
            protocol=dict(warmup=3, steps=1, response_tokens=128)))
        guard.p2p_preflight(run)
        output = run / 'baseline-full'
        output.mkdir()
        guard.command(run, 'baseline-full', [str(guard.PYTHON), str(source), '--run', str(output),
            '--variant', 'baseline', '--residual-rms-mode', 'off', '--router-tile-mode', 'off',
            '--warmup', '3', '--steps', '1', '--response-tokens', '128'], env,
            groups[0] / 'memory.usage_in_bytes', timeout=1200)
        resolved = json.loads((output / 'resolved-server-args.json').read_text())
        assert resolved['cuda_graph_config']['decode']['backend'] == 'full'
        guard.save(run / 'results.json', dict(status='benchmark_complete',
            summary=json.loads((output / 'summary.json').read_text())))
        guard.save(run / 'state.json', dict(status='complete', pid=os.getpid(), time=time.time()))
    except BaseException as exc:
        guard.save(run / 'state.json', dict(status='failed', error=str(exc), pid=os.getpid(), time=time.time()))
        raise
    finally:
        guard.stop_worker()
        print('FULL_GRAPH_PROBE_REMAINING_GPU_PIDS=' + guard.gpu_pids(), flush=True)


if __name__ == '__main__':
    main()
