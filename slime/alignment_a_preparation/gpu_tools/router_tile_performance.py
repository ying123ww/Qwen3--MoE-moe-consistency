"""Exact-gated, audit-free router tile comparison on an existing eight-GPU allocation."""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time

from resource_preflight import snapshot, validate

PREP = Path(__file__).resolve().parents[1]
PROJECT = PREP.parent.parent
PYTHON = Path('/public/home/yingxy2024/.conda/envs/slime/bin/python')
ENVROOT = PYTHON.parent.parent
FROZEN = Path('/public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt')
REFERENCE = PREP / 'gpu_runs/a-matmul-attribution-20260912-01/manifest.json'
ACTIVE = None
STOP = False


def save(path, value):
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def gpu_pids():
    return subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
        text=True, timeout=30).strip()


def p2p_preflight(run):
    reports = {}
    unsupported = []
    for capability in ('r', 'w'):
        raw = subprocess.check_output(['nvidia-smi', 'topo', '-p2p', capability], text=True, timeout=30)
        clean = re.sub(r'\x1b\[[0-9;]*m', '', raw)
        rows = {}
        for line in clean.splitlines():
            cells = line.split()
            if len(cells) == 9 and re.fullmatch(r'GPU[0-7]', cells[0]):
                rows[int(cells[0][3:])] = cells[1:]
        if set(rows) != set(range(8)):
            raise RuntimeError('Cannot establish the eight-GPU P2P matrix')
        for source, cells in rows.items():
            for destination, status in enumerate(cells):
                if source != destination and status != 'OK':
                    unsupported.append(dict(capability=capability, source=source, destination=destination, status=status))
        reports[capability] = raw
    save(run / 'p2p-preflight.json', dict(reports=reports, unsupported=unsupported))
    if unsupported:
        affected = sorted({item[k] for item in unsupported for k in ('source', 'destination')})
        raise RuntimeError('DeepEP requires working P2P; unsupported pairs involve GPUs ' + str(affected))


def stop_worker():
    if ACTIVE is None:
        return
    for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 10)):
        try:
            os.killpg(ACTIVE.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            try:
                os.killpg(ACTIVE.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.2)


def interrupted(signum, frame):
    global STOP
    STOP = True


def environment(run, resume=False):
    env = dict(os.environ)
    env.update(json.loads((PREP / 'gpu_runs/a-restore-20260909-06/runtime_env.json').read_text())['env_vars'])
    for key in list(env):
        if key.startswith('MEGATRON_USE_SGLANG_') or key.startswith('ALIGN_A_'):
            env.pop(key)
    # Recreate only the CUDA layout required by this node's existing Conda install.
    cuda = run / 'cuda'
    cuda.mkdir(exist_ok=resume)
    for name, target in {
        'bin': ENVROOT / 'bin', 'nvvm': ENVROOT / 'nvvm',
        'include': ENVROOT / 'targets/x86_64-linux/include',
        'lib64': ENVROOT / 'targets/x86_64-linux/lib',
    }.items():
        link = cuda / name
        if link.exists():
            if not resume or link.resolve() != target.resolve():
                raise RuntimeError('Unexpected CUDA layout: ' + str(link))
        else:
            link.symlink_to(target, target_is_directory=True)
    env.update(CUDA_HOME=str(cuda), CUDACXX=str(ENVROOT / 'bin/nvcc'),
               CPATH=str(cuda / 'include'), PATH=str(ENVROOT / 'nvvm/bin') + ':' + env['PATH'],
               SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE='1')
    scratch = PROJECT / '.a-tmp' / run.name
    scratch.mkdir(exist_ok=resume)
    env.update({key: str(scratch) for key in ('TMPDIR', 'TMP', 'TEMP')})
    cache = Path('/dev/shm') / run.name
    cache.mkdir(exist_ok=resume)
    for key, name in {
        'TVM_FFI_CACHE_DIR': 'tvm', 'TRITON_CACHE_DIR': 'triton',
        'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'CUDA_CACHE_PATH': 'cuda', 'XDG_CACHE_HOME': 'xdg',
    }.items():
        destination = cache / name
        destination.mkdir(exist_ok=resume)
        env[key] = str(destination)
    env.update(SLIME_QWEN3_ROUTER_TILE_MODE='off', SLIME_QWEN3_ROUTER_TILE_AUDIT='1')
    env.pop('SLIME_QWEN3_ROUTER_TILE_AUDIT_DIR', None)
    save(run / ('runtime_env-resume.json' if resume else 'runtime_env.json'), {'env_vars': {k: env[k] for k in (
        'PATH', 'PYTHONPATH', 'CUDA_HOME', 'CUDACXX', 'CPATH', 'LD_LIBRARY_PATH',
        'TMPDIR', 'TVM_FFI_CACHE_DIR', 'TRITON_CACHE_DIR', 'SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE',
        'SLIME_QWEN3_FIXED_DEEPEP_CONFIG', 'NCCL_ALGO', 'NCCL_NVLS_ENABLE', 'NCCL_P2P_LEVEL')}})
    return env


def command(run, label, argv, env, memory_file, timeout=3600):
    global ACTIVE
    if STOP:
        raise RuntimeError('Stop requested')
    if gpu_pids():
        raise RuntimeError('GPU process exists before ' + label)
    save(run / 'state.json', dict(status='running', phase=label, driver_pid=os.getpid(),
                                 driver_pgid=os.getpgrp(), time=time.time()))
    log_path = run / (label + '.log')
    with log_path.open('w') as log:
        ACTIVE = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        save(run / (label + '-launch.json'), dict(argv=argv, pid=ACTIVE.pid, pgid=ACTIVE.pid,
                                                log=str(log_path), time=time.time()))
        started = time.monotonic()
        try:
            while ACTIVE.poll() is None:
                usage = int(memory_file.read_text())
                save(run / 'memory-guard.json', dict(usage_bytes=usage, threshold_bytes=900 * 1024**3,
                                                   worker_pgid=ACTIVE.pid, time=time.time()))
                if usage >= 900 * 1024**3:
                    raise RuntimeError('Memory guard exceeded; stopping only this worker process group')
                if STOP or time.monotonic() - started > timeout:
                    raise RuntimeError('Worker interrupted or timed out: ' + label)
                time.sleep(2)
            code = ACTIVE.returncode
            save(run / (label + '-exit.json'), dict(returncode=code, elapsed_s=time.monotonic() - started))
            if code:
                raise RuntimeError(label + ' failed; see ' + str(log_path))
        finally:
            stop_worker()
            ACTIVE = None
    deadline = time.monotonic() + 60
    while gpu_pids():
        if time.monotonic() > deadline:
            raise RuntimeError('GPU memory still occupied after ' + label)
        time.sleep(2)
    print(json.dumps(dict(phase=label, status='complete')), flush=True)



def verify_resume(run, args):
    previous = json.loads((run / 'manifest.json').read_text())
    state = json.loads((run / 'state.json').read_text())
    if previous['host'] != socket.gethostname() or state['status'] != 'failed':
        raise RuntimeError('Resume requires this host and a reconciled failed run')
    if (previous['protocol']['warmup'], previous['protocol']['steps']) != (args.warmup, args.steps):
        raise RuntimeError('Resume protocol differs')
    comparison = json.loads((run / 'exact-comparison.json').read_text())
    if not comparison['model_exact'] or not comparison['batch_exact']:
        raise RuntimeError('Cannot resume without a passing model exact gate')
    capture = json.loads((run / 'exact-off/manifest.json').read_text())
    for name, digest in capture['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Model/gate source changed: ' + name)
    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != capture['source_rollout_sha256']:
        raise RuntimeError('Frozen rollout changed')
    for name, digest in previous['source_sha256'].items():
        if Path(name).name not in ('router_tile_performance.py', 'inference_benchmark.py'):
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Numerical source changed: ' + name)
    for directory in run.glob('perf-*'):
        summary = directory / 'summary.json'
        if summary.exists():
            value = json.loads(summary.read_text())
            if (value['formal_batches'], value['warmup_batches']) != (args.steps, args.warmup):
                raise RuntimeError('Completed measurement protocol differs')
    save(run / 'resume-validation.json', dict(
        numerical_sources_unchanged=True, model_exact_preserved=True,
        control_change='Only bounded successful-cache-flush retries outside timed generate and checkpoint resume',
        preserved_original_manifest=True, time=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not args.run_id.replace('-', '').replace('_', '').isalnum() or not args.run_id.startswith('router-'):
        parser.error('Use a router- prefixed simple run ID')
    if args.steps < 1 or args.warmup < 1:
        parser.error('Positive step counts required')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    lock = open(PROJECT / '.a-tmp/router-tile-performance.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = PREP / 'gpu_runs' / args.run_id
    run.mkdir(exist_ok=args.resume)
    if args.resume:
        verify_resume(run, args)
    state = snapshot(str(PROJECT / '.a-tmp'))
    validate(state, 512, 20)
    groups = [Path(g['path']) for g in state['cgroups']
              if Path(g['path']).name.startswith('job_') and 'memory' in g['path']]
    if len(groups) != 1:
        raise RuntimeError('Cannot establish allocation memory cgroup')
    save(run / ('resources-resume.json' if args.resume else 'resources.json'), state)
    env = environment(run, resume=args.resume)
    sources = [Path(__file__), PREP / 'gpu_tools/inference_benchmark.py',
               PREP / 'gpu_tools/router_tile_exact.py',
               PROJECT / 'experiments/sglang-main-sm80/python/sglang/srt/models/qwen3_router_tile.py',
               PROJECT / 'experiments/sglang-main-sm80/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py']
    save(run / ('resume-manifest.json' if args.resume else 'manifest.json'), dict(host=socket.gethostname(), pid=os.getpid(), pgid=os.getpgrp(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol=dict(warmup=args.warmup, steps=args.steps, response_tokens=128,
                      rounds=[['baseline', 'aligned', 'candidate'], ['aligned', 'candidate', 'baseline'],
                              ['candidate', 'baseline', 'aligned']]),
        limitation='Finite greedy exact gate; repeated complete generate timing. No cross-framework exact claim.'))
    try:
        p2p_preflight(run)
        if args.preflight_only:
            save(run / 'state.json', dict(status='preflight-passed', pid=os.getpid(), time=time.time()))
            return
        if not args.resume:
            command(run, 'router-microcheck', [str(PYTHON), str(PREP / 'gpu_tools/benchmark_router_timing.py'), '--output', str(run / 'router-microbenchmark.json')], env, groups[0] / 'memory.usage_in_bytes', timeout=900)
            for mode in ('off', 'shadow', 'candidate'):
                argv = [str(PYTHON), str(PREP / 'gpu_tools/router_tile_exact.py'), 'capture',
                        '--mode', mode, '--reference-manifest', str(REFERENCE), '--frozen-rollout', str(FROZEN),
                        '--output', str(run / ('exact-' + mode)), '--response-tokens', '16']
                if mode == 'candidate':
                    argv.append('--disable-audit')
                command(run, 'exact-' + mode, argv, env, groups[0] / 'memory.usage_in_bytes')
            command(run, 'exact-compare',
                    [str(PYTHON), str(PREP / 'gpu_tools/router_tile_exact.py'), 'compare',
                     '--baseline', str(run / 'exact-off'), '--shadow', str(run / 'exact-shadow'),
                     '--candidate', str(run / 'exact-candidate'), '--output', str(run / 'exact-comparison.json')],
                    env, groups[0] / 'memory.usage_in_bytes', timeout=180)
        comparisons = []
        rounds = [('baseline', 'aligned', 'candidate'), ('aligned', 'candidate', 'baseline'),
                  ('candidate', 'baseline', 'aligned')]
        for index, order in enumerate(rounds, 1):
            row = {'round': index}
            for name in order:
                destination = run / ('perf-%d-%s' % (index, name))
                exit_file = run / (destination.name + '-exit.json')
                if args.resume and (destination / 'summary.json').exists() and exit_file.exists():
                    if json.loads(exit_file.read_text())['returncode'] == 0:
                        row[name] = json.loads((destination / 'summary.json').read_text())
                        continue
                if destination.exists():
                    if not args.resume:
                        raise RuntimeError('Unreconciled measurement directory')
                    failed_name = destination.name + '-failed-before-resume-' + str(int(time.time()))
                    destination.rename(run / failed_name)
                    for suffix in ('.log', '-exit.json', '-launch.json'):
                        old_file = run / (destination.name + suffix)
                        if old_file.exists():
                            old_file.rename(run / (failed_name + suffix))
                destination.mkdir()
                worker_env = dict(env)
                worker_env['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE'] = '0' if name == 'baseline' else '1'
                argv = [str(PYTHON), str(PREP / 'gpu_tools/inference_benchmark.py'),
                        '--run', str(destination), '--variant', 'baseline' if name == 'baseline' else 'aligned',
                        '--router-tile-mode', 'candidate' if name == 'candidate' else 'off',
                        '--warmup', str(args.warmup), '--steps', str(args.steps), '--response-tokens', '128']
                command(run, destination.name, argv, worker_env, groups[0] / 'memory.usage_in_bytes')
                row[name] = json.loads((destination / 'summary.json').read_text())
            row['aligned_over_baseline_percent'] = (row['aligned']['median_s'] / row['baseline']['median_s'] - 1) * 100
            row['candidate_over_baseline_percent'] = (row['candidate']['median_s'] / row['baseline']['median_s'] - 1) * 100
            row['candidate_latency_reduction_percent'] = (1 - row['candidate']['median_s'] / row['aligned']['median_s']) * 100
            comparisons.append(row)
            save(run / 'comparison.json', dict(rounds=comparisons, complete=False))
        result = dict(rounds=comparisons, complete=True, exact=json.loads((run / 'exact-comparison.json').read_text()),
                      median_candidate_latency_reduction_percent=statistics.median(
                          r['candidate_latency_reduction_percent'] for r in comparisons),
                      note='Whole blocking generate, 8 requests, 1218 input / 1024 output tokens; no profiler/audit in timed runs.')
        save(run / 'comparison.json', result)
        save(run / 'state.json', dict(status='complete', pid=os.getpid(), time=time.time()))
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        stop_worker()
        save(run / 'state.json', dict(status='failed', pid=os.getpid(), error=str(exc), time=time.time()))
        raise


if __name__ == '__main__':
    main()

