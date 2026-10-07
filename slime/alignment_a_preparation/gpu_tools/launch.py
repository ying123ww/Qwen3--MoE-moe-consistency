"""Launch one explicitly requested A-stage experiment on ai_hgx_03."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import time

import yaml
from resource_preflight import snapshot, validate, GIB

ROOT = Path('/public/home/yingxy2024/moe-consistency/slime')
PREP = ROOT / 'alignment_a_preparation'
ENV = Path('/public/home/yingxy2024/.conda/envs/slime')
CUDA = Path('/tmp/slime-cuda-alignment-a-20260909')
RAY_TMP = os.environ.get('ALIGN_A_RAY_TMP', '/tmp/ray-alignment-a-20260909')
ADDRESS = 'http://127.0.0.1:8267'


def check_memory_guard(resources):
    state = Path(os.environ['ALIGN_A_MEMORY_GUARD'])
    info = json.loads(state.read_text())
    jobs = {Path(g['path']).name for g in resources['cgroups'] if Path(g['path']).name.startswith('job_')}
    if info['job'] not in jobs or info['triggered'] or info.get('error'):
        raise RuntimeError('Memory guard is triggered or belongs to another job')
    if not 0 <= time.time() - info['time'] <= 5 or info['threshold_bytes'] != 900 * GIB:
        raise RuntimeError('Memory guard heartbeat stale or cutoff is not 900 GiB')
    os.kill(info['pid'], 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['restore', 'aligned', 'native'], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--p1-train-profile', action='store_true', help='Frozen-data training-only third RPC capture with relocated writable caches')
    parser.add_argument('--p1-profile', action='store_true', help='Bounded third-iteration profiling; requires online timing')
    parser.add_argument('--online-timing', action='store_true', help='True online RL loop, first iteration warmup')
    parser.add_argument('--start-cluster', action='store_true', help='Explicitly start a new isolated cluster; never recover automatically')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--min-memory-gib', type=int, default=512, help='Conservative headroom gate, not a measured peak')
    parser.add_argument('--min-disk-gib', type=int, default=20)
    args = parser.parse_args()
    if args.p1_train_profile and (args.online_timing or args.p1_profile or args.mode == 'restore' or not args.timing or args.steps < 3):
        raise ValueError('Training profile requires aligned/native, --timing, >=3 steps and no online profile')
    if args.p1_profile and (not args.online_timing or args.steps < 3):
        raise ValueError('P1 requires online timing and at least three iterations')
    if args.online_timing and (args.mode == 'restore' or args.timing):
        raise ValueError('Online timing requires aligned/native mode and excludes fixed-data timing')
    if socket.gethostname().split('.')[0] != 'ai_hgx_03':
        raise RuntimeError('Only run on the authorized ai_hgx_03 node')
    if not args.run_id.replace('-', '').replace('_', '').isalnum() or args.steps < 1:
        raise ValueError('Invalid run ID or step count')
    if args.min_memory_gib < 1 or args.min_disk_gib < 1:
        raise ValueError('Resource thresholds must be positive')
    # Hold this lock through submission; two launchers must not race.
    lock = open('/tmp/alignment-a-launch.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = PREP / 'gpu_runs' / args.run_id
        run.mkdir(exist_ok=False)
        resources = snapshot(RAY_TMP)
        resources['thresholds_gib'] = {'memory': args.min_memory_gib, 'disk': args.min_disk_gib}
        (run / 'resource-preflight.json').write_text(json.dumps(resources, indent=2) + '\n')
        validate(resources, args.min_memory_gib, args.min_disk_gib)
        # Workspace/devshm headroom does not imply the node root filesystem is writable.
        if not args.p1_train_profile and shutil.disk_usage("/").free < GIB:
            raise RuntimeError("Node root filesystem has less than 1 GiB available; restore space before launching")
        if args.preflight_only:
            print(json.dumps(resources, indent=2))
            return
        check_memory_guard(resources)
        env = yaml.safe_load((PREP / 'source_snapshots/v89_runtime_env.yaml.txt').read_text())['env_vars']
        env = {k: v.replace('/tmp/slime-cuda-runtime-qwen-align-v41', str(CUDA)) for k, v in env.items()}
        env['PYTHONPATH'] = str(PREP / 'gpu_tools') + ':' + env['PYTHONPATH']
        task_tmp = ROOT.parent / '.a-tmp' / args.run_id
        task_tmp.mkdir(parents=True, exist_ok=False)
        env.update({key: str(task_tmp) for key in ('TMPDIR', 'TMP', 'TEMP')})
        env.update({'ENABLE_ROUTING_REPLAY': '0', 'OPEN_TRAINING_INT4_FAKE_QAT_FLAG': '0',
                    'ALIGN_A_RUN_DIR': str(run), 'ALIGN_A_VARIANT': args.mode,
                    'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'})
        # Node-local services must not inherit a login-node proxy tunnel.
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
            env[key] = ''
        env['NO_PROXY'] = env['no_proxy'] = '*'
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'NO_PROXY', 'no_proxy']:
            os.environ[key] = env[key]
        options = json.loads((PREP / 'v89_cli_static.json').read_text())['options_last_value']
        substitutions = {'$SLIME_DIR': str(ROOT), '$HF_MODEL': '/storage/data/yingxy2024/Models/Qwen3-30B-A3B',
                         '$CKPT': '/storage/data/yingxy2024/Models/Qwen3-30B-A3B_torch_dist',
                         '$ALIGN_ROOT': str(run), '$DEEPEP_CONFIG': env['SLIME_QWEN3_FIXED_DEEPEP_CONFIG']}
        for key, values in options.items():
            for old, new in substitutions.items():
                values = [v.replace(old, new) for v in values]
            options[key] = values
        if args.mode == 'restore':
            options['--num-rollout'] = ['3']
        elif not args.online_timing:
            options = {k: v for k, v in options.items() if not k.startswith('--sglang-')}
            options['--load-debug-rollout-data'] = ['/public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt']
            options['--lr'] = ['0']
            options['--num-rollout'] = [str(args.steps + 1)]
        if args.mode == 'native':
            for key in ['--custom-megatron-before-log-prob-hook-path', '--custom-megatron-before-train-step-hook-path', '--qwen3-alignment-layers']:
                options.pop(key, None)
            for key in list(env):
                if key.startswith('MEGATRON_USE_SGLANG_'):
                    env[key] = '0'
        if args.mode != 'restore' and not args.online_timing:
            options['--custom-megatron-before-train-step-hook-path'] = ['alignment_a_probe.before_train_step']
        if args.timing:
            if args.mode == 'restore':
                raise ValueError('Restore uses original correctness path')
            options.pop('--save-debug-train-data', None)
            options.pop('--save-debug-rollout-data', None)
            options['--custom-megatron-before-train-step-hook-path'] = ['alignment_a_probe.before_train_step']
        if args.online_timing:
            options['--num-rollout'] = [str(args.steps + 1)]
            # Keep real lr=1e-6 and the historical optimizer/synchronization setup.
            for key in ['--save-debug-train-data', '--save-debug-rollout-data', '--sglang-enable-fp32-moe-router']:
                options.pop(key, None)
            options['--custom-rollout-log-function-path'] = ['online_timing_helpers.log_workload']
            env['SLIME_RL_ITERATION_TIMING_FILE'] = str(run / 'iterations.jsonl')
            env['TVM_FFI_CACHE_DIR'] = '/dev/shm/alignment-a-tvm-ffi-999484'
            if args.mode == 'native':
                for key in ['--deterministic-mode', '--sglang-enable-deterministic-inference', '--sglang-deepep-config']:
                    options.pop(key, None)
                env['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE'] = '0'
                # Remove optional deterministic restrictions for this configured baseline.
                for key in ['CUBLAS_WORKSPACE_CONFIG', 'NVTE_ALLOW_NONDETERMINISTIC_ALGO']:
                    env.pop(key, None)
        if args.p1_train_profile:
            cache = Path('/dev/shm') / ('p1-cache-' + args.run_id)
            cache.mkdir(exist_ok=False)
            for key, subdir in {'TVM_FFI_CACHE_DIR':'tvm','TRITON_CACHE_DIR':'triton','TORCHINDUCTOR_CACHE_DIR':'inductor','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg'}.items():
                target = cache / subdir
                target.mkdir()
                env[key] = str(target)
            for folder in (task_tmp, cache, run, Path(RAY_TMP)):
                probe = folder / ('write-probe-' + args.run_id)
                probe.write_bytes(b'ok')
                probe.unlink()
            env['SLIME_P1_PROFILE_DIR'] = str(run / 'profiles')
            env['SLIME_P1_PROFILE_ROLLOUT'] = '3'
            env['SLIME_P1_PROFILE_STAGES'] = 'train'
            options['--debug-train-only'] = []
        if args.p1_profile:
            env['SLIME_P1_PROFILE_DIR'] = str(run / 'profiles')
            env['SLIME_P1_PROFILE_ROLLOUT'] = '3'
            env['SGLANG_PROFILE_WITH_STACK'] = '0'
        argv = [str(ENV / 'bin/python'), str(ROOT / 'train.py')]
        for key, values in options.items():
            argv.extend([key, *values])
        if any('$' in value for value in argv):
            raise RuntimeError('Unexpanded variable in argv')
        (run / 'argv.json').write_text(json.dumps(argv, indent=2) + '\n')
        (run / 'runtime_env.json').write_text(json.dumps({'env_vars': env}, indent=2) + '\n')
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=15)
        if gpu.strip():
            raise RuntimeError('GPUs occupied; refusing to overlap experiment')
        (run / 'gpu-before.csv').write_text(subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv'], text=True, timeout=15))
        for d in [CUDA / 'bin', CUDA / 'lib64/stubs']:
            d.mkdir(parents=True, exist_ok=True)
        links = {'bin/nvcc': ENV / 'bin/nvcc', 'bin/cicc': ENV / 'nvvm/bin/cicc',
                 'include': ENV / 'targets/x86_64-linux/include', 'nvvm': ENV / 'nvvm',
                 'lib64/libcudart.so': ENV / 'targets/x86_64-linux/lib/libcudart.so',
                 'lib64/stubs/libcuda.so': Path('/lib64/libcuda.so')}
        for name, target in links.items():
            dest = CUDA / name
            if not dest.is_symlink():
                dest.symlink_to(target)
        # No broad exception fallback: an unresponsive cluster is not an absent cluster.
        manifest = Path(RAY_TMP) / 'alignment-a-resources.json'
        if args.start_cluster:
            if Path(RAY_TMP).exists():
                raise RuntimeError('Existing Ray directory: reconcile the old cluster before starting another')
            for port in (6387, 8267):
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', port))
            start_env = dict(os.environ, PATH=env['PATH'], RAY_TMPDIR=RAY_TMP)
            with (run / 'ray-start.log').open('w') as log:
                subprocess.run([str(ENV / 'bin/ray'), 'start', '--head', '--node-ip-address', '127.0.0.1',
                                '--port', '6387', '--num-gpus', '8', '--num-cpus', str(resources['ray_cpus']),
                                '--object-store-memory', str(4 * GIB), '--disable-usage-stats',
                                '--dashboard-host', '127.0.0.1', '--dashboard-port', '8267', '--temp-dir', RAY_TMP],
                               env=start_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
            manifest.write_text(json.dumps({'ray_cpus': resources['ray_cpus'], 'cpu_affinity': resources['cpu_affinity']}) + '\n')
        if not manifest.exists():
            raise RuntimeError('Cluster lacks the corrected resource manifest; reconcile it before reuse')
        expected = json.loads(manifest.read_text())
        if expected != {k: resources[k] for k in ('ray_cpus', 'cpu_affinity')}:
            raise RuntimeError('Cluster CPU allocation differs from current allocation')
        from ray.job_submission import JobSubmissionClient
        class BoundedClient(JobSubmissionClient):
            def _do_request(self, method, endpoint, **kwargs):
                kwargs.setdefault('timeout', (5, 15))
                return super()._do_request(method, endpoint, **kwargs)
    
        client = BoundedClient(ADDRESS)
        active = [j for j in client.list_jobs() if j.status in ('RUNNING', 'PENDING')]
        if active:
            raise RuntimeError('Experiment Ray cluster already has an active job')
        uncertain = Path(RAY_TMP) / 'submission-uncertain.json'
        if uncertain.exists():
            raise RuntimeError('Previous submission outcome unknown; reconcile its job ID before another submission')
        uncertain.write_text(json.dumps({'submission_id': args.run_id}) + '\n')
        check_memory_guard(resources)
        job_id = client.submit_job(entrypoint=shlex.join(argv), submission_id=args.run_id, runtime_env={'env_vars': env})
        (run / 'submission.json').write_text(json.dumps({'job_id': job_id, 'address': ADDRESS, 'mode': args.mode, 'timing': args.timing}, indent=2) + '\n')
        uncertain.unlink()
        print(json.dumps({'job_id': job_id, 'run_dir': str(run)}), flush=True)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
