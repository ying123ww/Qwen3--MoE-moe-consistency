"""Isolated local-Ray B1 gate for the prepared batch; forward only."""
import argparse
import json
import os
import resource
from pathlib import Path
import runpy
import shutil
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'alignment_a_preparation/gpu_runs/routepack-b1-20260912-02'


def main():
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cuda-home', type=Path, required=True)
    p.add_argument('--min-memory-gib', type=int, default=512)
    p.add_argument('--global-batch-size', type=int, default=32)
    p.add_argument('--rollout-batch-size', type=int, default=16)
    p.add_argument('--n-samples-per-prompt', type=int, default=2)
    p.add_argument('--max-tokens-per-gpu', type=int, default=512)
    p.add_argument('--expected-samples', type=int, default=32)
    p.add_argument('--rollout-max-response-len', type=int, default=128)
    p.add_argument('--rollout-max-context-len', type=int, default=512)
    p.add_argument('--balance-data', action='store_true')
    p.add_argument('--profile-directory', type=Path, help='Place large B1 arrays on node-local disk')
    p.add_argument('--schedule-manifest', type=Path)
    p.add_argument('--execution-mode', choices=('gate', 'timing'), default='gate')
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--expected-host', default='ai_hgx_05', choices=('ai_hgx_05', 'ai_hgx_03'))
    p.add_argument('--component-profile', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if socket.gethostname().split('.')[0] != a.expected_host:
        raise RuntimeError(f'Expected host {a.expected_host}')
    if min(a.global_batch_size, a.rollout_batch_size, a.n_samples_per_prompt,
           a.max_tokens_per_gpu, a.expected_samples, a.rollout_max_response_len, a.rollout_max_context_len) < 1:
        raise ValueError('Batch and capacity controls must be positive')
    if a.rollout_batch_size * a.n_samples_per_prompt != a.expected_samples:
        raise ValueError('Rollout prompt count times samples per prompt must match expected samples')
    if a.expected_samples != a.global_batch_size:
        raise ValueError('This forward gate expects one complete global batch')
    profile = (a.profile_directory or (a.output / 'b1')).resolve()
    if profile.exists():
        raise FileExistsError(profile)
    env = json.loads((OLD / 'runtime_env.json').read_text())['env_vars']
    env = {k: v.replace('/tmp/slime-cuda-alignment-a-20260909', str(a.cuda_home)) for k, v in env.items()}
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
        env['PYTORCH_CUDA_ALLOC_CONF'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if os.environ.get('ROUTEPACK_CHECKPOINT_CPU_OFFLOAD'):
        env['ROUTEPACK_CHECKPOINT_CPU_OFFLOAD'] = os.environ['ROUTEPACK_CHECKPOINT_CPU_OFFLOAD']
    if a.component_profile:
        if a.execution_mode != 'timing':
            raise ValueError('Component profiling requires timing mode')
        env['ROUTEPACK_COMPONENT_PROFILE'] = '1'
    argv = json.loads((OLD / 'argv.json').read_text())
    def option(key, value):
        argv[argv.index(key) + 1] = str(value)
    option('--global-batch-size', a.global_batch_size)
    option('--rollout-batch-size', a.rollout_batch_size)
    option('--n-samples-per-prompt', a.n_samples_per_prompt)
    option('--max-tokens-per-gpu', a.max_tokens_per_gpu)
    option('--rollout-max-response-len', a.rollout_max_response_len)
    option('--rollout-max-context-len', a.rollout_max_context_len)
    if a.balance_data and '--balance-data' not in argv:
        argv.append('--balance-data')
    option('--load-debug-rollout-data', a.rollout.resolve())
    if a.schedule_manifest:
        if not a.schedule_manifest.is_file():
            raise FileNotFoundError(a.schedule_manifest)
        env['ROUTEPACK_SCHEDULE_FILE'] = str(a.schedule_manifest.resolve())
    if a.execution_mode == 'timing':
        option('--custom-megatron-init-path', 'routepack.gpu_schedule_benchmark.install')
        env.update(ROUTEPACK_BENCHMARK_DIR=str(profile),
                   ROUTEPACK_BENCHMARK_WARMUP=str(a.warmups),
                   ROUTEPACK_BENCHMARK_REPEATS=str(a.repeats))
    a.output.mkdir(parents=True, exist_ok=False)
    cache = Path(tempfile.mkdtemp(prefix='ap-b1-cache-', dir='/tmp'))
    env.update(CUDA_LAUNCH_BLOCKING='0', ROUTEPACK_PROFILE_DIR=str(profile),
        ALIGN_A_RUN_DIR=str(a.output.resolve()), TMPDIR=str(cache.resolve()), TMP=str(cache.resolve()), TEMP=str(cache.resolve()))
    for key in ('TVM_FFI_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'XDG_CACHE_HOME'):
        d = cache / key.lower()
        d.mkdir()
        env[key] = str(d.resolve())
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env[key] = ''
    env.update(NO_PROXY='*', no_proxy='*')
    (a.output / 'argv.json').write_text(json.dumps(argv, indent=2))
    (a.output / 'runtime_env.json').write_text(json.dumps({'env_vars': env}, indent=2))
    if a.dry_run:
        return
    import psutil
    if psutil.virtual_memory().available < a.min_memory_gib * 2**30 or shutil.disk_usage(a.output).free < 2 * 2**30 or shutil.disk_usage(cache).free < 20 * 2**30:
        raise RuntimeError('Insufficient memory or disk headroom')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ.update(env)
    import torch
    samples = torch.load(a.rollout, map_location='cpu', weights_only=True)['samples']
    if len(samples) != a.expected_samples:
        raise RuntimeError(f'Expected {a.expected_samples} samples')
    if any(len(s['tokens']) > a.rollout_max_context_len or
           s['response_length'] > a.rollout_max_response_len for s in samples):
        raise ValueError('Frozen samples exceed declared length caps; increase explicit caps, do not truncate')
    del samples
    for path in reversed(env['PYTHONPATH'].split(':')):
        sys.path.insert(0, path)
    import ray
    # address=local never attaches to unrelated existing cluster.
    ray.init(address='local', num_gpus=8, num_cpus=min(64, len(os.sched_getaffinity(0))),
        object_store_memory=4 * 2**30, include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(prefix='ap-b1-', dir='/tmp'), runtime_env={'env_vars': env})
    try:
        sys.argv = argv[1:]
        runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
