"""Run the dedicated capture with frozen environment and node-local CUDA layout."""
import argparse
import json
import os
import resource
import signal
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / 'alignment_a_preparation'
ENV = Path('/public/home/yingxy2024/.conda/envs/slime')
CUDA = Path('/tmp/ap-cuda-20260914')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--prompts-json', type=Path, default=PREP/'gpu_runs/autopack-hgx05-20260914-input/prompts.json')
    parser.add_argument('--real-training', action='store_true')
    parser.add_argument('--response-tokens', type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--context-length', type=int)
    parser.add_argument('--capture-batch-size', type=int)
    parser.add_argument('--timeout', type=int, default=2400)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    for d in ('bin', 'lib64/stubs'):
        (CUDA / d).mkdir(parents=True, exist_ok=True)
    links = {'bin/nvcc': ENV/'bin/nvcc', 'bin/cicc': ENV/'nvvm/bin/cicc',
        'include': ENV/'targets/x86_64-linux/include', 'nvvm': ENV/'nvvm',
        'lib64/libcudart.so': ENV/'targets/x86_64-linux/lib/libcudart.so',
        'lib64/stubs/libcuda.so': Path('/lib64/libcuda.so')}
    for name, target in links.items():
        if not target.exists():
            raise FileNotFoundError(target)
        dest = CUDA / name
        if not dest.exists():
            dest.symlink_to(target)
    env = json.loads((PREP/'gpu_runs/routepack-b1-20260912-02/runtime_env.json').read_text())['env_vars']
    env = {k:v.replace('/tmp/slime-cuda-alignment-a-20260909',str(CUDA)) for k,v in env.items()}
    tmp = Path('/tmp') / ('ap-cap-' + str(os.getpid()))
    tmp.mkdir()
    env.update(CUDA_LAUNCH_BLOCKING='0', TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp))
    for key in ('TVM_FFI_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','XDG_CACHE_HOME'):
        path = tmp/key.lower()
        path.mkdir()
        env[key] = str(path)
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        env[key]=''
    env.update(NO_PROXY='*',no_proxy='*')
    cmd = [str(ENV/'bin/python'),str(PREP/'gpu_tools/autopack_capture_hgx05.py'),
        '--prompts-json',str(args.prompts_json.resolve()),
        '--reference-manifest',str(PREP/'gpu_runs/router-exact-off-20260912-02/capture/manifest.json'),
        '--output',str(run/'capture')]
    if args.real_training:
        cmd.append('--real-training')
    for name in ('response_tokens', 'temperature', 'context_length', 'capture_batch_size'):
        value = getattr(args, name)
        if value is not None:
            cmd.extend(['--' + name.replace('_', '-'), str(value)])
    (run/'launch-manifest.json').write_text(json.dumps({
        'arguments': vars(args), 'timeout_seconds': args.timeout, 'temporary_cache': str(tmp),
        'scope': 'capture and frozen rollout preparation; no GPU timing claim'}, indent=2, default=str))
    (run/'runtime_env.json').write_text(json.dumps({'env_vars':env},indent=2))
    (run/'argv.json').write_text(json.dumps(cmd,indent=2))
    with (run/'capture.log').open('w') as log:
        process = subprocess.Popen(cmd, env=dict(os.environ, **env), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        (run/'process.json').write_text(json.dumps({'pid': process.pid, 'pgid': process.pid}))
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            # The leader may exit before its workers; terminate the same owned group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            (run/'exit.json').write_text(json.dumps({'returncode': 124, 'timeout_seconds': args.timeout}))
            raise SystemExit(124)
    (run/'exit.json').write_text(json.dumps({'returncode': returncode}))
    raise SystemExit(returncode)

if __name__=='__main__':
    main()
