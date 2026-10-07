"""One guarded shape or route capture; no optimizer or online IPC sync."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import socket
import subprocess
import time

from launch import ROOT, PREP, ENV, check_memory_guard
from resource_preflight import snapshot, validate


def safe_run_id(value):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,95}', value):
        raise argparse.ArgumentTypeError('run ID must be 1..96 lowercase letters, digits, underscores or hyphens, starting with a letter/digit')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-routes', action='store_true', help='Capture B2 route preflight data instead of matmul profiling')
    parser.add_argument('--run-id', type=safe_run_id, help='Fresh experiment directory name; existing directories are rejected')
    args = parser.parse_args()
    assert socket.gethostname().split('.')[0] == 'ai_hgx_03'
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lock = open('/tmp/alignment-a-launch.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_name = 'routepack-capture-20260912-01' if args.capture_routes else 'a-matmul-attribution-20260912-01'
    if args.run_id is not None:
        run_name = args.run_id
    run = PREP/'gpu_runs'/run_name
    run.mkdir(exist_ok=False)
    tmp = ROOT.parent/'.a-tmp'/run.name
    tmp.mkdir(exist_ok=False)
    resources = snapshot(str(tmp))
    validate(resources, 512, 20)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPUs are busy')
    jobs = [Path(g['path']) for g in resources['cgroups'] if Path(g['path']).name.startswith('job_') and 'memory' in g['path']]
    if len(jobs) != 1:
        raise RuntimeError(f'Expected one allocation memory cgroup: {jobs}')
    (run/'resource-preflight.json').write_text(json.dumps(resources, indent=2)+'\n')
    env = dict(os.environ)
    env.update(json.loads((PREP/'gpu_runs/a-restore-20260909-06/runtime_env.json').read_text())['env_vars'])
    env.update({k:str(tmp) for k in ('TMPDIR','TMP','TEMP')})
    cache = Path('/dev/shm')/run.name
    cache.mkdir(exist_ok=False)
    for key, sub in {'TVM_FFI_CACHE_DIR':'tvm','TRITON_CACHE_DIR':'triton','TORCHINDUCTOR_CACHE_DIR':'inductor','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg'}.items():
        dest = cache/sub
        dest.mkdir()
        env[key] = str(dest)
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        env[key] = ''
    env['NO_PROXY'] = env['no_proxy'] = '*'
    env['SGLANG_PROFILE_WITH_STACK'] = '0'
    for key in list(env):
        if key.startswith('MEGATRON_USE_SGLANG_'):
            env.pop(key)
    for path in (tmp, cache, run):
        probe = path/'write-probe'
        probe.write_text('ok')
        probe.unlink()
    guard_state = run/'memory-guard.json'
    os.environ['ALIGN_A_MEMORY_GUARD'] = str(guard_state)
    guard = subprocess.Popen([str(ENV/'bin/python'), str(PREP/'gpu_tools/memory_guard.py'), '--cgroup', str(jobs[0]), '--state', str(guard_state)])
    try:
        for _ in range(50):
            if guard_state.exists():
                break
            time.sleep(.1)
        check_memory_guard(resources)
        if args.capture_routes:
            reference = PREP/'gpu_runs/a-matmul-attribution-20260912-01/manifest.json'
            argv = [str(ENV/'bin/python'), str(ROOT/'tools/routepack/capture_engine.py'),
                    '--reference-manifest', str(reference),
                    '--frozen-rollout', '/public/home/yingxy2024/.codex/qwen3-v89-regression/baseline/rollout/1.pt',
                    '--output', str(run/'capture'), '--response-tokens', '16']
            (run/'launch-manifest.json').write_text(json.dumps({
                'mode': 'routepack_b2_preflight', 'reference_manifest': str(reference),
                'output': str(run/'capture'), 'response_tokens': 16,
                'timing_use': 'capture only; not a benchmark or B1 training replay gate',
                'guard': 'same allocation memory guard, GPU idle check and isolated caches as matmul mode',
            }, indent=2)+'\n')
        else:
            argv = [str(ENV/'bin/python'),str(PREP/'gpu_tools/inference_benchmark.py'),'--run',str(run),'--variant','aligned','--warmup','1','--steps','1','--profile','--profile-shapes','--response-tokens','16']
        (run/'argv.json').write_text(json.dumps(argv)+'\n')
        (run/'runtime_env.json').write_text(json.dumps({'env_vars':{k:v for k,v in env.items() if k in json.loads((PREP/'gpu_runs/a-restore-20260909-06/runtime_env.json').read_text())['env_vars'] or k.endswith('CACHE_DIR')}}, indent=2)+'\n')
        sources = [Path(__file__), PREP/'gpu_tools/inference_benchmark.py', ROOT.parent/'experiments/sglang-main-sm80/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py']
        if args.capture_routes:
            sources.extend([ROOT/'tools/routepack/capture_engine.py', ROOT/'tools/routepack/capture_routes.py', reference])
            sglang_srt = ROOT.parent/'experiments/sglang-main-sm80/python/sglang/srt'
            sources.extend(sglang_srt / relative for relative in (
                'models/qwen3_moe.py', 'layers/moe/topk.py',
                'state_capturer/routed_experts.py', 'state_capturer/base.py',
            ))
        (run/'source-hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2)+'\n')
        with (run/'job.log').open('w') as log:
            result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        (run/'exit.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
        if result.returncode:
            raise RuntimeError(f'Inference failed: {result.returncode}')
        shutil.rmtree(cache)
        shutil.rmtree(tmp)
    finally:
        guard.terminate()
        guard.wait(timeout=10)
    print('ROUTEPACK_CAPTURE_COMPLETE' if args.capture_routes else 'MATMUL_CAPTURE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
