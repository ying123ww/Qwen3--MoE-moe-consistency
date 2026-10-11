"""First-warmup-only worker fingerprint; no tensor reads or numerical changes."""
import json
import os
from pathlib import Path
import platform

import torch

ENV_KEYS = (
    'OMP_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES','OMP_WAIT_POLICY',
    'MKL_NUM_THREADS','MKL_DYNAMIC','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
    'KMP_AFFINITY','KMP_BLOCKTIME','GOMP_CPU_AFFINITY','MALLOC_ARENA_MAX',
    'PYTHONHASHSEED','PYTORCH_CUDA_ALLOC_CONF','CUDA_VISIBLE_DEVICES',
    'CUDA_DEVICE_MAX_CONNECTIONS','CUDA_MODULE_LOADING','LD_PRELOAD',
    'SLURM_JOB_ID','SLURM_STEP_ID','SLURM_PROCID','SLURM_LOCALID',
    'SLURM_CPUS_PER_TASK','SLURM_NTASKS','SLURM_CPU_BIND','SLURM_CPU_BIND_LIST',
    'SLURM_CPU_BIND_TYPE','SLURM_MEM_BIND','SLURM_MEM_BIND_LIST',
    'PMI_RANK','PMI_SIZE','PMI_FD',
)


def capture():
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.batch_invariant_ops import fused_rmsnorm_candidate as rms
    from sglang.srt.layers import layernorm
    return dict(pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),
                cgroup=Path('/proc/self/cgroup').read_text(),
                torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
                environment={key:os.environ[key] for key in ENV_KEYS if key in os.environ},
                batch_invariant_enabled=is_batch_invariant_mode_enabled(),
                residual_rms_mode=rms._MODE,residual_rms_enabled=layernorm._residual_rms_enabled,
                residual_rms_audit=rms._AUDIT_ENABLED,python=platform.python_version(),
                torch=str(torch.__version__),
                cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                               if line.startswith('model name')),platform.processor()),
                uname=platform.uname()._asdict())


def install(config):
    directory=Path(config['output_dir'])
    done=False
    def hook(module,inputs,output):
        nonlocal done
        if not done:
            value=capture()
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            directory.mkdir(parents=True,exist_ok=True)
            (directory/f'rank-{rank}-pid-{os.getpid()}.json').write_text(json.dumps(value,indent=2)+'\n')
            done=True
        return None
    return hook
