"""128-sample continuation using the existing bounded sequence supervisor."""
import os
from pathlib import Path
import sys

import autopack_gpu_compare_sequence as sequence

# The 256-sample supervisor source/defaults remain unchanged.
sequence.BASE = sequence.ROOT / 'alignment_a_preparation/gpu_runs/autopack-gpu-compare128-20260914'
sequence.STAGES = [('gate', 'routepack'), ('gate', 'random'),
                   ('timing', 'random'), ('timing', 'routepack')]
original_command = sequence.command_for


def command_for(mode, method):
    run, profile, cmd = original_command(mode, method)
    profile = Path(str(profile).replace('/tmp/ap-exchange-', '/tmp/ap-exchange128-'))
    replacements = {
        '--rollout': '/tmp/ap-real-capture128-20260914/rollout.pt',
        '--global-batch-size': '128', '--rollout-batch-size': '16',
        '--expected-samples': '128', '--max-tokens-per-gpu': '10240',
        '--profile-directory': str(profile),
    }
    for key, value in replacements.items():
        cmd[cmd.index(key) + 1] = value
    return run, profile, cmd


if __name__ == '__main__':
    if '--resume-timing' in sys.argv:
        raise ValueError('This continuation runs its two gates first; do not use resume-timing')
    os.environ['ROUTEPACK_CHECKPOINT_CPU_OFFLOAD'] = '0'
    os.environ.pop('ROUTEPACK_CHUNK_GRADIENT_AGGREGATOR', None)
    sequence.command_for = command_for
    sequence.main()
