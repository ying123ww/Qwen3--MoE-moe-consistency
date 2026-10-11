"""Reuse the finite model gate with explicit fixed-capacity graph settings."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['eager','breakable','full'],required=True)
    parser.add_argument('--shadow',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.shadow and args.backend=='full':raise ValueError('Shadow must run outside full capture/replay')
    os.environ.update(SLIME_QWEN3_RESIDUAL_RMS_MODE='candidate',SLIME_QWEN3_RESIDUAL_RMS_AUDIT='0',
        SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH='1',SGLANG_MEMORY_SAVER_CUDA_GRAPH='0')
    import fixed_capacity_experiment as experiment
    if args.backend=='full':experiment.allow_full_configuration(args.output)
    import breakable_model_gate as gate
    gate.configure=lambda config,mode,evidence:experiment.configure(config,args.backend,evidence,shadow=args.shadow)
    gate.capture(SimpleNamespace(mode='breakable-rms',output=args.output))


if __name__=='__main__':main()
