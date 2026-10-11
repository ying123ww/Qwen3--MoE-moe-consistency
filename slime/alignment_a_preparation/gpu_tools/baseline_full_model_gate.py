"""Controlled-prefill baseline full graph gate; eager fixed mode audits padding math."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import baseline_full_experiment as experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['eager','full'],required=True)
    parser.add_argument('--shadow',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.shadow and args.backend=='full':raise ValueError('Audit outside native full capture/replay')
    experiment.environment()
    if args.backend=='full':experiment.allow_full_configuration(args.output)
    import breakable_model_gate as gate
    def configure(config,mode,evidence):
        result=experiment.configure(config,args.backend,evidence,args.shadow)
        result['prefill_max_requests']=1
        return result
    gate.configure=configure
    gate.capture(SimpleNamespace(mode='breakable',output=args.output))


if __name__=='__main__':main()
