"""P10 full/eager gate with controlled prefill and explicit communication axis."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import full_graph_ab_experiment as experiment
from full_graph_ab_control import control


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=experiment.MODES,required=True)
    parser.add_argument('--backend',choices=('eager','full'),required=True)
    parser.add_argument('--shadow',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.shadow and args.backend=='full':raise ValueError('Math shadow is outside capture/replay')
    experiment.environment(args.mode)
    if args.backend=='full':experiment.allow_full_configuration(args.output,args.mode)
    import breakable_model_gate as gate
    def configure(config,mode,evidence):
        result=experiment.configure(config,args.mode,evidence,args.backend,args.shadow)
        result['prefill_max_requests']=1
        return result
    gate.configure=configure
    import sglang
    original=sglang.Engine
    def engine(**kwargs):
        result=original(**kwargs)
        control(result,'reset','gate')
        original_shutdown=result.shutdown
        def shutdown():
            try:control(result,'flush','gate')
            finally:original_shutdown()
        result.shutdown=shutdown
        return result
    sglang.Engine=engine
    gate.capture(SimpleNamespace(mode='breakable-rms' if args.mode=='C' else 'breakable',output=args.output))


if __name__=='__main__':main()
