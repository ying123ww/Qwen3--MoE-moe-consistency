"""Greedy tokens/routes/logprob bits, batch gate and native full capture proofs."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import p12_experiment as experiment


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=experiment.VARIANTS,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    experiment.environment(args.variant);experiment.base.allow_full_configuration(args.output,'C')
    import breakable_model_gate as gate
    def configure(config,mode,evidence):
        result=experiment.configure(config,args.variant,evidence);result['prefill_max_requests']=1
        return result
    gate.configure=configure
    import sglang
    from full_graph_ab_control import control
    original=sglang.Engine
    def engine(**kwargs):
        result=original(**kwargs);control(result,'reset','gate');shutdown=result.shutdown
        def stop():
            try:control(result,'flush','gate')
            finally:shutdown()
        result.shutdown=stop
        return result
    sglang.Engine=engine
    gate.capture(SimpleNamespace(mode='breakable-rms' if experiment.VARIANTS[args.variant]['rms'] else 'breakable',output=args.output))


if __name__=='__main__':main()
