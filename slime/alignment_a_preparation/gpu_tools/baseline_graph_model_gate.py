"""Compare same workload baseline outputs; batch invariance is not required."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import baseline_graph_experiment as experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    cap=sub.add_parser('capture');cap.add_argument('--mode',choices=experiment.MODES,required=True)
    cap.add_argument('--output',type=Path,required=True);cap.add_argument('--shadow',action='store_true')
    compare=sub.add_parser('compare')
    for name in ('reference','candidate','output'):compare.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.action=='capture':
        experiment.environment()
        import breakable_model_gate as gate
        def configure(config,mode,evidence):
            result=experiment.configure(config,args.mode,evidence,args.shadow)
            # The baseline is batch-sensitive: fix per-DP prefill grouping for
            # numerical comparisons only. Timed generation retains defaults.
            result['prefill_max_requests']=1
            return result
        gate.configure=configure
        gate.capture(SimpleNamespace(mode='breakable',output=args.output))
    else:
        import residual_rms_performance as guard
        from router_tile_exact import load_case,compare_samples
        failures=[]
        for name in ('batch8','batch3','batch1','variable3'):
            failures.extend(compare_samples(load_case(args.reference,name),load_case(args.candidate,name),name))
        result=dict(model_exact=not failures,failures=failures,
            scope='Same-case greedy token/route/logprob-bit comparison; no baseline batch-invariance requirement')
        guard.save(args.output,result)
        print(result,flush=True)
        if failures:raise RuntimeError('Baseline same-workload exact gate failed')


if __name__=='__main__':main()
