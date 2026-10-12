"""Full generate protocol, P12 axes; observation work outside timing."""
import sys
import p12_experiment as experiment
import full_graph_ab_benchmark as benchmark


def main():
    if '--variant' not in sys.argv:raise ValueError('Specify a validated variant')
    index=sys.argv.index('--variant');variant=sys.argv[index+1];del sys.argv[index:index+2]
    if variant not in experiment.VARIANTS:raise ValueError(variant)
    def environment(mode):
        if mode!='C':raise ValueError('P12 timing is C alignment')
        experiment.environment(variant)
    def configure(config,mode,evidence,*args,**kwargs):
        if mode!='C' or args or kwargs:raise ValueError('Unexpected timing configuration')
        return experiment.configure(config,variant,evidence)
    benchmark.experiment.environment=environment
    benchmark.experiment.configure=configure
    benchmark.main()


if __name__=='__main__':main()
