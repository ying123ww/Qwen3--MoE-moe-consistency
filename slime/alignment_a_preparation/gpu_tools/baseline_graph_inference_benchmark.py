"""Matched baseline generate for native norms plus optional MoE graph changes."""
if __name__=='__main__':
    import json
    import os
    from pathlib import Path
    import sys
    import baseline_graph_experiment as experiment
    experiment.environment()
    def value(name,default):return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default
    mode=os.environ['SLIME_BASELINE_GRAPH_MODE'];run=Path(value('--run',''))
    import sglang
    original=sglang.Engine
    def engine(**kwargs):
        config=experiment.configure(kwargs,mode,run/'graph-evidence')
        (run/'experimental-server-args.json').write_text(json.dumps(config,indent=2)+'\n')
        result=original(**config)
        import torch
        from inference_benchmark import FROZEN,flush_cache_until_idle
        sources=torch.load(FROZEN,map_location='cpu',weights_only=False)['samples']
        prompts=[list(map(int,s['tokens'][:-int(s['response_length'])])) for s in sources]
        flush_cache_until_idle(result)
        result.generate(input_ids=prompts[:3],sampling_params=dict(temperature=0,max_new_tokens=2,ignore_eos=True))
        return result
    sglang.Engine=engine
    import inference_benchmark
    inference_benchmark.main()
