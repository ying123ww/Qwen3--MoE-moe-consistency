"""Matched deterministic generate with fixed-capacity normal dispatch."""
if __name__=='__main__':
    import json
    import os
    from pathlib import Path
    import sys
    def value(name,default):return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default
    run=Path(value('--run',''));backend=os.environ['SLIME_FIXED_CAPACITY_BACKEND']
    os.environ.update(SLIME_QWEN3_RESIDUAL_RMS_MODE='candidate',SLIME_QWEN3_RESIDUAL_RMS_AUDIT='0',
        SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH='1',SLIME_QWEN3_ROUTER_TILE_MODE='off',SLIME_QWEN3_ROUTER_TILE_AUDIT='0')
    import sglang
    import fixed_capacity_experiment as experiment
    if backend=='full':experiment.allow_full_configuration(run)
    original=sglang.Engine
    def engine(**kwargs):
        config=experiment.configure(kwargs,backend,run/'graph-evidence')
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
