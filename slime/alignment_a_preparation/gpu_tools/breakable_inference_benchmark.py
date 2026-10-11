"""Matched generate benchmark with an explicit Plan 2 graph backend."""
if __name__ == '__main__':
    import json
    import os
    from pathlib import Path
    import sys
    mode = os.environ['SLIME_BREAKABLE_EXPERIMENT_MODE']
    def value(name, default):
        return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_MODE'] = value('--residual-rms-mode', 'off')
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_AUDIT'] = '0'
    os.environ['SLIME_QWEN3_ROUTER_TILE_MODE'] = 'off'
    os.environ['SLIME_QWEN3_ROUTER_TILE_AUDIT'] = '0'
    import sglang
    from breakable_experiment import configure
    original = sglang.Engine
    run = Path(value('--run', ''))
    def engine(**kwargs):
        config = configure(kwargs, mode, run/'graph-evidence')
        (run/'experimental-server-args.json').write_text(json.dumps(config, indent=2)+'\n')
        result = original(**config)
        if os.environ.get('SLIME_GRAPH_FUSION_PREWARM') == '1':
            import torch
            from inference_benchmark import FROZEN, flush_cache_until_idle
            samples = torch.load(FROZEN, map_location='cpu', weights_only=False)['samples']
            prompts = [list(map(int, s['tokens'][:-int(s['response_length'])])) for s in samples]
            flush_cache_until_idle(result)
            result.generate(input_ids=prompts[:3], sampling_params=dict(temperature=0, max_new_tokens=2, ignore_eos=True))
            (run/'shape2-prewarm.json').write_text(json.dumps(dict(prompts=3, response_tokens=2,
                timed=False, purpose='Pay first replay evidence writes before measurement'))+'\n')
        return result
    sglang.Engine = engine
    import inference_benchmark
    from profile_export_guard import install
    install(inference_benchmark, run)
    inference_benchmark.main()
