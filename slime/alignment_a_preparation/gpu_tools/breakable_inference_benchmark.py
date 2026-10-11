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
        return original(**config)
    sglang.Engine = engine
    import inference_benchmark
    from profile_export_guard import install
    install(inference_benchmark, run)
    inference_benchmark.main()
