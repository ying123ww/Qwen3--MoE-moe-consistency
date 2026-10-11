"""Matched benchmark with an explicitly configured experimental model hook."""
if __name__ == '__main__':
    import os
    import sys
    def value(name, default):
        return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_MODE']=value('--residual-rms-mode','off')
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_AUDIT']='0'
    os.environ['SLIME_QWEN3_ROUTER_TILE_MODE']=value('--router-tile-mode','off')
    os.environ['SLIME_QWEN3_ROUTER_TILE_AUDIT']='0'
    import sglang
    original=sglang.Engine
    def engine(**kwargs):
        config=dict(kwargs)
        mode=os.environ.get('SLIME_QK_EXPERIMENT_MODE','off')
        config['forward_hooks']=[dict(name='qk-experiment',target_modules=['model'],hook_factory='qk_runtime_hooks:install',
                                     config=dict(mode=mode,audit_dir=None))]
        return original(**config)
    sglang.Engine=engine
    import inference_benchmark
    inference_benchmark.main()
