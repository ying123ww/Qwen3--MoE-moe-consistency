if __name__=='__main__':
    import os
    import sys
    from pathlib import Path
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_MODE']='candidate'
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_AUDIT']='0'
    os.environ['SLIME_QWEN3_ROUTER_TILE_MODE']='off'
    os.environ['SLIME_QWEN3_ROUTER_TILE_AUDIT']='0'
    output=Path(sys.argv[sys.argv.index('--run')+1])
    import sglang
    original=sglang.Engine
    def engine(**kwargs):
        config=dict(kwargs)
        hooks=[dict(name='qk-reference',target_modules=['model'],hook_factory='qk_runtime_hooks:install',config=dict(mode='joint',audit_dir=None))]
        for layer in (0,47):
            for kind,suffix in (('qkv','qkv_proj'),('output','o_proj')):
                hooks.append(dict(name=f'dense-{layer}-{kind}',target_modules=[f'model.layers.{layer}.self_attn.{suffix}'],
                    hook_factory='dense_projection_capture:install',config=dict(output_dir=str(output/'evidence'),kind=kind,layer=layer)))
        config['forward_hooks']=hooks
        return original(**config)
    sglang.Engine=engine
    import inference_benchmark
    inference_benchmark.main()
