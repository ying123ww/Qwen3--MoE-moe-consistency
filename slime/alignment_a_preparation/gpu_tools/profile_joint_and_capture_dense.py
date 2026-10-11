"""Combined evidence Engine: Q/K profile and real QKV/O operands."""
import profiling_annotations
import qk_profile_annotations


def install(config):
    import qk_runtime_hooks
    qk_runtime_hooks.install(dict(mode='joint',audit_dir=None))
    return qk_profile_annotations.install(config)


profiling_annotations.install=install

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
        config=dict(kwargs);hooks=list(config.get('forward_hooks',[]))
        for layer in (0,47):
            for kind,suffix in (('qkv','qkv_proj'),('output','o_proj')):
                hooks.append(dict(name=f'dense-{layer}-{kind}',target_modules=[f'model.layers.{layer}.self_attn.{suffix}'],
                    hook_factory='dense_projection_capture:install',config=dict(output_dir=str(output/'dense-evidence'),kind=kind,layer=layer)))
        config['forward_hooks']=hooks
        return original(**config)
    sglang.Engine=engine
    import inference_benchmark
    import profile_export_guard
    profile_export_guard.install(inference_benchmark,output)
    inference_benchmark.main()
