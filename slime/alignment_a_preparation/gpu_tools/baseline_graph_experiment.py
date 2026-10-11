"""Non-deterministic baseline Breakable variants; baseline norms stay native."""
import os

MODES=('original','experts','fixed','fixed-experts')


def configure(config,mode,evidence,shadow=False):
    if mode not in MODES:raise ValueError(mode)
    result=dict(config)
    result.pop('deepep_config',None)
    result.update(enable_deterministic_inference=False,disable_cuda_graph=False,disable_prefill_cuda_graph=True,
        cuda_graph_config=dict(decode=dict(backend='breakable',bs=[2,4],max_bs=4),prefill=dict(backend='disabled')))
    hooks=[]
    if mode in ('fixed','fixed-experts'):
        hooks.append(dict(name='baseline-fixed-dispatch',target_modules=['model'],before_cuda_graph_capture=True,
            hook_factory='fixed_capacity_dispatch_hooks:install',config=dict(backend='breakable',capacity=16,
                shadow=shadow,output_dir=str(evidence))))
    if mode in ('experts','fixed-experts'):
        settings=dict(max_rows=16,shadow=shadow,output_dir=str(evidence))
        if mode=='fixed-experts':settings['cache_rows']=[16]
        hooks.append(dict(name='baseline-expert-graph',target_modules=['model'],before_cuda_graph_capture=True,
            hook_factory='expert_graph_hooks:install',config=settings))
    hooks.append(dict(name='breakable-proof',target_modules=['model'],hook_factory='breakable_experiment:install',
                      config=dict(output_dir=str(evidence))))
    result['forward_hooks']=hooks
    return result


def environment():
    os.environ.update(SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE='0',SGLANG_MEMORY_SAVER_CUDA_GRAPH='0',
        SLIME_QWEN3_RESIDUAL_RMS_MODE='off',SLIME_QWEN3_RESIDUAL_RMS_AUDIT='0',SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH='0',
        SLIME_QWEN3_ROUTER_TILE_MODE='off',SLIME_QWEN3_ROUTER_TILE_AUDIT='0')
