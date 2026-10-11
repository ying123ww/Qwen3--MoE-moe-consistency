"""P10 full decode A/B/C; communicate configuration is an explicit experimental axis."""
import json
import os

import fixed_capacity_experiment as fixed

MODES=('A','B','C')


def fixed_config(value=None):
    value=value or os.environ['SLIME_QWEN3_FIXED_DEEPEP_CONFIG']
    parsed=json.loads(value)
    if set(parsed)!= {'normal_dispatch','normal_combine'}:
        raise ValueError('P10 requires explicit normal dispatch/combine configs')
    if any(v['num_sms']!=20 for v in parsed.values()):
        raise ValueError('P10 fixed configuration requires 20 SMs')
    return json.dumps(parsed,sort_keys=True,separators=(',',':'))


def environment(mode):
    if mode not in MODES:raise ValueError(mode)
    aligned=mode=='C'
    os.environ.update(SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE='1' if aligned else '0',
        SGLANG_MEMORY_SAVER_CUDA_GRAPH='0',SLIME_QWEN3_RESIDUAL_RMS_MODE='candidate' if aligned else 'off',
        SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH='1' if aligned else '0',SLIME_QWEN3_RESIDUAL_RMS_AUDIT='0',
        SLIME_QWEN3_ROUTER_TILE_MODE='off',SLIME_QWEN3_ROUTER_TILE_AUDIT='0')


def configure(config,mode,evidence,backend='full',shadow=False):
    if mode not in MODES:raise ValueError(mode)
    result=dict(config)
    result['enable_deterministic_inference']=mode=='C'
    if mode=='A':result.pop('deepep_config',None)
    else:result['deepep_config']=fixed_config(result.get('deepep_config'))
    result=fixed.configure(result,backend,evidence,shadow)
    result['forward_hooks'].append(dict(name='full-runtime-audit',target_modules=['model'],
        before_cuda_graph_capture=True,hook_factory='full_graph_ab_metrics:install',
        config=dict(mode=mode,output_dir=str(evidence))))
    return result


def allow_full_configuration(output,mode):
    fixed.allow_full_configuration(output,deterministic=mode=='C')


def base_config():
    return dict(model_path='/storage/data/yingxy2024/Models/Qwen3-30B-A3B',trust_remote_code=True,
        random_seed=1234,dtype='bfloat16',tp_size=8,dp_size=2,attn_cp_size=2,ep_size=4,moe_dp_size=2,
        enable_dp_attention=True,enable_dp_lm_head=True,moe_dense_tp_size=1,moe_a2a_backend='deepep',
        deepep_mode='normal',moe_runner_backend='triton',attention_backend='triton',kv_cache_dtype='bfloat16',
        page_size=64,mem_fraction_static=0.65,context_length=1024,max_prefill_tokens=512,
        disable_prefill_cuda_graph=True,enable_memory_saver=True,skip_server_warmup=True,
        log_level='info',port=15800,nccl_port=15801)
