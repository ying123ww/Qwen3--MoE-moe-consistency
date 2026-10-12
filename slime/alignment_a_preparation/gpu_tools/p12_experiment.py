"""P12 axes under the same C full graph configuration."""
import os
from pathlib import Path
import full_graph_ab_experiment as base
_BASE_ENVIRONMENT=base.environment
_BASE_CONFIGURE=base.configure

VARIANTS={
 'reference':dict(dense=False,rms=True,qk=False,router=False),
 'dense':dict(dense=True,rms=True,qk=False,router=False),
 'rms-off':dict(dense=False,rms=False,qk=False,router=False),
 'dense-rms-off':dict(dense=True,rms=False,qk=False,router=False),
 'dense-qk':dict(dense=True,rms=True,qk=True,router=False),
 'dense-router':dict(dense=True,rms=True,qk=False,router=True),
 'dense-qk-router':dict(dense=True,rms=True,qk=True,router=True),
}


def environment(variant):
    _BASE_ENVIRONMENT('C');axis=VARIANTS[variant]
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_MODE']='candidate' if axis['rms'] else 'off'
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH']='1' if axis['rms'] else '0'


def configure(config,variant,evidence):
    axis=VARIANTS[variant];result=_BASE_CONFIGURE(config,'C',evidence)
    if axis['dense']:
        result['forward_hooks'].append(dict(name='p12-dense-tile',target_modules=['model'],before_cuda_graph_capture=True,hook_factory='dense_tile_hooks:install',config=dict(output_dir=str(evidence))))
    if axis['qk']:
        result['forward_hooks'].append(dict(name='p12-qk',target_modules=['model'],before_cuda_graph_capture=True,hook_factory='qk_runtime_hooks:install',config=dict(mode='joint',audit_dir=None,allow_cuda_graph=True)))
    if axis['router']:
        result['forward_hooks'].append(dict(name='p12-router-tile',target_modules=['model'],before_cuda_graph_capture=True,hook_factory='router_full_tile_hooks:install',config=dict(output_dir=str(evidence))))
    result['forward_hooks'].append(dict(name='p12-fusion-proof',target_modules=['model'],before_cuda_graph_capture=True,hook_factory='full_fusion_proof:install',config=dict(variant=variant,axis=axis,output_dir=str(evidence))))
    return result
