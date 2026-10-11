"""Experiment-only graph configuration and normal-mode full-backend bypass."""
import json
from pathlib import Path


def allow_full_configuration(output, *, deterministic=True):
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.model_executor.cuda_graph_config import Backend
    original=ServerArgs._handle_a2a_moe
    output=Path(output)
    def handle(self):
        requested=self.cuda_graph_config.decode.backend
        original(self)
        if requested==Backend.FULL:
            if (self.moe_a2a_backend!='deepep' or self.deepep_mode!='normal'
                    or bool(self.enable_deterministic_inference)!=deterministic):
                raise RuntimeError('Fixed-capacity full probe requires the requested DeepEP normal variant')
            self.cuda_graph_config.decode.backend=Backend.FULL
            output.mkdir(parents=True,exist_ok=True)
            (output/'full-config-bypass.json').write_text(json.dumps(dict(decode='full',prefill='disabled',
                deepep_mode='normal',deterministic=deterministic,experimental=True),indent=2)+'\n')
            print('FIXED_CAPACITY_FULL_BACKEND_RESTORED',flush=True)
    ServerArgs._handle_a2a_moe=handle


def configure(config,backend,evidence,shadow=False):
    if backend not in ('eager','breakable','full'):raise ValueError(backend)
    result=dict(config)
    result.update(disable_cuda_graph=backend=='eager',disable_prefill_cuda_graph=True,
        cuda_graph_config=dict(decode=dict(backend='disabled' if backend=='eager' else backend,bs=[2,4],max_bs=4),
                               prefill=dict(backend='disabled')))
    result['forward_hooks']=[dict(name='fixed-capacity-dispatch',target_modules=['model'],
        hook_factory='fixed_capacity_dispatch_hooks:install',before_cuda_graph_capture=True,
        config=dict(backend=backend,capacity=16,shadow=shadow,output_dir=str(evidence)))]
    if backend=='breakable':
        result['forward_hooks'].append(dict(name='breakable-proof',target_modules=['model'],
            hook_factory='breakable_experiment:install',config=dict(output_dir=str(evidence))))
    return result
