"""Capture real small-M projection operands without changing model results."""
import json
from pathlib import Path
import torch


def install(config):
    directory=Path(config['output_dir'])
    kind,layer=config['kind'],config['layer']
    saved=False
    def hook(module,inputs,output):
        nonlocal saved
        rank=torch.distributed.get_rank()
        x=inputs[0]
        if saved or rank not in (0,4) or x.shape[0]!=4 or torch.autograd._profiler_enabled():
            return None
        y=output[0] if isinstance(output,tuple) else output
        bias=getattr(module,'bias',None)
        if bias is not None:
            raise RuntimeError('Projection probe expects the configured bias-free Qwen3 case')
        directory.mkdir(parents=True,exist_ok=True)
        metadata=dict(rank=rank,layer=layer,kind=kind,input_shape=list(x.shape),input_stride=list(x.stride()),
                      weight_shape=list(module.weight.shape),weight_stride=list(module.weight.stride()),
                      output_shape=list(y.shape),dtype=str(x.dtype),module_type=type(module).__name__)
        torch.save(dict(input=x.detach().cpu(),weight=module.weight.detach().cpu(),output=y.detach().cpu(),metadata=metadata),
                   directory/f'rank-{rank}-layer-{layer}-{kind}.pt')
        print('DENSE_REAL_TENSOR_SAVED '+json.dumps(metadata),flush=True)
        saved=True
        return None
    return hook
