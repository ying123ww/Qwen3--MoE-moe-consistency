"""Explicit graph-only experiment adapter for the validated FP32 router tile."""
from collections import Counter

CAPTURED=Counter()


def install(config):
    import torch
    from sglang.srt.models import qwen3_moe as model,qwen3_router_tile as tile
    original=model.router_matmul
    def router(a,b,*,layer_id):
        if (a.is_cuda and a.device==b.device and a.dtype==b.dtype==torch.float32 and tile.eligible_shape(a.shape,b.shape)):
            result=tile._candidate(a,b)
            if torch.cuda.is_current_stream_capturing():CAPTURED[layer_id]+=1
            return result
        return original(a,b,layer_id=layer_id)
    model.router_matmul=router
    return lambda module,inputs,output:None
