"""Explicit BF16 Qwen3 QKV/O small tile adapter, no K reduction changes."""
from collections import Counter,defaultdict
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path
import re

_SIZE=ContextVar('p12_size',default=None)
_LAYER=ContextVar('p12_layer',default=-1)
_KIND=ContextVar('p12_kind',default=None)


def eligible_shape(a,b):
    return len(a)==len(b)==2 and a[0] in (1,2,4) and a[1]==b[0]==2048 and b[1] in (2048,2560)


def install(config):
    import torch
    import triton
    import torch.distributed as dist
    from sglang.srt.batch_invariant_ops import batch_invariant_ops as bio
    from sglang.srt.layers.linear import QKVParallelLinear,RowParallelLinear
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    rank=dist.get_rank();directory=Path(config['output_dir']);directory.mkdir(parents=True,exist_ok=True)
    counts=Counter();layers=defaultdict(lambda:defaultdict(set));shapes=defaultdict(Counter)
    original=bio._matmul_persistent_triton
    def matmul(a,b,bias=None):
        kind=_KIND.get()
        accepted=(kind in ('qkv','output_projection') and a.is_cuda and a.device==b.device
                  and a.dtype==b.dtype==torch.bfloat16 and bias is None and eligible_shape(a.shape,b.shape))
        if not accepted:return original(a,b,bias)
        m,k=a.shape;n=b.shape[1];c=torch.empty((m,n),device=a.device,dtype=a.dtype)
        sms=torch.cuda.get_device_properties(a.device).multi_processor_count
        grid=(min(sms,triton.cdiv(m,16)*triton.cdiv(n,64)),)
        bio.matmul_kernel_persistent[grid](a,b,c,None,m,n,k,*a.stride(),*b.stride(),*c.stride(),
            BLOCK_SIZE_M=16,BLOCK_SIZE_N=64,BLOCK_SIZE_K=64,GROUP_SIZE_M=8,NUM_SMS=sms,
            A_LARGE=a.numel()>2**31,B_LARGE=b.numel()>2**31,C_LARGE=c.numel()>2**31,HAS_BIAS=False,num_stages=3,num_warps=4)
        counts[kind]+=1
        if torch.cuda.is_current_stream_capturing():
            size=_SIZE.get();layer=_LAYER.get()
            if size not in (2,4) or layer not in range(48):raise RuntimeError('Candidate source provenance missing')
            layers[size][kind].add(layer);shapes[size][(kind,m,k,n)]+=1
        return c
    bio._matmul_persistent_triton=matmul
    for owner,kind in ((QKVParallelLinear,'qkv'),(RowParallelLinear,'output_projection')):
        method=owner.forward
        def make(method,kind):
            @wraps(method)
            def call(module,*args,**kwargs):
                path=getattr(module,'_p12_path','');match=re.search(r'layers\.(\d+)',path)
                layer=int(match.group(1)) if match else -1
                token=_LAYER.set(layer);kind_token=_KIND.set(kind)
                try:return method(module,*args,**kwargs)
                finally:_LAYER.reset(token);_KIND.reset(kind_token)
            return call
        owner.forward=make(method,kind)
    original_capture=FullCudaGraphBackend.capture_one
    def capture(self,shape_key,*args,**kwargs):
        size=shape_key.size;token=_SIZE.set(size)
        try:result=original_capture(self,shape_key,*args,**kwargs)
        finally:_SIZE.reset(token)
        if any(layers[size][kind]!=set(range(48)) for kind in ('qkv','output_projection')):raise RuntimeError('Not all 48 QKV/O candidates were captured')
        report=dict(rank=rank,size=size,tile=[16,64,64],warps=4,layers={k:sorted(v) for k,v in layers[size].items()},
            shapes=[dict(component=k[0],M=k[1],K=k[2],N=k[3],calls=v) for k,v in shapes[size].items()],counts=dict(counts),scope='Explicit BF16 QKV/O shapes and source layers; original K tiles and tl.dot accumulation retained')
        (directory/f'dense-capture-size-{size}-rank-{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        return result
    FullCudaGraphBackend.capture_one=capture
    tagged=False
    def tag(model,inputs,output):
        nonlocal tagged
        if not tagged:
            for path,module in model.named_modules():module._p12_path=path
            tagged=True
    return tag
