"""Raw original GEMM snapshots before downstream mutation, P11 only."""
import json
from pathlib import Path

KEY='_slime_raw_projection_dump'


def install(config):
    import torch
    import torch.distributed as dist
    from sglang.srt.batch_invariant_ops import batch_invariant_ops as bio
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import SetInternalStateReqOutput
    from full_graph_profile_hooks import _CAPTURE_SIZE,_LAYER
    rank=dist.get_rank();refs={};last_size=None;directory=Path(config['output_dir'])
    original=bio._matmul_persistent_triton
    def matmul(a,b,bias=None):
        layer=_LAYER.get();size=_CAPTURE_SIZE.get();kind=None
        if a.ndim==2 and b.ndim==2 and a.shape[0] in (1,2,4) and a.shape[1]==2048:
            if a.dtype==torch.bfloat16 and b.shape[1]==2560:kind='qkv'
            elif a.dtype==torch.bfloat16 and b.shape[1]==2048:kind='output_projection'
            elif a.dtype==torch.float32 and b.shape[1]==128:kind='router'
        snapshot=(kind is not None and rank in (0,4) and layer in (0,47)
                  and size in (2,4) and torch.cuda.is_current_stream_capturing())
        if snapshot:
            if bias is not None:raise RuntimeError('Unexpected bias in selected raw GEMM')
            x=a.detach().clone()
        y=original(a,b,bias)
        if snapshot:
            refs[(size,kind,layer)]=(x,b,y.detach().clone(),dict(input_stride=list(a.stride()),weight_transpose_stride=list(b.stride()),output_stride=list(y.stride())))
        return y
    bio._matmul_persistent_triton=matmul
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    original_replay=FullCudaGraphBackend.replay
    def replay(self,shape_key,*args,**kwargs):
        nonlocal last_size
        result=original_replay(self,shape_key,*args,**kwargs)
        last_size=shape_key.size
        return result
    FullCudaGraphBackend.replay=replay
    original_control=Scheduler.set_internal_state
    def control(self,request):
        if set(request.server_args)=={KEY}:
            if request.server_args[KEY]!=1:raise ValueError('Invalid raw dump command')
            torch.cuda.synchronize();dist.barrier(group=get_tp_group().cpu_group)
            if last_size not in (2,4):raise RuntimeError('No selected raw replay to dump')
            selected={key:value for key,value in refs.items() if key[0]==last_size}
            expected={(s,k,l) for s in (last_size,) for k in ('qkv','output_projection','router') for l in (0,47)} if rank in (0,4) else set()
            if set(selected)!=expected:raise RuntimeError(f'Incomplete raw original GEMM snapshots rank={rank}: {set(selected)}')
            directory.mkdir(parents=True,exist_ok=True)
            for (size,kind,layer),(x,b,y,metadata) in selected.items():
                torch.save(dict(input=x.cpu(),weight=b.t().cpu(),output=y.cpu(),rank=rank,layer=layer,size=size,component=kind,mode='C',metadata=metadata,scope='Captured clones of original persistent GEMM operands/output, before downstream inplace operations'),directory/f'raw-size-{size}-{kind}-layer-{layer}-rank-{rank}.pt')
            (directory/f'complete-size-{last_size}-rank-{rank}.json').write_text(json.dumps(dict(rank=rank,capture_size=last_size,packets=len(selected),complete=True))+'\n')
            return SetInternalStateReqOutput(updated=True)
        return original_control(self,request)
    Scheduler.set_internal_state=control
    return lambda model,inputs,output:None
