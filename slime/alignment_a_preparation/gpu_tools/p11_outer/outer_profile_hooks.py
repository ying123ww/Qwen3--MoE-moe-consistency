"""Short actual prefill/sampling window, complementing selected decode graphs."""
from collections import Counter
import json
from pathlib import Path
import os

KEY='_slime_p11_outer_profile'

def install(config):
    import torch
    import torch.distributed as dist
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import SetInternalStateReqOutput
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    import full_graph_profile_hooks
    full_graph_profile_hooks.install(dict(config,nsys=False,steps=32))
    rank=dist.get_rank();armed=False;counts=Counter();shapes=[]
    original=EagerRunner.execute
    def eager(self,batch,*args,**kwargs):
        if not armed:return original(self,batch,*args,**kwargs)
        kind='prefill' if batch.forward_mode.is_extend() else ('idle' if batch.forward_mode.is_idle() else 'decode')
        counts[kind]+=1;shapes.append(dict(mode=kind,input_rows=int(batch.input_ids.numel()),batch_size=int(batch.batch_size)))
        torch.cuda.nvtx.range_push(f'P11/eager_{kind}/rank={rank}/shape={tuple(batch.input_ids.shape)}')
        try:return original(self,batch,*args,**kwargs)
        finally:torch.cuda.nvtx.range_pop()
    EagerRunner.execute=eager
    replay=FullCudaGraphBackend.replay
    def full(self,*args,**kwargs):
        if armed:counts['full_replays']+=1
        return replay(self,*args,**kwargs)
    FullCudaGraphBackend.replay=full
    original_control=Scheduler.set_internal_state
    def control(self,request):
        nonlocal armed
        if set(request.server_args)!={KEY}:return original_control(self,request)
        value=request.server_args[KEY]
        torch.cuda.synchronize();dist.barrier(group=get_tp_group().cpu_group)
        if value==1:
            counts.clear();shapes.clear();armed=True;torch.cuda.cudart().cudaProfilerStart()
        elif value==2:
            armed=False
            Path(config['output_dir'],f'outer-rank-{rank}.json').write_text(json.dumps(dict(rank=rank,pid=os.getpid(),counts=dict(counts),shapes=shapes,scope='Actual uncached eight-prompt, one-response-token prefill+sampling, incidental decode/idle/replay counts recorded separately'),indent=2)+'\n')
            dist.barrier(group=get_tp_group().cpu_group)
            torch.cuda.cudart().cudaProfilerStop()
        else:raise ValueError('Unknown outer profile command')
        return SetInternalStateReqOutput(updated=True)
    Scheduler.set_internal_state=control
    return lambda module,inputs,output:None
