"""Real-step gradient and sampled FP32 CPU-master delta evidence, P13 only."""
import functools
import json
import os
from pathlib import Path


def cpu_master_tensors(optimizer):
    import torch
    pending=[optimizer];seen_objects=set();seen_tensors=set();tensors=[]
    while pending:
        obj=pending.pop()
        if obj is None or id(obj) in seen_objects:continue
        seen_objects.add(id(obj))
        for group in getattr(obj,'param_groups',[]) or []:
            for tensor in group['params']:
                if tensor.device.type=='cpu' and tensor.dtype==torch.float32 and tensor.numel() and id(tensor) not in seen_tensors:
                    seen_tensors.add(id(tensor));tensors.append(tensor)
        chained=getattr(obj,'chained_optimizers',None)
        keys=('gpu_optimizer',) if chained is not None else ('optimizer','gpu_optimizer')
        for key in keys:
            value=getattr(obj,key,None)
            if value is not None:pending.append(value)
        for key in ('chained_optimizers','sub_optimizers','cpu_optimizers'):
            pending.extend(getattr(obj,key,[]) or [])
    return tensors


def before_train_step(args,rollout_id,step_id,model,optimizer,opt_param_scheduler):
    from slime.backends.megatron_utils.alignment.qwen3 import enable_qwen3_moe_alignment_before_train_step
    enable_qwen3_moe_alignment_before_train_step(args,rollout_id,step_id,model,optimizer,opt_param_scheduler)
    import torch
    import torch.distributed as dist
    from slime.backends.megatron_utils import model as training
    optimizer._p13_context=(rollout_id,step_id)
    if getattr(optimizer,'_p13_update_observed',False):return
    original=optimizer.step;rank=dist.get_rank();directory=Path(os.environ['SLIME_P13_RUN_DIR'])/'training-update-evidence';directory.mkdir(parents=True,exist_ok=True)
    optimizer._p13_context=(rollout_id,step_id)
    @functools.wraps(original)
    def measured(*positional,**kwargs):
        masters=cpu_master_tensors(optimizer)
        if not masters:raise RuntimeError('P13 could not find real FP32 CPU optimizer masters')
        before=[x.detach().reshape(-1)[:64].clone() for x in masters]
        result=original(*positional,**kwargs);torch.cuda.synchronize()
        if not result[0]:raise RuntimeError('Real optimizer step failed')
        gradient=result[1]
        if torch.is_tensor(gradient):gradient=float(gradient.item())
        else:gradient=float(gradient)
        changed=sum(int((x.detach().reshape(-1)[:64].view(torch.int32)!=b.view(torch.int32)).sum()) for x,b in zip(masters,before))
        current_rollout,current_step=optimizer._p13_context
        row=dict(rank=rank,rollout_id=int(current_rollout),step_id=int(current_step),gradient_norm=gradient,sampled_master_parameters=len(masters),sampled_master_elements=sum(x.numel() for x in before),sampled_master_changed_elements=changed,lr=float(args.lr),optimizer_kind='v89 StatelessAdam with reset states; not persistent Adam',scope='Actual optimizer.step and actual optimizer masters; sampled delta is a nonzero-update witness, not full tensor checksum')
        with (directory/f'updates-rank-{rank}.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
        if gradient<=0 or changed<=0:raise RuntimeError('P13 real-step nonzero gradient/update witness failed')
        return result
    optimizer._p13_update_observed=True;optimizer.step=measured
