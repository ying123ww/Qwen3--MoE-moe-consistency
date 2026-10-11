"""Reset/flush replay and fallback counters via the untimed tokenizer broadcast."""
from collections import Counter
import json
from pathlib import Path

_INSTALLED=False


def install(config):
    global _INSTALLED
    if _INSTALLED:raise RuntimeError('Install the runtime audit once per worker')
    _INSTALLED=True
    import torch.distributed as dist
    from deep_ep import Buffer
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch
    from sglang.srt.managers.io_struct import SetInternalStateReqOutput
    from full_graph_ab_control import KEY,decode
    rank=dist.get_rank();directory=Path(config['output_dir'])
    counts=Counter();shapes=Counter();communication={};eager_details=[];armed=False
    fields=('num_sms','num_max_nvl_chunked_send_tokens','num_max_nvl_chunked_recv_tokens',
            'num_max_rdma_chunked_send_tokens','num_max_rdma_chunked_recv_tokens')
    # Installed deep_ep_cpp.Config has no public field getters. Observe the
    # actual constructor arguments; never infer them from an opaque repr.
    import deep_ep.buffer as buffer_module
    import sglang.srt.layers.moe.token_dispatcher.deepep as dispatcher_module
    config_class=buffer_module.Config;constructed={};latest=[None]
    def config_constructor(*args,**kwargs):
        values=dict(zip(fields,(20,6,256,6,256)))
        if len(args)>len(fields):raise ValueError('Unexpected Config positional arguments')
        values.update(zip(fields,args));values.update(kwargs)
        result=config_class(*args,**kwargs)
        constructed[id(result)]={k:int(values[k]) for k in fields};latest[0]=result
        return result
    buffer_module.Config=config_constructor
    dispatcher_module.Config=config_constructor

    original_replay=FullCudaGraphBackend.replay
    def replay(self,shape_key,*args,**kwargs):
        if armed:counts['full_replays']+=1;shapes[shape_key.size]+=1
        return original_replay(self,shape_key,*args,**kwargs)
    FullCudaGraphBackend.replay=replay

    original_eager=EagerRunner.execute
    def eager(self,forward_batch,*args,**kwargs):
        if armed:
            mode=forward_batch.forward_mode
            if mode.is_decode():
                global_mode=getattr(forward_batch,'global_forward_mode',None)
                global_prefill=bool(get_is_extend_in_batch()) or (
                    global_mode is not None and global_mode.is_extend(include_draft_extend_v2=True))
                counts['raw_eager_local_decode']+=1
                counts['eager_decode_during_global_prefill' if global_prefill else 'eager_decode']+=1
                eager_details.append(dict(local_mode=str(mode),global_mode=str(global_mode),
                    is_extend_in_batch=bool(get_is_extend_in_batch()),batch_size=forward_batch.batch_size,
                    input_rows=forward_batch.input_ids.shape[0],
                    original_global_num_tokens_cpu=getattr(forward_batch,'original_global_num_tokens_cpu',None),
                    can_run_dp_cuda_graph=getattr(forward_batch,'can_run_dp_cuda_graph',None),
                    classified_global_prefill=global_prefill))
            elif mode.is_idle():
                global_mode=getattr(forward_batch,'global_forward_mode',None)
                if global_mode is not None and global_mode.is_decode():counts['eager_decode_idle']+=1
                else:counts['eager_other_idle']+=1
            else:counts['eager_prefill']+=1
        return original_eager(self,forward_batch,*args,**kwargs)
    EagerRunner.execute=eager

    def wrap_communication(name):
        original=getattr(Buffer,name)
        def call(self,*args,**kwargs):
            chosen=kwargs.get('config')
            if chosen is not None and id(chosen) not in constructed:
                raise RuntimeError('Communication Config predates constructor audit')
            latest[0]=None
            result=original(self,*args,**kwargs)
            if chosen is None:chosen=latest[0]
            if chosen is None or id(chosen) not in constructed:
                raise RuntimeError('Actual communication Config was not observed')
            record=constructed[id(chosen)]
            previous=communication.get(name)
            if previous is not None and previous!=record:
                raise RuntimeError('Communication config changed within an Engine')
            communication[name]=record
            return result
        setattr(Buffer,name,call)
    wrap_communication('dispatch');wrap_communication('combine')

    def rpc(self,action,phase):
        nonlocal armed
        if not phase or not all(c.isalnum() or c in '-_' for c in phase):raise ValueError('Invalid phase label')
        if action=='reset':counts.clear();shapes.clear();eager_details.clear();armed=True
        elif action=='flush':
            directory.mkdir(parents=True,exist_ok=True)
            data=dict(rank=rank,mode=config['mode'],phase=phase,armed=armed,counts=dict(counts),
                      replay_shapes=dict(shapes),communication=communication,eager_decode_details=eager_details,
                      scope='Pure global decode must not fall back; DP-local decode in global prefill is reported separately')
            (directory/f'runtime-{phase}-rank-{rank}.json').write_text(json.dumps(data,indent=2)+'\n')
            armed=False
        else:raise ValueError(action)
    original_control=Scheduler.set_internal_state
    def set_internal_state(self,request):
        if set(request.server_args)=={KEY}:
            action,phase=decode(request.server_args[KEY])
            rpc(self,action,phase)
            print(f'FULL_GRAPH_AUDIT_CONTROL rank={rank} action={action} phase={phase}',flush=True)
            return SetInternalStateReqOutput(updated=True)
        return original_control(self,request)
    Scheduler.set_internal_state=set_internal_state
    def observe(model,inputs,output):return None
    return observe
