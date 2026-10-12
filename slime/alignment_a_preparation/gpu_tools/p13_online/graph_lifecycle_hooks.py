"""P13-only lifecycle guard; real RL gates must still validate these mechanisms."""
from collections import Counter
import gc
import json
from pathlib import Path
import time
from pointer_ledger import PointerLedger,parameter_addresses,tensor_address,address_digest

KEY='_slime_p13_lifecycle'


def install(config):
    import torch
    import torch.distributed as dist
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager
    from sglang.srt.managers.io_struct import SetInternalStateReqOutput
    runners=[x for x in gc.get_objects() if type(x) is ModelRunner and getattr(x,'model',None) is not None and not x.is_draft_worker]
    if len(runners)!=1:raise RuntimeError('P13 requires exactly one target ModelRunner per worker')
    runner=runners[0];rank=dist.get_rank();directory=Path(config['output_dir']);directory.mkdir(parents=True,exist_ok=True)
    ledger=PointerLedger();release_update_calls=0;counts=Counter();seen=set();retained=[];capture_rows=[]
    def addresses():
        rows={'weight:'+k:v for k,v in parameter_addresses(runner.model).items()}
        pool=runner.token_to_kv_pool
        for kind in ('k_buffer','v_buffer'):
            buffers=getattr(pool,kind,None)
            if buffers is not None:
                for i,t in enumerate(buffers):rows[f'kv:{kind}:{i}']=tensor_address(t)
        request=getattr(runner.req_to_token_pool,'req_to_token',None)
        if request is not None:rows['request:req_to_token']=tensor_address(request)
        return rows
    def save(event,payload):
        row=dict(event=event,rank=rank,time=time.time(),update_calls=ledger.value_updates,capture_generation=ledger.capture_generation,asleep=ledger.asleep,invalidated=ledger.invalidated,**payload)
        with (directory/f'lifecycle-rank-{rank}.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
    original_capture=FullCudaGraphBackend.capture_one
    def capture(self,shape_key,*args,**kwargs):
        result=original_capture(self,shape_key,*args,**kwargs)
        state=addresses();ledger.capture(state)
        row=dict(size=shape_key.size,address_digest=address_digest(state),parameter_count=sum(k.startswith('weight:') for k in state),kv_tensor_count=sum(k.startswith('kv:') for k in state),graph_object_id=id(self._graphs[shape_key]),pool_id=list(self._pool))
        capture_rows.append(row);save('capture',row)
        (directory/f'addresses-size-{shape_key.size}-rank-{rank}.json').write_text(json.dumps(state,indent=2)+'\n')
        return result
    FullCudaGraphBackend.capture_one=capture
    original_replay=FullCudaGraphBackend.replay
    def replay(self,shape_key,*args,**kwargs):
        ledger.require_replay();result=original_replay(self,shape_key,*args,**kwargs);counts['native_replays']+=1
        key=(ledger.value_updates,ledger.capture_generation,shape_key.size)
        if key not in seen:
            seen.add(key);save('first_replay',dict(size=shape_key.size,graph_object_id=id(self._graphs[shape_key]),pool_id=list(self._pool)))
        return result
    FullCudaGraphBackend.replay=replay
    for name in ('update_weights_from_distributed','update_weights_from_tensor','update_weights_from_disk'):
        original=getattr(WeightUpdater,name)
        def make(original,name):
            def update(self,*args,**kwargs):
                result=original(self,*args,**kwargs)
                if self.get_model_runner() is runner:
                    if not result[0]:raise RuntimeError('P13 weight update failed')
                    state=addresses();changes=ledger.observe_update(state);counts[name]+=1
                    save(name,dict(address_changes=changes,address_digest=address_digest(state),load_format=kwargs.get('load_format'),positional_names_count=len(args[0]) if args and isinstance(args[0],list) else None))
                return result
            return update
        setattr(WeightUpdater,name,make(original,name))
    original_release=SchedulerWeightUpdaterManager.release_memory_occupation
    def release(self,request):
        nonlocal release_update_calls
        tags=request.tags or ['weights','kv_cache','cuda_graph']
        discarded='weights' in tags and not runner.server_args.enable_weights_cpu_backup
        release_update_calls=ledger.value_updates
        ledger.release(weights_discarded=discarded);result=original_release(self,request);counts['release']+=1
        save('release',dict(offload_tags=sorted(self.offload_tags)));return result
    SchedulerWeightUpdaterManager.release_memory_occupation=release
    original_resume=SchedulerWeightUpdaterManager.resume_memory_occupation
    def resume(self,request):
        result=original_resume(self,request);counts['resume']+=1
        if not self.offload_tags:
            if not ledger.weight_values_ready:
                if ledger.value_updates<=release_update_calls:raise RuntimeError('Weights were discarded; full restoration must finish before KV wake')
                ledger.mark_weights_restored()
            state=addresses();changes=ledger.resume(state)
            save('resume',dict(address_changes=changes,address_digest=address_digest(state)))
            if ledger.invalidated:
                self.flush_cache();runner.decode_cuda_graph_runner.backend.cleanup();runner.decode_cuda_graph_runner.capture();counts['automatic_rebuild']+=1
                ledger.require_replay();save('automatic_rebuild',dict(reason='captured storage addresses changed after wake'))
        else:save('partial_resume',dict(offload_tags=sorted(self.offload_tags)))
        return result
    SchedulerWeightUpdaterManager.resume_memory_occupation=resume
    original_control=Scheduler.set_internal_state
    def control(self,request):
        if set(request.server_args)=={KEY}:
            value=request.server_args[KEY]
            if not self.is_fully_idle():raise RuntimeError('P13 control requires idle scheduler')
            if value==1:
                (directory/f'summary-rank-{rank}.json').write_text(json.dumps(dict(rank=rank,counts=dict(counts),captures=capture_rows,update_calls=ledger.value_updates,address_digest=address_digest(addresses())),indent=2)+'\n')
            elif value==2:
                if ledger.asleep:raise RuntimeError('Recapture requires wake')
                self.flush_cache();runner.decode_cuda_graph_runner.backend.cleanup();runner.decode_cuda_graph_runner.capture();counts['requested_rebuild']+=1
                ledger.require_replay();save('requested_rebuild',dict())
            elif value==3:
                parameter_name,parameter=next((n,p) for n,p in runner.model.named_parameters() if n.endswith('layers.0.input_layernorm.weight'))
                retained.append(parameter.data);parameter.data=parameter.data.clone()
                changes=ledger.observe_update(addresses())
                if 'weight:'+parameter_name not in changes:raise RuntimeError('Replacement negative did not change captured address')
                try:ledger.require_replay()
                except RuntimeError:counts['replacement_replay_rejected']+=1
                else:raise RuntimeError('Unsafe old replay was accepted')
                save('replacement_negative',dict(parameter=parameter_name,address_changes=changes,old_replay_rejected=True))
            else:raise ValueError('Unknown P13 command')
            return SetInternalStateReqOutput(updated=True)
        return original_control(self,request)
    Scheduler.set_internal_state=control
    return lambda module,inputs,output:None
