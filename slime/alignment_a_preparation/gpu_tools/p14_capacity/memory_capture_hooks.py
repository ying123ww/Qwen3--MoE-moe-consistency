"""P14 post-model-setup capture/replay diagnostics; not normal latency timing."""
from collections import Counter
import json
from pathlib import Path
import time

KEY='_slime_p14_memory_dump'


def install(config):
    import torch
    import torch.distributed as dist
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import SetInternalStateReqOutput
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch
    rank=dist.get_rank();directory=Path(config['output_dir']);directory.mkdir(parents=True,exist_ok=True)
    torch.cuda.reset_peak_memory_stats();captures=[];counts=Counter();shapes=Counter()
    def memory():return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    baseline=memory();original=FullCudaGraphBackend.capture_one
    def capture(self,shape_key,*args,**kwargs):
        torch.cuda.synchronize();start=time.perf_counter();result=original(self,shape_key,*args,**kwargs);torch.cuda.synchronize()
        captures.append(dict(size=shape_key.size,capture_and_backend_warmup_s=time.perf_counter()-start,graph_count=len(self._graphs),pool_id=list(self._pool),memory=memory()))
        return result
    FullCudaGraphBackend.capture_one=capture
    replay=FullCudaGraphBackend.replay
    def run(self,shape_key,*args,**kwargs):
        result=replay(self,shape_key,*args,**kwargs);counts['full_replays']+=1;shapes[shape_key.size]+=1;return result
    FullCudaGraphBackend.replay=run
    eager=EagerRunner.execute
    def execute(self,batch,*args,**kwargs):
        if batch.forward_mode.is_decode():counts['eager_decode_during_global_prefill' if get_is_extend_in_batch() else 'pure_eager_decode']+=1
        elif batch.forward_mode.is_idle():counts['eager_idle']+=1
        else:counts['eager_prefill']+=1
        return eager(self,batch,*args,**kwargs)
    EagerRunner.execute=execute
    original_control=Scheduler.set_internal_state
    def control(self,request):
        if set(request.server_args)=={KEY}:
            command=request.server_args[KEY]
            if command==2:
                from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
                import bounded_capacity_dispatch_hooks as bounded
                buffer=DeepEPBuffer._state().buffer
                if buffer is None:raise RuntimeError('Actual DeepEP buffer absent')
                rows=config['capacity']//4+1
                x=torch.zeros((rows,2048),device='cuda',dtype=torch.bfloat16)
                token=bounded._DECODE.set(True)
                try:
                    try:buffer.dispatch(x)
                    except RuntimeError as exc:
                        if 'Unexpected fixed-capacity dispatch input shape/dtype' not in str(exc):raise
                        counts['unsafe_physical_send_rejected']+=1
                    else:raise RuntimeError('Unsafe physical send was not rejected')
                finally:bounded._DECODE.reset(token)
                return SetInternalStateReqOutput(updated=True)
            if command!=1:raise ValueError('Unknown P14 dump command')
            torch.cuda.synchronize()
            report=dict(rank=rank,capacity=config['capacity'],model_setup_baseline=baseline,captures=captures,counts=dict(counts),replay_shapes=dict(shapes),final_memory=memory(),scope='Torch allocator peaks reset after model/pool setup at hook installation; does not cover earlier loading peaks or non-Torch allocations. NVML resident observations are required separately.')
            (directory/f'memory-rank-{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
            return SetInternalStateReqOutput(updated=True)
        return original_control(self,request)
    Scheduler.set_internal_state=control
    return lambda module,inputs,output:None
