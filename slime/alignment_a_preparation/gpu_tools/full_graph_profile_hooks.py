"""P11-only NVTX provenance and external CUDA-event nodes; never used for timing claims."""
from collections import Counter
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path
import re

_CAPTURE_SIZE=ContextVar('p11_capture_size',default=None)
_DEPTH=ContextVar('p11_component_depth',default=0)
_LAYER=ContextVar('p11_layer',default=-1)
_INSTALLED=False


def install(config):
    global _INSTALLED
    if _INSTALLED:raise RuntimeError('One profiler installation per worker')
    _INSTALLED=True
    import torch
    import torch.distributed as dist
    from deep_ep import Buffer
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.linear import QKVParallelLinear,RowParallelLinear
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    from sglang.srt.managers.io_struct import SetInternalStateReqOutput
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock,Qwen3MoeAttention,Qwen3MoeForCausalLM
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    rank=dist.get_rank();directory=Path(config['output_dir']);directory.mkdir(parents=True,exist_ok=True)
    records={};operand_refs={};operand_values={};saved_components=[]
    steps=0;armed=False;last_size=None;capture_started=False
    profile_capture=False;profile_captured_sizes=set()
    counters=Counter();targets=int(config.get('steps',32));tagged=False

    def annotate(owner,name,component):
        original=getattr(owner,name)
        @wraps(original)
        def wrapped(module,*args,**kwargs):
            path=getattr(module,'_p11_path',type(module).__name__)
            layer=getattr(module,'layer_id',_LAYER.get())
            match=re.search(r'layers\.(\d+)',path)
            if match:layer=int(match.group(1))
            shape=tuple(args[0].shape) if args and torch.is_tensor(args[0]) else None
            kind=component
            if component=='norm':
                residual=kwargs.get('residual',args[1] if len(args)>1 else None)
                kind='residual_norm' if residual is not None else 'other_norm'
            label=f'P11/{kind}/rank={rank}/layer={layer}/path={path}/shape={shape}'
            capturing=torch.cuda.is_current_stream_capturing();size=_CAPTURE_SIZE.get();entry=None
            depth=_DEPTH.get();token=_DEPTH.set(depth+1)
            layer_token=_LAYER.set(layer)
            if capturing and size is not None:
                start=torch.cuda.Event(enable_timing=True,external=True)
                end=torch.cuda.Event(enable_timing=True,external=True)
                entry=dict(component=kind,layer=layer,path=path,input_shape=shape,depth=depth,start=start,end=end)
                records.setdefault(size,[]).append(entry);start.record()
            torch.cuda.nvtx.range_push(label)
            try:
                output=original(module,*args,**kwargs)
                if capturing and component in ('qkv','output_projection') and layer in (0,47):
                    if hasattr(module,'weight'):
                        y=output[0] if isinstance(output,tuple) else output
                        operand_refs[(size,component,layer)]=(args[0],module.weight,y)
                return output
            finally:
                torch.cuda.nvtx.range_pop()
                if entry is not None:entry['end'].record()
                _DEPTH.reset(token)
                _LAYER.reset(layer_token)
        setattr(owner,name,wrapped)

    for owner,name,component in (
        (Qwen3MoeForCausalLM,'forward','model_forward'),
        (Qwen3MoeAttention,'forward','attention'),
        (QKVParallelLinear,'forward','qkv'),(RowParallelLinear,'forward','output_projection'),
        (RMSNorm,'forward','norm'),(Qwen3MoeAttention,'apply_qk_norm_rope','qk_rope'),
        (RadixAttention,'forward','attention_core'),
        (Qwen3MoeSparseMoeBlock,'_router_logits','router'),
        (Qwen3MoeSparseMoeBlock,'_select_experts','router_select'),
        (DeepEPMoE,'forward','moe'),(DeepEPMoE,'run_moe_core','experts'),(Buffer,'dispatch','deepep_dispatch'),
        (Buffer,'combine','deepep_combine'),(LogitsProcessor,'forward','logits'),
        (Sampler,'forward','sampling'),(TritonAttnBackend,'init_forward_metadata','attention_metadata'),
        (TritonAttnBackend,'init_forward_metadata_out_graph','metadata_out_graph'),
        (TritonAttnBackend,'init_forward_metadata_in_graph','metadata_in_graph'),
    ):annotate(owner,name,component)

    original_capture=FullCudaGraphBackend.capture_one
    def capture(self,shape_key,*args,**kwargs):
        nonlocal capture_started
        size=shape_key.size;token=_CAPTURE_SIZE.set(size)
        records[size]=[]
        if config.get('nsys') and profile_capture and not capture_started:
            torch.cuda.synchronize();dist.barrier(group=get_tp_group().cpu_group)
            torch.cuda.cudart().cudaProfilerStart();capture_started=True
        try:return original_capture(self,shape_key,*args,**kwargs)
        finally:
            _CAPTURE_SIZE.reset(token)
            profile_captured_sizes.add(size)
            if profile_capture and profile_captured_sizes=={2,4} and config.get('nsys'):
                torch.cuda.synchronize();dist.barrier(group=get_tp_group().cpu_group)
                torch.cuda.cudart().cudaProfilerStop()
    FullCudaGraphBackend.capture_one=capture

    original_replay=FullCudaGraphBackend.replay
    def replay(self,shape_key,*args,**kwargs):
        nonlocal steps,last_size,saved_components
        if armed and steps<targets:
            if steps==0:
                torch.cuda.synchronize();dist.barrier(group=get_tp_group().cpu_group)
                if config.get('nsys'):torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f'P11/steady_decode/rank={rank}/step={steps}/size={shape_key.size}')
            try:output=original_replay(self,shape_key,*args,**kwargs)
            finally:torch.cuda.nvtx.range_pop()
            steps+=1;last_size=shape_key.size;counters[str(shape_key.size)]+=1
            if steps==targets:
                torch.cuda.synchronize();dist.barrier(group=get_tp_group().cpu_group)
                if config.get('nsys'):torch.cuda.cudart().cudaProfilerStop()
                saved_components=[]
                for entry in records[last_size]:
                    row={k:v for k,v in entry.items() if k not in ('start','end')}
                    row['last_replay_gpu_ms']=entry['start'].elapsed_time(entry['end']);saved_components.append(row)
                if rank in (0,4):
                    for key,(x,w,y) in operand_refs.items():
                        if key[0]==last_size:operand_values[key]=(x.detach().clone(),w,y.detach().clone())
            return output
        return original_replay(self,shape_key,*args,**kwargs)
    FullCudaGraphBackend.replay=replay

    original_control=Scheduler.set_internal_state
    def control(self,request):
        nonlocal armed,steps,last_size,profile_capture,capture_started
        if set(request.server_args)=={'_slime_full_graph_profile'}:
            command=request.server_args['_slime_full_graph_profile']
            if command==1:armed=True;steps=0;last_size=None;counters.clear()
            elif command==3:
                # First prewarm real prefill, decode and sampling requests. Lazy
                # JIT loads after a profiling start/stop can crash Nsight even
                # outside its active range. Recapture the original model only
                # once all request paths have been warmed, for node provenance.
                if armed or not config.get('nsys'):raise RuntimeError('Invalid recapture state')
                if profile_capture:raise RuntimeError('Only one profiled recapture is allowed')
                profile_capture=True;capture_started=False;profile_captured_sizes.clear()
                self.tp_worker.model_runner.decode_cuda_graph_runner.capture()
                if profile_captured_sizes!={2,4}:raise RuntimeError('Incomplete profiled recapture')
                profile_capture=False
            elif command==2:
                armed=False;torch.cuda.synchronize()
                if last_size is None or steps!=targets:raise RuntimeError('Incomplete steady decode profile')
                report=dict(rank=rank,pid=__import__('os').getpid(),mode=config['mode'],steps=steps,
                    replay_shapes=dict(counters),last_size=last_size,components=saved_components,
                    scope='Source-marked inclusive CUDA event intervals of last selected replay; includes dependencies; not additive wall partition')
                (directory/f'components-rank-{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
                if rank in (0,4):
                    for (size,kind,layer),(x,w,y) in operand_values.items():
                        if size==last_size:
                            torch.save(dict(input=x.detach().cpu(),weight=w.detach().cpu(),output=y.detach().cpu(),
                                rank=rank,layer=layer,component=kind,size=size,mode=config['mode']),
                                directory/f'operand-{kind}-layer-{layer}-rank-{rank}.pt')
            else:raise ValueError('Unknown profiling command')
            return SetInternalStateReqOutput(updated=True)
        return original_control(self,request)
    Scheduler.set_internal_state=control

    def tag_modules(model,inputs,output):
        nonlocal tagged
        if not tagged:
            for path,module in model.named_modules():module._p11_path=path
            tagged=True
        return None
    return tag_modules
