"""Use DeepEP intranode fixed receive capacity only for decode.

No routing handle is cached. Real and padding rows are identified on the GPU;
the existing Triton expert core consumes the fixed-capacity dispatch output.
"""
from collections import Counter
from contextvars import ContextVar
import json
from pathlib import Path

import torch

_DECODE=ContextVar('fixed_capacity_decode',default=False)
_INSTALLED=False


def install(config):
    global _INSTALLED
    if _INSTALLED:raise RuntimeError('Install fixed-capacity experiment once per worker')
    _INSTALLED=True
    from deep_ep import Buffer
    from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import BreakableCudaGraphBackend
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    import torch.distributed as dist
    rank=dist.get_rank();backend=config['backend'];capacity=int(config['capacity']);shadow=bool(config['shadow'])
    if capacity!=16:raise ValueError('This experiment requires capacity16, EP4 and decode bucket<=4')
    output=Path(config['output_dir']);output.mkdir(parents=True,exist_ok=True)
    counts=Counter();source_rows=Counter();captured_layers=set();capture_dispatches=0;captured_by_shape={}

    def save(name,extra=None):
        row=dict(rank=rank,backend=backend,capacity=capacity,counts=dict(counts),source_rows=dict(source_rows),
            captured_layers=sorted(captured_layers),captured_dispatch_calls=capture_dispatches,
            scope='Fixed-capacity intranode normal decode; prefill unchanged')
        if extra:row.update(extra)
        (output/f'{name}-rank-{rank}.json').write_text(json.dumps(row,indent=2)+'\n')

    original_forward=Qwen3MoeSparseMoeBlock.forward_deepep
    def forward(self,hidden_states,forward_batch):
        token=_DECODE.set(forward_batch.forward_mode.is_decode_or_idle())
        try:return original_forward(self,hidden_states,forward_batch)
        finally:_DECODE.reset(token)
    Qwen3MoeSparseMoeBlock.forward_deepep=forward

    original_dispatch=Buffer.dispatch
    def dispatch(self,x,*args,**kwargs):
        nonlocal capture_dispatches
        fixed=_DECODE.get() or torch.cuda.is_current_stream_capturing()
        if not fixed:return original_dispatch(self,x,*args,**kwargs)
        if self.group_size!=4 or self.runtime.get_num_rdma_ranks()!=1:
            raise RuntimeError('Fixed-capacity dispatch requires single-node EP4')
        if not torch.is_tensor(x) or x.dtype!=torch.bfloat16 or x.ndim!=2 or x.shape[1]!=2048 or x.shape[0]>4:
            raise RuntimeError('Unexpected fixed-capacity dispatch input shape/dtype')
        if kwargs.get('expert_alignment',1)!=1 or kwargs.get('handle') is not None:
            raise RuntimeError('Fixed-capacity experiment needs uncached, unaligned normal dispatch')
        kwargs['num_worst_tokens']=capacity
        result=original_dispatch(self,x,*args,**kwargs)
        assert result[0].shape==(capacity,2048) and len(result[3])==0
        source_rows[x.shape[0]]+=1
        if torch.cuda.is_current_stream_capturing():capture_dispatches+=1
        else:counts['fixed_eager_dispatch']+=1
        return result
    Buffer.dispatch=dispatch

    original_core=DeepEPMoE.run_moe_core
    def core(self,dispatch_output):
        if not (_DECODE.get() or torch.cuda.is_current_stream_capturing()):
            return original_core(self,dispatch_output)
        x=dispatch_output.hidden_states;ids=dispatch_output.topk_ids;weights=dispatch_output.topk_weights
        if x.shape!=(capacity,2048) or dispatch_output.hidden_states_scale is not None:
            raise RuntimeError('Unexpected fixed-capacity expert payload')
        capturing=torch.cuda.is_current_stream_capturing()
        if capturing:captured_layers.add(self.layer_id)
        # DeepEP only marks padding route IDs as -1; padding X/weights are
        # otherwise uninitialized. Sanitize them on device, without .item().
        valid=(ids>=0).any(dim=1)
        x.masked_fill_(~valid[:,None],0.)
        weights.masked_fill_(ids<0,0.)
        expected_packet=None
        if shadow and not capturing:
            # Drop only padding rows to recover the original actual-M computation.
            expected_packet=dispatch_output._replace(hidden_states=x[valid].clone(),
                topk_ids=ids[valid].clone(),topk_weights=weights[valid].clone())
        result=original_core(self,dispatch_output)
        if expected_packet is not None:
            expected=original_core(self,expected_packet)
            actual=result.hidden_states[valid]
            if not torch.equal(actual.view(torch.int16),expected.hidden_states.view(torch.int16)):
                raise RuntimeError(f'Fixed-capacity expert bit mismatch rank={rank} layer={self.layer_id}')
            counts['shadow_calls']+=1;counts['shadow_elements']+=actual.numel()
            counts['empty_receive_calls']+=int(expected_packet.hidden_states.shape[0]==0)
        return result
    DeepEPMoE.run_moe_core=core

    if backend=='breakable':
        original_replay=BreakableCudaGraphBackend.replay
        first=False
        def replay(self,*args,**kwargs):
            nonlocal first
            token=_DECODE.set(True)
            try:result=original_replay(self,*args,**kwargs)
            finally:_DECODE.reset(token)
            counts['breakable_replays']+=1
            if shadow or not first:
                save('fixed-replay');first=True
            return result
        BreakableCudaGraphBackend.replay=replay
    if backend=='full':
        original_capture=FullCudaGraphBackend.capture_one
        def capture(self,shape_key,*args,**kwargs):
            before=capture_dispatches
            result=original_capture(self,shape_key,*args,**kwargs)
            captured_by_shape[shape_key.size]=capture_dispatches-before
            if captured_by_shape[shape_key.size]!=48:raise RuntimeError('Full graph must capture all 48 dispatches')
            save(f'fixed-capture-size-{shape_key.size}',dict(size=shape_key.size,dispatches_this_shape=captured_by_shape[shape_key.size]))
            print('FIXED_CAPACITY_FULL_CAPTURE '+json.dumps(dict(rank=rank,size=shape_key.size,dispatches=48)),flush=True)
            return result
        FullCudaGraphBackend.capture_one=capture
        original_replay=FullCudaGraphBackend.replay;seen=set()
        def replay(self,shape_key,*args,**kwargs):
            result=original_replay(self,shape_key,*args,**kwargs)
            counts['full_replays']+=1
            if shape_key.size not in seen:
                seen.add(shape_key.size)
                save(f'fixed-full-replay-size-{shape_key.size}',dict(size=shape_key.size,successful_replay=True,
                    backend_class=type(self).__name__))
                print('FIXED_CAPACITY_FULL_REPLAY '+json.dumps(dict(rank=rank,size=shape_key.size)),flush=True)
            return result
        FullCudaGraphBackend.replay=replay

    def flush(model,inputs,result):
        if shadow and not torch.cuda.is_current_stream_capturing():save('fixed-shadow')
        return None
    return flush
