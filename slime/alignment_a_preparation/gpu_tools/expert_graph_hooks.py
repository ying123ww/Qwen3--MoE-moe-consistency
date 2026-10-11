"""Opt-in BF16 DeepEP-normal expert graphs; communication stays eager.

Cache exact receive row counts, never routing layouts. Intended for the fixed
Qwen3 HGX03 experiment with decode buckets <=4 and EP4 (at most 16 rows).
"""
from collections import Counter
from contextvars import ContextVar
import json
from pathlib import Path
import time
from types import MethodType

import torch
import triton
import triton.language as tl

_IN_REPLAY = ContextVar('expert_graph_in_outer_replay', default=False)
_INSTALLED = False


@triton.jit
def _pack_inputs(X, I, W, SX, SI, SW, NX: tl.constexpr, NI: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(SX + index, tl.load(X + index, index < NX, other=0), index < NX)
    tl.store(SI + index, tl.load(I + index, index < NI, other=0), index < NI)
    tl.store(SW + index, tl.load(W + index, index < NI, other=0), index < NI)


class ExpertGraph:
    def __init__(self, original, rows, hidden, topk, device, stream):
        from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalDispatchOutput
        self.x = torch.zeros(rows, hidden, dtype=torch.bfloat16, device=device)
        self.ids = torch.zeros(rows, topk, dtype=torch.int64, device=device)
        self.weights = torch.full((rows,topk),1./topk,dtype=torch.float32,device=device)
        packet = DeepEPNormalDispatchOutput(self.x,None,self.ids,self.weights,[])
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                original(packet)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = original(packet)
        torch.cuda.current_stream().wait_stream(stream)

    def run(self, packet):
        _pack_inputs[(triton.cdiv(self.x.numel(),1024),)](
            packet.hidden_states,packet.topk_ids,packet.topk_weights,
            self.x,self.ids,self.weights,NX=self.x.numel(),NI=self.ids.numel(),BLOCK=1024)
        self.graph.replay()
        return self.output


def _compare(actual, expected, *, rank, layer, rows):
    a,e=actual.hidden_states,expected.hidden_states
    if a.shape != e.shape or a.dtype != e.dtype or not torch.equal(a.view(torch.int16),e.view(torch.int16)):
        raise RuntimeError(f'Expert graph bit mismatch: rank={rank} layer={layer} rows={rows}')
    return a.numel()


def install(config):
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError('Install expert graph experiment once per worker')
    _INSTALLED=True
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalDispatchOutput
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import BreakableCudaGraphBackend
    from sglang.srt.runtime_context import get_exec
    import torch.distributed as dist
    rank=dist.get_rank()
    output=Path(config['output_dir'])
    output.mkdir(parents=True,exist_ok=True)
    max_rows=int(config.get('max_rows',16))
    if max_rows != 16:
        raise ValueError('This experiment is bounded to decode<=4 and EP4')
    shadow=bool(config.get('shadow',False))
    state=dict(ready=False,replays=0,zero_fallbacks=0,unsupported_fallbacks=0,
               shadow_calls=0,shadow_elements=0,synthetic_calls=0,synthetic_elements=0)
    row_hist=Counter()
    layer_hist=Counter()
    caches={}
    first_written=False

    def record(name):
        row=dict(state,rank=rank,rows=dict(row_hist),layers=dict(layer_hist),
                 cache_graphs=sum(len(c) for c in caches.values()),packing_kernel_launches_per_hit=1,
                 scope='Expert compute only; exact receive rows; dispatch/combine eager')
        (output/f'{name}-rank-{rank}.json').write_text(json.dumps(row,indent=2)+'\n')

    original_replay=BreakableCudaGraphBackend.replay
    def replay(self,*args,**kwargs):
        nonlocal first_written
        token=_IN_REPLAY.set(True)
        try:
            result=original_replay(self,*args,**kwargs)
        finally:
            _IN_REPLAY.reset(token)
        if not first_written and state['replays']:
            first_written=True
            record('expert-first-replay')
            print('EXPERT_GRAPH_SUCCESSFUL_REPLAY '+json.dumps(dict(rank=rank,replays=state['replays'],
                cache_graphs=sum(len(c) for c in caches.values()))),flush=True)
        if shadow:
            record('expert-shadow')
        return result
    BreakableCudaGraphBackend.replay=replay

    def bind(layer,original,cache):
        layer_id=layer.layer_id
        weight_addresses=(layer.w13_weight.data_ptr(),layer.w2_weight.data_ptr())
        def run(self,dispatch_output):
            if not _IN_REPLAY.get():
                return original(dispatch_output)
            x=dispatch_output.hidden_states
            m=x.shape[0]
            if m==0:
                state['zero_fallbacks']+=1
                # Preserve the existing zero-row workaround and DeepEP payload.
                return original(dispatch_output)
            supported=(isinstance(dispatch_output,DeepEPNormalDispatchOutput)
                and dispatch_output.hidden_states_scale is None
                and 1<=m<=max_rows and x.is_cuda and x.dtype==torch.bfloat16
                and x.shape[1]==2048 and x.is_contiguous()
                and dispatch_output.topk_ids.shape==(m,8)
                and dispatch_output.topk_ids.dtype==torch.int64
                and dispatch_output.topk_weights.shape==(m,8)
                and dispatch_output.topk_weights.dtype==torch.float32
                and dispatch_output.topk_ids.is_contiguous() and dispatch_output.topk_weights.is_contiguous())
            if not supported:
                state['unsupported_fallbacks']+=1
                return original(dispatch_output)
            if weight_addresses!=(self.w13_weight.data_ptr(),self.w2_weight.data_ptr()):
                raise RuntimeError('Expert weights replaced; rebuild the experiment graph cache')
            reference_packet=None
            if shadow:
                reference_packet=dispatch_output._replace(hidden_states=x.clone())
            actual=cache[m].run(dispatch_output)
            if shadow:
                expected=original(reference_packet)
                state['shadow_elements']+=_compare(actual,expected,rank=rank,layer=layer_id,rows=m)
                state['shadow_calls']+=1
            state['replays']+=1
            row_hist[m]+=1
            layer_hist[layer_id]+=1
            return actual
        layer.run_moe_core=MethodType(run,layer)

    def prepare(model,inputs,result):
        if state['ready']:
            return None
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Build expert caches during eager warmup, before outer graph capture')
        if get_exec().moe.enable_fused_moe_sum_all_reduce:
            raise RuntimeError('Expert compute graph must not contain distributed collectives')
        layers=[layer for layer in model.modules() if isinstance(layer,DeepEPMoE)]
        if len(layers)!=48:
            raise RuntimeError(f'Expected 48 Qwen3 expert layers, got {len(layers)}')
        torch.cuda.synchronize()
        started=time.perf_counter()
        before=torch.cuda.memory_allocated()
        stream=torch.cuda.Stream()
        for layer in layers:
            if not layer.quant_method.runner.runner_backend.is_triton():
                raise RuntimeError('Expert graph experiment requires the Triton runner')
            if layer.w13_weight.dtype!=torch.bfloat16 or layer.w2_weight.dtype!=torch.bfloat16:
                raise RuntimeError('Expert graph experiment requires unquantized BF16 weights')
            if layer.moe_runner_config.top_k!=8 or layer.w2_weight.shape[1]!=2048:
                raise RuntimeError('Unexpected Qwen3 expert shape')
            original=layer.run_moe_core
            cache={m:ExpertGraph(original,m,2048,8,layer.w13_weight.device,stream)
                   for m in range(1,max_rows+1)}
            caches[layer.layer_id]=cache
            if shadow and layer.layer_id in (0,47):
                for m,entry in cache.items():
                    for seed,scale in enumerate((.01,1.,10.)):
                        generator=torch.Generator(device=layer.w13_weight.device)
                        generator.manual_seed(20261011+seed+100*m)
                        x=(torch.randn((m,2048),device=layer.w13_weight.device,generator=generator)*scale).to(torch.bfloat16)
                        ids=torch.randint(-1,32,(m,8),device=x.device,generator=generator)
                        ids[:,0]=seed
                        weights=torch.rand((m,8),device=x.device,generator=generator)
                        weights.masked_fill_(ids<0,0.)
                        packet=DeepEPNormalDispatchOutput(x,None,ids,weights,[])
                        actual=entry.run(packet)
                        expected=original(packet._replace(hidden_states=x.clone()))
                        state['synthetic_elements']+=_compare(actual,expected,rank=rank,layer=layer.layer_id,rows=m)
                        state['synthetic_calls']+=1
            bind(layer,original,cache)
        torch.cuda.synchronize()
        state.update(ready=True,build_seconds=time.perf_counter()-started,
                     allocated_memory_delta_bytes=torch.cuda.memory_allocated()-before)
        record('expert-cache')
        print('EXPERT_GRAPH_CACHE_READY '+json.dumps(dict(rank=rank,graphs=48*max_rows,
            build_seconds=state['build_seconds'],allocated_memory_delta_bytes=state['allocated_memory_delta_bytes'])),flush=True)
        return None
    return prepare
