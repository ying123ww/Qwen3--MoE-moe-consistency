"""Untimed capture/replay proofs for full graph fusion ablations."""
import json
from pathlib import Path


def install(config):
    import torch.distributed as dist
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.batch_invariant_ops import fused_rmsnorm_candidate as rms
    import qk_runtime_hooks as qk
    import router_full_tile_hooks as router
    rank=dist.get_rank();directory=Path(config['output_dir']);directory.mkdir(parents=True,exist_ok=True)
    original=FullCudaGraphBackend.capture_one;captured={};seen=set()
    def capture(self,shape_key,*args,**kwargs):
        before_rms=rms._GRAPH_CAPTURE_CALLS;before_qk=qk._COUNTS['captured'];before_router=router.CAPTURED.copy()
        result=original(self,shape_key,*args,**kwargs)
        counts=router.CAPTURED-before_router
        row=dict(rank=rank,size=shape_key.size,variant=config['variant'],rms_captured_calls=rms._GRAPH_CAPTURE_CALLS-before_rms,qk_captured_calls=qk._COUNTS['captured']-before_qk,router_captured_calls=dict(counts),axis=config['axis'])
        if bool(row['rms_captured_calls'])!=config['axis']['rms']:raise RuntimeError('RMS ablation did not enter the capture')
        if row['qk_captured_calls']!=(48 if config['axis']['qk'] else 0):raise RuntimeError('QK capture coverage mismatch')
        if config['axis']['router']:
            if set(counts)!=set(range(48)) or any(v!=1 for v in counts.values()):raise RuntimeError('Router capture coverage mismatch')
        elif counts:raise RuntimeError('Unexpected router tile capture')
        captured[shape_key.size]=row
        (directory/f'p12-capture-size-{shape_key.size}-rank-{rank}.json').write_text(json.dumps(row,indent=2)+'\n')
        return result
    FullCudaGraphBackend.capture_one=capture
    replay=FullCudaGraphBackend.replay
    def run(self,shape_key,*args,**kwargs):
        result=replay(self,shape_key,*args,**kwargs)
        if shape_key.size not in seen:
            if shape_key.size not in captured:raise RuntimeError('Replay without verified variant capture')
            seen.add(shape_key.size)
            (directory/f'p12-replay-size-{shape_key.size}-rank-{rank}.json').write_text(json.dumps(dict(rank=rank,size=shape_key.size,variant=config['variant'],successful_replay=True,backend_class=type(self).__name__))+'\n')
        return result
    FullCudaGraphBackend.replay=run
    return lambda module,inputs,output:None
