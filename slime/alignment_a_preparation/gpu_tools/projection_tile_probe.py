"""Raw real operands: exact/batch gate and interleaved CUDA Graph kernel timing."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import residual_rms_performance as guard


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--run-id',required=True);args=p.parse_args()
    files=sorted(args.evidence.glob('raw-size-*.pt'))
    if len(files)!=24:raise RuntimeError('Expected validated per-bucket raw packets')
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if guard.gpu_pids():raise RuntimeError('Other GPU task active')
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    try:
        os.environ.update(guard.environment(run))
        import torch,triton
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_kernel_persistent
        torch.cuda.set_device(0);sms=torch.cuda.get_device_properties(0).multi_processor_count
        results=[]
        for file in files:
            data=torch.load(file,map_location='cpu',weights_only=True);a=data['input'].cuda();b=data['weight'].cuda().t();expected=data['output'].cuda()
            m,k=a.shape;n=b.shape[1];bk=32 if a.dtype==torch.float32 else 64;bits=torch.int32 if a.dtype==torch.float32 else torch.int16
            def compute(x,tile,out=None):
                bm,bn,warps=tile
                c=torch.empty((x.shape[0],n),device=x.device,dtype=x.dtype) if out is None else out
                grid=(min(sms,triton.cdiv(x.shape[0],bm)*triton.cdiv(n,bn)),)
                matmul_kernel_persistent[grid](x,b,c,None,x.shape[0],n,k,*x.stride(),*b.stride(),*c.stride(),BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,GROUP_SIZE_M=8,NUM_SMS=sms,A_LARGE=False,B_LARGE=False,C_LARGE=False,HAS_BIAS=False,num_stages=3,num_warps=warps)
                return c
            configs={'original':(128,128,8),'small':(16,64,4)};checked={};graphs={};outputs={}
            for label,tile in configs.items():
                c=compute(a,tile);mismatch=int((c.view(bits)!=expected.view(bits)).sum());prefix_mismatch=0
                for rows in sorted({1,m}|({2} if m>=2 else set())):
                    y=compute(a[:rows],tile);prefix_mismatch+=int((y.view(bits)!=expected[:rows].view(bits)).sum())
                checked[label]=dict(raw_mismatches=mismatch,batch_prefix_mismatches=prefix_mismatch,finite=bool(torch.isfinite(c).all()))
                if mismatch or prefix_mismatch or not checked[label]['finite']:raise RuntimeError(f'Finite raw gate failed {file} {label}')
                for _ in range(5):compute(a,tile,c)
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(100):compute(a,tile,c)
                graphs[label]=graph;outputs[label]=c
                graph.replay();torch.cuda.synchronize()
                if not torch.equal(c.view(bits),expected.view(bits)):raise RuntimeError('Captured tile output mismatch')
            trials={k:[] for k in configs}
            for r in range(9):
                order=['original','small'] if r%2==0 else ['small','original']
                for label in order:
                    for _ in range(3):graphs[label].replay()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[label].replay()
                    end.record();end.synchronize()
                    trials[label].append(start.elapsed_time(end)*1000/100)
            medians={k:statistics.median(v) for k,v in trials.items()}
            row=dict(file=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),rank=data['rank'],layer=data['layer'],component=data['component'],capture_size=data['size'],M=m,K=k,N=n,dtype=str(a.dtype),checks=checked,graph_us=trials,median_graph_us=medians,change_percent=(medians['small']/medians['original']-1)*100,original_grid=min(sms,triton.cdiv(m,128)*triton.cdiv(n,128)),small_grid=min(sms,triton.cdiv(m,16)*triton.cdiv(n,64)))
            results.append(row);print(json.dumps(row),flush=True)
        guard.save(run/'results.json',dict(complete=True,rows=results,scope='Finite real raw output/batch/graph gate. Single-GPU 100-kernel CUDA Graph burst interval /100, warm same-weight cache, no per-kernel host launch loop or model speed claim. K reduction and accumulation unchanged.'))
        guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise


if __name__=='__main__':main()
