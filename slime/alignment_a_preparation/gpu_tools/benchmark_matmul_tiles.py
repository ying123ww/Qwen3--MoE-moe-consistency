"""Standalone candidate tiles; never changes the registered alignment kernel."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    import torch
    import triton
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_kernel_persistent
    torch.cuda.set_device(0)
    sms=torch.cuda.get_device_properties(0).multi_processor_count
    results=[]
    for family,m,n,dtype in [('router',2,128,torch.float32),('attention_qkv',4,2560,torch.bfloat16),('attention_output',4,2048,torch.bfloat16)]:
        k=2048
        bk=32 if dtype==torch.float32 else 64
        configs=[('original',128,128,8),('m16',16,128,4),('m16_n64',16,64,4)]
        inputs=[]
        for seed in range(3):
            torch.manual_seed(20260912+seed)
            a=torch.randn((m+1,k),device='cuda',dtype=dtype)
            b=torch.randn((n,k),device='cuda',dtype=dtype).t()
            inputs.append((a,b))
        def run(a,b,bm,bn,warps):
            rows=a.shape[0]
            c=torch.empty((rows,n),device='cuda',dtype=dtype)
            grid=(min(sms,triton.cdiv(rows,bm)*triton.cdiv(n,bn)),)
            matmul_kernel_persistent[grid](a,b,c,None,rows,n,k,*a.stride(),*b.stride(),*c.stride(),BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,GROUP_SIZE_M=8,NUM_SMS=sms,A_LARGE=False,B_LARGE=False,C_LARGE=False,HAS_BIAS=False,num_stages=3,num_warps=warps)
            return c
        for label,bm,bn,warps in configs:
            mismatches=0
            batch_mismatches=0
            max_abs=0.
            for a,b in inputs:
                reference=run(a[:m],b,128,128,8)
                candidate=run(a[:m],b,bm,bn,warps)
                combined=run(a,b,bm,bn,warps)[:m]
                bits=torch.int32 if dtype==torch.float32 else torch.int16
                mismatches+=int((reference.view(bits)!=candidate.view(bits)).sum())
                batch_mismatches+=int((candidate.view(bits)!=combined.view(bits)).sum())
                max_abs=max(max_abs,float((reference-candidate).abs().max()))
            a,b=inputs[0]
            for _ in range(5):
                run(a[:m],b,bm,bn,warps)
            gpu=[]
            wall=[]
            for _ in range(5):
                torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                t=time.perf_counter()
                start.record()
                for _ in range(20):
                    run(a[:m],b,bm,bn,warps)
                end.record()
                end.synchronize()
                wall.append((time.perf_counter()-t)*1000/20)
                gpu.append(start.elapsed_time(end)/20)
            results.append(dict(family=family,m=m,k=k,n=n,dtype=str(dtype),tile=label,block_m=bm,block_n=bn,block_k=bk,warps=warps,mismatches=mismatches,batch_mismatches=batch_mismatches,max_abs=max_abs,gpu_interval_ms=statistics.median(gpu),wall_ms=statistics.median(wall)))
    out=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,rows=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Single GPU synthetic random-input microbenchmark; three seeds. CUDA intervals include idle launch gaps. Finite probes do not establish full model exactness, arbitrary batching invariance, or end-to-end speedup. No runtime kernel dispatch changed.')
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
