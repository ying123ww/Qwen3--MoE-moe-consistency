"""Real-operand BF16 tile investigation; registered matmul is never changed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import time

import torch
import triton
import residual_rms_performance as guard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if guard.gpu_pids():raise RuntimeError('Other GPU task exists before standalone projection probe')
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    try:
        os.environ.update(guard.environment(run));torch.cuda.set_device(0)
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_kernel_persistent
        sms=torch.cuda.get_device_properties(0).multi_processor_count
        configs=[('original',128,128,8),('m16_n128',16,128,4),('m16_n64',16,64,4),('m16_n32',16,32,4),('m32_n64',32,64,4)]
        files=sorted(args.evidence.glob('rank-*-layer-*.pt'))
        if len(files)!=8:raise RuntimeError('Expected 2 layers x 2 projection types x 2 ranks')
        results=[];checked=[]
        for file in files:
            d=torch.load(file,map_location='cpu',weights_only=False)
            a=d['input'].cuda();b=d['weight'].cuda().t();m,k=a.shape;n=b.shape[1]
            if a.dtype!=torch.bfloat16 or (m,k,n) not in ((4,2048,2560),(4,2048,2048)):
                raise RuntimeError('Projection shape differs from current profiler evidence')
            def calculate(a,config):
                label,bm,bn,warps=config
                c=torch.empty((a.shape[0],n),device=a.device,dtype=a.dtype)
                grid=(min(sms,triton.cdiv(a.shape[0],bm)*triton.cdiv(n,bn)),)
                matmul_kernel_persistent[grid](a,b,c,None,a.shape[0],n,k,*a.stride(),*b.stride(),*c.stride(),
                    BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=64,GROUP_SIZE_M=8,NUM_SMS=sms,
                    A_LARGE=False,B_LARGE=False,C_LARGE=False,HAS_BIAS=False,num_stages=3,num_warps=warps)
                return c
            reference=calculate(a,configs[0])
            if not torch.equal(reference.cpu().view(torch.int16),d['output'].view(torch.int16)):
                raise RuntimeError('Standalone original does not match captured model projection')
            for config in configs:
                candidate=calculate(a,config)
                differences=int((reference.view(torch.int16)!=candidate.view(torch.int16)).sum())
                batch_differences=0
                for rows in (1,2,4):
                    small=calculate(a[:rows],config)
                    batch_differences+=int((small.view(torch.int16)!=candidate[:rows].view(torch.int16)).sum())
                if differences or batch_differences:
                    checked.append(dict(file=str(file),tile=config[0],valid=False,mismatches=differences,batch_mismatches=batch_differences))
                    continue
                checked.append(dict(file=str(file),tile=config[0],valid=True,mismatches=0,batch_mismatches=0))
            trials={config[0]:[] for config in configs if next(x for x in reversed(checked) if x['file']==str(file) and x['tile']==config[0])['valid']}
            for round_id in range(9):
                order=configs[round_id%len(configs):]+configs[:round_id%len(configs)]
                for config in order:
                    if config[0] not in trials:continue
                    for _ in range(5):calculate(a,config)
                    torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    t=time.perf_counter();start.record()
                    for _ in range(50):calculate(a,config)
                    end.record();end.synchronize()
                    trials[config[0]].append(dict(wall_us=(time.perf_counter()-t)*1e6/50,gpu_interval_us=start.elapsed_time(end)*1000/50))
            row=dict(example=str(file),metadata=d['metadata'],medians={name:{key:statistics.median(x[key] for x in data) for key in ('wall_us','gpu_interval_us')} for name,data in trials.items()})
            results.append(row);print(json.dumps(row),flush=True)
        guard.save(run/'results.json',dict(complete=True,checks=checked,rows=results,
            limitation='Finite real-matrix exact/batch checks and single-GPU intervals (including host launch gaps); no model integration or whole-model speed claim'))
        guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise


if __name__=='__main__':main()
