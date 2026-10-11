"""Finite actual full-graph raw GEMM output and batch-prefix gate."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import os
import sys
import time
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import residual_rms_performance as guard


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--run-id',required=True);args=p.parse_args()
    packets=sorted(args.evidence.glob('raw-size-*.pt'))
    if len(packets)!=24:raise RuntimeError('Expected two ranks, two sizes, two layers, three raw GEMM types')
    records=[json.loads(x.read_text()) for x in args.evidence.glob('complete-size-*-rank-*.json')]
    if {(x['rank'],x['capture_size']) for x in records}!={(r,size) for r in range(8) for size in (2,4)} or not all(x['complete'] for x in records):raise RuntimeError('Incomplete raw snapshot dump')
    lock=open(guard.PROJECT/'.a-tmp/router-tile-performance.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if guard.gpu_pids():raise RuntimeError('Another GPU task is active')
    run=guard.PREP/'gpu_runs'/args.run_id;run.mkdir(exist_ok=False)
    try:
        os.environ.update(guard.environment(run))
        import torch
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import _matmul_persistent_triton
        torch.cuda.set_device(0);rows=[];total=0
        for file in packets:
            data=torch.load(file,map_location='cpu',weights_only=True)
            a=data['input'].cuda();b=data['weight'].cuda().t();y=data['output'].cuda()
            if a.shape[0] not in (1,2,4) or a.shape[1]!=2048:raise RuntimeError('Unexpected raw input shape')
            expected_dtype=torch.float32 if data['component']=='router' else torch.bfloat16
            expected_n={'router':128,'qkv':2560,'output_projection':2048}[data['component']]
            if a.dtype!=expected_dtype or b.shape!=(2048,expected_n):raise RuntimeError('Unexpected observed GEMM type')
            bits=torch.int32 if expected_dtype==torch.float32 else torch.int16
            original=_matmul_persistent_triton(a,b)
            finite=bool(torch.isfinite(original).all())
            mismatch=int((original.view(bits)!=y.view(bits)).sum())
            batch_mismatch=0
            for size in sorted({1,a.shape[0]} | ({2} if a.shape[0]>=2 else set())):
                small=_matmul_persistent_triton(a[:size],b)
                batch_mismatch+=int((small.view(bits)!=y[:size].view(bits)).sum())
            row=dict(file=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),component=data['component'],rank=data['rank'],layer=data['layer'],capture_size=data['size'],input_shape=list(a.shape),weight_transpose_shape=list(b.shape),dtype=str(a.dtype),elements=y.numel(),finite_original_output=finite,raw_output_bit_mismatches=mismatch,batch_prefix_bit_mismatches=batch_mismatch)
            rows.append(row);total+=y.numel();print(json.dumps(row),flush=True)
            # Collect every packet to diagnose which replay bucket is stale.
        passed=all(x['finite_original_output'] and x['raw_output_bit_mismatches']==0 and x['batch_prefix_bit_mismatches']==0 for x in rows)
        guard.save(run/'results.json',dict(complete=passed,packets=24,raw_output_elements=total,rows=rows,scope='Finite raw local GEMM and batch-prefix exact checks; no whole-model optimization or NCU permission claim'))
        if not passed:raise RuntimeError('Raw original GEMM finite bit gate failed')
        guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()));raise


if __name__=='__main__':main()
