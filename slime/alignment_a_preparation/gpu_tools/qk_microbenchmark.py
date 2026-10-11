"""Gate standalone Q/K prototypes against real tensors and upstream arithmetic."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import residual_rms_performance as guard


def strided_inputs(q_cpu, k_cpu, metadata):
    q = torch.empty_strided(tuple(q_cpu.shape), tuple(metadata['q']['stride']), dtype=q_cpu.dtype, device='cuda')
    k = torch.empty_strided(tuple(k_cpu.shape), tuple(metadata['k']['stride']), dtype=k_cpu.dtype, device='cuda')
    q.copy_(q_cpu)
    k.copy_(k_cpu)
    return q, k


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def measure(fn, calls=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    begin.record()
    for _ in range(calls):
        output = fn()
    end.record()
    end.synchronize()
    return dict(wall_us=(time.perf_counter() - start) * 1e6 / calls, gpu_interval_us=begin.elapsed_time(end) * 1000 / calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    lock = open(guard.PROJECT / '.a-tmp/router-tile-performance.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if guard.gpu_pids():
        raise RuntimeError('GPU task exists; wait for profiling to finish')
    run = guard.PREP / 'gpu_runs' / args.run_id
    run.mkdir(exist_ok=False)
    try:
        os.environ.update(guard.environment(run))
        torch.cuda.set_device(0)
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import rms_norm_batch_invariant
        import qk_norm_prototype as candidate
        guard.save(run / 'manifest.json', dict(pid=os.getpid(), pgid=os.getpgrp(), evidence=str(args.evidence),
            source_sha256={str(Path(candidate.__file__)):hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()},
            reference='upstream rms_norm_batch_invariant; compare real captured outputs as well',
            timing='Standalone wall/allocation/launch + GPU interval; not model latency'))
        sample_files = sorted(args.evidence.glob('rank-*-layer-*-m-*.pt'))
        if not sample_files:
            raise RuntimeError('No real Q/K evidence')
        checked, elements = [], 0
        shapes = {}
        for file in sample_files:
            data = torch.load(file, map_location='cpu', weights_only=False)
            q, k = strided_inputs(data['q'], data['k'], data['metadata'])
            qw, kw = data['q_weight'].cuda(), data['k_weight'].cuda()
            h = data['metadata']['head_dim']
            def reference():
                return (rms_norm_batch_invariant(q.reshape(-1,h),qw,data['q_eps']).view(q.shape),
                        rms_norm_batch_invariant(k.reshape(-1,h),kw,data['k_eps']).view(k.shape))
            reference_out = reference()
            if not all(bits_equal(x.cpu(), data[key]) for x,key in zip(reference_out, ('q_output','k_output'))):
                raise RuntimeError('Captured norm is not equivalent to the proposed upstream reference: ' + str(file))
            for mode in ('strided', 'joint'):
                outputs = candidate.forward(q,k,qw,kw,h,data['q_eps'],data['k_eps'],mode)
                if not all(bits_equal(x,y) for x,y in zip(outputs,reference_out)):
                    torch.save(dict(source=str(file),mode=mode,q=q.cpu(),k=k.cpu(),reference=[x.cpu() for x in reference_out],candidate=[x.cpu() for x in outputs]),run/'first-failure.pt')
                    raise RuntimeError('Q/K prototype bit mismatch: ' + mode + ' ' + str(file))
                elements += q.numel() + k.numel()
            checked.append(str(file))
            shapes.setdefault((tuple(q.shape),tuple(k.shape),tuple(q.stride()),tuple(k.stride()),str(qw.dtype),str(kw.dtype)),file)
        # Check accepted weight dtypes and batch sizes beyond the real samples.
        torch.manual_seed(20261011)
        synthetic = 0
        first = torch.load(sample_files[0],map_location='cpu',weights_only=False)
        q_width,k_width = first['q'].shape[1],first['k'].shape[1]
        for m in (1,2,4):
            for scale in (0.,1e-4,1.,100.):
                for wdtype in (torch.bfloat16,torch.float32):
                    packed=torch.randn((m,q_width+2*k_width),device='cuda',dtype=torch.float32).mul_(scale).to(torch.bfloat16)
                    q,k=packed[:,:q_width],packed[:,q_width:q_width+k_width]
                    qw=torch.randn(128,device='cuda',dtype=wdtype);kw=torch.randn_like(qw)
                    refs=(rms_norm_batch_invariant(q.reshape(-1,128),qw,1e-6).view(q.shape),rms_norm_batch_invariant(k.reshape(-1,128),kw,1e-6).view(k.shape))
                    for mode in ('strided','joint'):
                        outs=candidate.forward(q,k,qw,kw,128,1e-6,1e-6,mode)
                        if not all(bits_equal(x,y) for x,y in zip(outs,refs)):
                            raise RuntimeError(f'Synthetic bit gate failed: m={m} scale={scale} weight={wdtype} mode={mode}')
                        elements += q.numel()+k.numel()
                        synthetic += 1
        guard.save(run/'exact-comparison.json',dict(valid=True,real_samples=len(checked),synthetic_mode_cases=synthetic,compared_bf16_elements=elements,real_sample_paths=checked))
        timings=[]
        for shape,file in shapes.items():
            data=torch.load(file,map_location='cpu',weights_only=False)
            q,k=strided_inputs(data['q'],data['k'],data['metadata']);qw,kw=data['q_weight'].cuda(),data['k_weight'].cuda()
            h=data['metadata']['head_dim']
            functions=dict(reference=lambda:(rms_norm_batch_invariant(q.reshape(-1,h),qw,data['q_eps']).view(q.shape),rms_norm_batch_invariant(k.reshape(-1,h),kw,data['k_eps']).view(k.shape)))
            for mode in ('strided','joint'):
                functions[mode]=lambda mode=mode:candidate.forward(q,k,qw,kw,h,data['q_eps'],data['k_eps'],mode)
            trials={name:[] for name in functions}
            order=('reference','strided','joint')
            for index in range(9):
                for name in order[index%3:]+order[:index%3]:
                    trials[name].append(measure(functions[name]))
            medians={name:{key:statistics.median(row[key] for row in rows) for key in ('wall_us','gpu_interval_us')} for name,rows in trials.items()}
            timings.append(dict(shape=str(shape),example=str(file),medians=medians,trials=trials))
            print(json.dumps(dict(shape=str(shape),medians=medians)),flush=True)
        guard.save(run/'timings.json',dict(complete=True,rows=timings))
        guard.save(run/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        guard.save(run/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()))
        raise


if __name__ == '__main__':
    main()
