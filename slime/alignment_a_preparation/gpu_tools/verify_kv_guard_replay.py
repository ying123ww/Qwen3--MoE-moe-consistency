"""Isolated assertion replay tests, queued behind the active formal GPU run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def save(path,value):
    temporary=path.with_suffix('.new');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def child(args):
    import torch
    values=torch.ones(4,dtype=torch.int32,device='cuda')
    message='Unified decode alignment received an empty KV sequence'
    if args.case=='eager-invalid':
        values[1]=0
        rejected=False
        try:
            if bool(torch.any(values<=0).item()):raise ValueError(message)
        except ValueError:rejected=True
        torch.ones(1,device='cuda');torch.cuda.synchronize()
        result=dict(case=args.case,rejected=rejected,exception_type='ValueError',context_usable=True,passed=rejected)
    else:
        for _ in range(3):torch._assert_async(torch.all(values>0),message)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch._assert_async(torch.all(values>0),message)
            output=values*2
        if args.case=='capture-valid':
            samples=[]
            for data in ([1,192,257,1000],[1,1,1,1],[17,63,129,1]):
                values.copy_(torch.tensor(data,dtype=torch.int32,device='cuda'))
                graph.replay();torch.cuda.synchronize()
                actual=output.cpu().tolist()
                if actual!=[x*2 for x in data]:raise RuntimeError('Stale captured inputs')
                samples.append(dict(input=data,output=actual))
            result=dict(case=args.case,passed=True,dynamic_replays=samples)
        else:
            values.copy_(torch.tensor([1,0,193,1],dtype=torch.int32,device='cuda'))
            error=None;context_error=None
            try:graph.replay();torch.cuda.synchronize()
            except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
            try:torch.ones(1,device='cuda');torch.cuda.synchronize()
            except Exception as exc:context_error=dict(type=type(exc).__name__,message=str(exc))
            result=dict(case=args.case,error=error,subsequent_cuda_error=context_error,
                        passed=bool(error and context_error and 'assert' in error['message'].lower()))
    save(args.output,result);print(json.dumps(result),flush=True)
    # A failed device assertion invalidates this test process's CUDA context.
    # Exit it directly; the OS/driver releases its resources.
    if args.case=='capture-invalid':os._exit(0 if result['passed'] else 1)
    if not result['passed']:raise RuntimeError('Guard test failed')


def parent(args):
    if socket.gethostname()!='ai_hgx_03':raise RuntimeError('Use ai_hgx_03')
    args.output.mkdir(parents=True,exist_ok=False)
    project=Path(__file__).resolve().parents[3]
    core=project/'experiments/sglang-main-sm80/python/sglang/srt/layers/attention/triton_backend.py'
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),core)}
    save(args.output/'manifest.json',dict(pid=os.getpid(),wait_run=str(args.wait_run),source_sha256=hashes,
         scope='Standalone assertion primitive; not a negative full-model gate',restore_keepalive=False))
    lock=open(project/'.a-tmp/router-tile-performance.lock','a')
    deadline=time.monotonic()+12000
    while True:
        state=json.loads((args.wait_run/'state.json').read_text())
        save(args.output/'state.json',dict(status='waiting',phase=state.get('phase'),pid=os.getpid(),time=time.time()))
        if state['status'] in ('complete','failed'):
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:pass
        if time.monotonic()>deadline:raise RuntimeError('Formal run did not finish before guard-test deadline')
        time.sleep(5)
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if pids:raise RuntimeError('GPU task remains; assertion tests will not compete with it')
    env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='0'
    results=[]
    for name in ('eager-invalid','capture-valid','capture-invalid'):
        save(args.output/'state.json',dict(status='running',phase=name,pid=os.getpid(),time=time.time()))
        output=args.output/(name+'.json')
        with (args.output/(name+'.log')).open('w') as log:
            completed=subprocess.run([sys.executable,str(Path(__file__)), '--case',name,'--output',str(output)],
                                     env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        if completed.returncode!=0 or not output.exists():raise RuntimeError('Guard child failed: '+name)
        result=json.loads(output.read_text())
        if not result['passed']:raise RuntimeError('Guard assertion did not behave as expected: '+name)
        results.append(result)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in hashes.items()):
        raise RuntimeError('Guard source changed while test was queued')
    save(args.output/'results.json',dict(passed=True,results=results,
         limitation='Negative CUDA assertion primitive tested in a separate process; not a full-model invalid-KV test'))
    save(args.output/'state.json',dict(status='complete',pid=os.getpid(),time=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['eager-invalid','capture-valid','capture-invalid'])
    parser.add_argument('--wait-run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.case:child(args)
    else:
        try:parent(args)
        except BaseException as exc:
            if args.output.exists():save(args.output/'state.json',dict(status='failed',error=str(exc),pid=os.getpid(),time=time.time()))
            raise


if __name__=='__main__':main()
