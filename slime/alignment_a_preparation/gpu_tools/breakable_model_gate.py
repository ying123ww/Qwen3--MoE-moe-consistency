"""Finite greedy tokens, routes and logprob-bit gate for Plan 2."""
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

import residual_rms_performance as guard
from breakable_experiment import configure
from router_tile_exact import cases, load_case, compare_samples
sys.path.insert(0,str(guard.PROJECT/'slime/tools/routepack'))
from capture_routes import save_engine_routes


def capture(args):
    os.environ.update(SLIME_QWEN3_ROUTER_TILE_MODE='off',SLIME_QWEN3_ROUTER_TILE_AUDIT='0',
        SLIME_QWEN3_RESIDUAL_RMS_MODE='candidate' if args.mode in ('plan1','breakable-rms','breakable-rms-qk','breakable-rms-experts') else 'off',
        SLIME_QWEN3_RESIDUAL_RMS_AUDIT='0',SGLANG_MEMORY_SAVER_CUDA_GRAPH='0')
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_CUDA_GRAPH']='1' if args.mode in ('breakable-rms','breakable-rms-qk','breakable-rms-experts') else '0'
    sources=torch.load(guard.FROZEN,map_location='cpu',weights_only=False)['samples']
    prompts=[list(map(int,s['tokens'][:-int(s['response_length'])])) for s in sources]
    reference=guard.PREP/'gpu_runs/remaining-profile-hgx03-qk-20261011-01/sanity-candidate/manifest.json'
    config=dict(json.loads(reference.read_text())['requested_server_args'])
    config['enable_return_routed_experts']=True
    config=configure(config,args.mode,args.output/'graph-evidence')
    sampling=dict(temperature=0,max_new_tokens=16,ignore_eos=True)
    args.output.mkdir(exist_ok=False)
    model_config=json.loads((Path(config['model_path'])/'config.json').read_text())
    paths=[Path(__file__),Path(__file__).with_name('breakable_experiment.py')]
    guard.save(args.output/'manifest.json',dict(mode=args.mode,requested_server_args=config,sampling=sampling,
        frozen_sha256=hashlib.sha256(guard.FROZEN.read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        residual_rms_mode=os.environ['SLIME_QWEN3_RESIDUAL_RMS_MODE'],scope='Finite exact gate; not timed'))
    import sglang
    from inference_benchmark import flush_cache_until_idle
    engine=sglang.Engine(**config)
    try:
        guard.save(args.output/'resolved-server-args.json',dataclasses.asdict(engine.server_args))
        for name,indices,inputs in cases(prompts):
            flush_cache_until_idle(engine)
            outputs=engine.generate(input_ids=inputs,sampling_params=sampling,return_logprob=True,
                logprob_start_len=-1,return_routed_experts=True,routed_experts_start_len=0)
            directory=args.output/f'case-{name}'/'captured'
            save_engine_routes(outputs,inputs,[sources[i] for i in indices],directory,
                               model_version=config['model_path'],model_config=model_config)
            metadata=directory/'samples_captured.jsonl'
            records=[json.loads(x) for x in metadata.read_text().splitlines()]
            for i,(out,record) in enumerate(zip(outputs,records)):
                raw=out['meta_info']['output_token_logprobs']
                if len(raw)!=16 or not all(math.isfinite(float(x[0])) for x in raw):raise RuntimeError('Invalid logprobs')
                values=np.asarray([x[0] for x in raw],dtype='<f4')
                record.update(response_logprob_path=f'logprobs-{i}.npy',response_logprob_bits_path=f'logprob-bits-{i}.npy',
                    effective_logprob_temperature=1.,sampling_temperature=0,input_prompt_length=len(inputs[i]))
                np.save(directory/record['response_logprob_path'],values,allow_pickle=False)
                np.save(directory/record['response_logprob_bits_path'],values.view('<u4'),allow_pickle=False)
            metadata.write_text(''.join(json.dumps(x)+'\n' for x in records))
            print(json.dumps(dict(mode=args.mode,case=name,completed=True)),flush=True)
    finally:engine.shutdown()
    guard.save(args.output/'complete.json',dict(mode=args.mode,complete=True))


def compare(args):
    failures=[];batch_failures=[]
    for name in ('batch8','batch3','batch1','variable3'):
        failures.extend(compare_samples(load_case(args.reference,name),load_case(args.candidate,name),name))
    for mode,path in (('eager',args.reference),('breakable',args.candidate)):
        original=load_case(path,'batch8')
        batch_failures.extend(compare_samples(original[:3],load_case(path,'batch3'),mode+':batch8-vs-batch3'))
        batch_failures.extend(compare_samples(original[:1],load_case(path,'batch1'),mode+':batch8-vs-batch1'))
    result=dict(model_exact=not failures,batch_exact=not batch_failures,failures=failures,batch_failures=batch_failures,
        limitation='Finite greedy tokens/routes/logprob-bit and batch gate; cross-framework training not tested')
    guard.save(args.output,result);print(json.dumps(result),flush=True)
    if failures or batch_failures:raise RuntimeError('Plan 2 exact gate failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    cap=sub.add_parser('capture');cap.add_argument('--mode',choices=['eager','breakable','breakable-rms','breakable-rms-qk','breakable-rms-experts','plan1'],required=True);cap.add_argument('--output',type=Path,required=True)
    cmp=sub.add_parser('compare')
    for name in ('reference','candidate','output'):cmp.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();capture(args) if args.action=='capture' else compare(args)


if __name__=='__main__':main()
