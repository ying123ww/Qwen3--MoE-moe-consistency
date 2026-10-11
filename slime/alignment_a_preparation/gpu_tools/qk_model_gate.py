"""Finite model and batch exact gate for the opt-in Q/K experiment hook."""
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
from router_tile_exact import cases, load_case, compare_samples
sys.path.insert(0,str(guard.PROJECT/'slime/tools/routepack'))
from capture_routes import save_engine_routes


def capture(args):
    os.environ['SLIME_QWEN3_ROUTER_TILE_MODE']='off'
    os.environ['SLIME_QWEN3_ROUTER_TILE_AUDIT']='0'
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_MODE']='candidate'
    os.environ['SLIME_QWEN3_RESIDUAL_RMS_AUDIT']='0'
    sources=torch.load(guard.FROZEN,map_location='cpu',weights_only=False)['samples']
    prompts=[list(map(int,s['tokens'][:-int(s['response_length'])])) for s in sources]
    model_config=json.loads(Path('/storage/data/yingxy2024/Models/Qwen3-30B-A3B/config.json').read_text())
    ref=guard.PREP/'gpu_runs/remaining-profile-hgx03-qk-20261011-01/sanity-candidate/manifest.json'
    config=dict(json.loads(ref.read_text())['requested_server_args'])
    audit=str(args.output/'audit') if args.mode=='shadow' else None
    config.update(enable_return_routed_experts=True,disable_cuda_graph=True,disable_prefill_cuda_graph=True,
        forward_hooks=[dict(name='qk-experiment',target_modules=['model'],hook_factory='qk_runtime_hooks:install',
                           config=dict(mode=args.mode,audit_dir=audit))])
    sampling=dict(temperature=0,max_new_tokens=16,ignore_eos=True)
    args.output.mkdir(exist_ok=False)
    paths=[Path(__file__),Path(__file__).with_name('qk_runtime_hooks.py'),Path(__file__).with_name('qk_norm_prototype.py')]
    manifest=dict(mode=args.mode,requested_server_args=config,sampling=sampling,
                  frozen_sha256=hashlib.sha256(guard.FROZEN.read_bytes()).hexdigest(),
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  scope='Finite greedy four-workload model/batch exact gate; RMS candidate fixed on; no performance claim')
    guard.save(args.output/'manifest.json',manifest)
    import sglang
    engine=sglang.Engine(**config)
    try:
        (args.output/'resolved-server-args.json').write_text(json.dumps(dataclasses.asdict(engine.server_args),indent=2,default=str)+'\n')
        for name,indices,inputs in cases(prompts):
            from inference_benchmark import flush_cache_until_idle
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
                if len(raw)!=16 or not all(math.isfinite(float(x[0])) for x in raw):
                    raise RuntimeError('Invalid exact-gate logprobs')
                values=np.asarray([x[0] for x in raw],dtype='<f4')
                record.update(response_logprob_path=f'logprobs-{i}.npy',response_logprob_bits_path=f'logprob-bits-{i}.npy',
                              effective_logprob_temperature=1.,sampling_temperature=0,input_prompt_length=len(inputs[i]))
                np.save(directory/record['response_logprob_path'],values,allow_pickle=False)
                np.save(directory/record['response_logprob_bits_path'],values.view('<u4'),allow_pickle=False)
            metadata.write_text(''.join(json.dumps(x)+'\n' for x in records))
            print(json.dumps(dict(mode=args.mode,case=name,completed=True)),flush=True)
    finally:
        engine.shutdown()
    if args.mode=='shadow':
        files=list((args.output/'audit').glob('rank-*.json'))
        if len(files)!=8:raise RuntimeError('Missing eight shadow audit records')
        if any(json.loads(p.read_text())['counts'].get('eligible',0)==0 for p in files):
            raise RuntimeError('Shadow did not execute candidate on every rank')
    guard.save(args.output/'complete.json',dict(mode=args.mode,complete=True))


def compare(args):
    failures=[];batch_failures=[]
    for name in ('batch8','batch3','batch1','variable3'):
        ref=load_case(args.reference,name)
        for mode,path in (('shadow',args.shadow),('joint',args.candidate)):
            failures.extend(compare_samples(ref,load_case(path,name),f'{mode}:{name}'))
    for mode,path in (('off',args.reference),('shadow',args.shadow),('joint',args.candidate)):
        original=load_case(path,'batch8')
        batch_failures.extend(compare_samples(original[:3],load_case(path,'batch3'),mode+':batch8-vs-batch3'))
        batch_failures.extend(compare_samples(original[:1],load_case(path,'batch1'),mode+':batch8-vs-batch1'))
    audit=[json.loads(x.read_text()) for x in (args.shadow/'audit').glob('rank-*.json')]
    result=dict(model_exact=not failures,batch_exact=not batch_failures,failures=failures,batch_failures=batch_failures,
                shadow_compared_elements=sum(x['counts'].get('compared_elements',0) for x in audit),
                shadow_eligible_calls=sum(x['counts'].get('eligible',0) for x in audit),
                limitation='Finite greedy model/batch gate; cross-framework training not tested')
    guard.save(args.output,result);print(json.dumps(result),flush=True)
    if failures or batch_failures:raise RuntimeError('Q/K model/batch exact gate failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    cap=sub.add_parser('capture');cap.add_argument('--mode',choices=['off','shadow','joint'],required=True);cap.add_argument('--output',type=Path,required=True)
    cmp=sub.add_parser('compare')
    for name in ('reference','shadow','candidate','output'):cmp.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    capture(args) if args.action=='capture' else compare(args)


if __name__=='__main__':main()
