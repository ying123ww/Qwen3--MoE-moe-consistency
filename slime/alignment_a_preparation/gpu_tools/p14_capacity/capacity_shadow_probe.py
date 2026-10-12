"""Eager finite expert-padding shadow; complements full graph whole-model gate."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import p12_experiment as experiment
from memory_capture_hooks import KEY


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--capacity',type=int,choices=(8,16),required=True);parser.add_argument('--variant',choices=experiment.VARIANTS,required=True);args=parser.parse_args()
    experiment.environment(args.variant)
    import torch,sglang
    from inference_benchmark import FROZEN,flush_cache_until_idle
    config=experiment.configure(experiment.base.base_config(),args.variant,args.run/'graph-evidence')
    config['max_running_requests']=8;config['enable_return_routed_experts']=True
    config['disable_cuda_graph']=True;config['cuda_graph_config']['decode']['backend']='disabled'
    hook=config['forward_hooks'][0];hook['hook_factory']='bounded_capacity_dispatch_hooks:install';hook['config']['capacity']=args.capacity;hook['config']['backend']='eager';hook['config']['shadow']=True
    config['forward_hooks'].append(dict(name='p14-memory',target_modules=['model'],before_cuda_graph_capture=True,hook_factory='memory_capture_hooks:install',config=dict(capacity=args.capacity,output_dir=str(args.run/'memory-evidence'))))
    sources=torch.load(FROZEN,map_location='cpu',weights_only=False)['samples'];prompts=[list(map(int,s['tokens'][:-int(s['response_length'])])) for s in sources]
    cases=[]
    for n in (1,2,3,4,7,8,12):cases.append((f'batch{n}',[prompts[i%8] for i in range(n)],[16]*n))
    cases.extend([('variable4',[prompts[i][:n] for i,n in enumerate((17,63,129,192))],[16]*4),('mixed-response4',prompts[:4],[1,4,16,32]),('repeat-prompt8',[prompts[0]]*8,[16]*8)])
    model_config=json.loads((Path(config['model_path'])/'config.json').read_text())
    from full_graph_ab_workload import prepare
    (args.run/'requested-server-args.json').write_text(json.dumps(config,indent=2)+'\n')
    engine=sglang.Engine(**config);results=[]
    try:
        for name,inputs,lengths in cases:
            flush_cache_until_idle(engine)
            sampling=[dict(temperature=0,max_new_tokens=n,ignore_eos=True) for n in lengths]
            outputs=engine.generate(input_ids=inputs,sampling_params=sampling,return_logprob=True,logprob_start_len=-1,return_routed_experts=True,routed_experts_start_len=0)
            if len(outputs)!=len(inputs):raise RuntimeError('Incomplete model outputs')
            records=[]
            for i,out in enumerate(outputs):
                pairs=out['meta_info']['output_token_logprobs'];lp=torch.tensor([x[0] for x in pairs],dtype=torch.float32)
                if len(pairs)!=lengths[i] or not torch.isfinite(lp).all():raise RuntimeError('Incomplete/nonfinite variable response')
                arrays,observations=prepare([out],[inputs[i]],lengths[i],model_config)
                routes=observations[0]
                records.append(dict(prompt=inputs[i],response_length=lengths[i],tokens=[int(x[1]) for x in pairs],logprob_bits=lp.view(torch.int32).tolist(),route_sha256=routes['route_sha256'],route_shape=routes['route_shape'],cached_tokens=out['meta_info'].get('cached_tokens')))
            results.append(dict(case=name,records=records));print(json.dumps(dict(case=name,capacity=args.capacity,complete=True)),flush=True)
        from sglang.srt.managers.io_struct import SetInternalStateReq
        for command in (2,1):
            replies=engine.loop.run_until_complete(asyncio.wait_for(engine.tokenizer_manager.set_internal_state(SetInternalStateReq(server_args={KEY:command})),timeout=120))
            if not replies or not all(replies):raise RuntimeError('P14 memory/guard evidence not acknowledged')
        (args.run/'results.json').write_text(json.dumps(dict(complete=True,capacity=args.capacity,variant=args.variant,cases=results,scope='Finite eager padding shadow plus10greedyworkloads; not CUDAgraph proof. Global max running8 divided by DP2 gives4 per scheduler, bounding admitted decode shape; request batch12 queues safely.'),indent=2)+'\n')
    finally:engine.shutdown()


if __name__=='__main__':main()
