"""P11 instrumented capture and 32-replay window; no normal performance claims."""
import argparse
import asyncio
import dataclasses
import json
from pathlib import Path

import full_graph_ab_experiment as experiment
from inference_benchmark import FROZEN,flush_cache_until_idle


def command(engine,value):
    from sglang.srt.managers.io_struct import SetInternalStateReq
    request=SetInternalStateReq(server_args={'_slime_full_graph_profile':value})
    replies=engine.loop.run_until_complete(asyncio.wait_for(
        engine.tokenizer_manager.set_internal_state(request),timeout=120))
    if not replies or not all(replies):raise RuntimeError('Profile control was not acknowledged')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('A','C'),required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--nsys',action='store_true')
    args=parser.parse_args();experiment.environment(args.mode)
    experiment.allow_full_configuration(args.run,args.mode)
    import torch
    import sglang
    inputs=torch.load(FROZEN,map_location='cpu',weights_only=False)['samples']
    prompts=[list(map(int,s['tokens'][:-int(s['response_length'])])) for s in inputs]
    config=experiment.configure(experiment.base_config(),args.mode,args.run/'graph-evidence')
    config['enable_return_routed_experts']=True
    config['forward_hooks'].append(dict(name='p11-source-profile',target_modules=['model'],
        before_cuda_graph_capture=True,hook_factory='full_graph_profile_hooks:install',
        config=dict(mode=args.mode,steps=32,nsys=args.nsys,output_dir=str(args.run/'profile-evidence'))))
    (args.run/'requested-server-args.json').write_text(json.dumps(config,indent=2)+'\n')
    engine=sglang.Engine(**config)
    try:
        (args.run/'resolved-server-args.json').write_text(json.dumps(dataclasses.asdict(engine.server_args),indent=2,default=str)+'\n')
        flush_cache_until_idle(engine)
        engine.generate(input_ids=prompts[:3],sampling_params=dict(temperature=0,max_new_tokens=2,ignore_eos=True))
        for _ in range(3):
            flush_cache_until_idle(engine)
            engine.generate(input_ids=prompts,sampling_params=dict(temperature=.6,max_new_tokens=128,ignore_eos=True))
        flush_cache_until_idle(engine);command(engine,1)
        outputs=engine.generate(input_ids=prompts,sampling_params=dict(temperature=.6,max_new_tokens=32,ignore_eos=True),
            return_logprob=True,logprob_start_len=-1,return_routed_experts=True,routed_experts_start_len=0)
        command(engine,2)
        from full_graph_ab_workload import prepare
        import numpy as np
        model_config=json.loads((Path(config['model_path'])/'config.json').read_text())
        arrays,records=prepare(outputs,prompts,32,model_config)
        np.savez_compressed(args.run/'observed-profile-workload.npz',**arrays)
        (args.run/'observed-profile-workload.json').write_text(json.dumps(records,indent=2)+'\n')
        (args.run/'complete.json').write_text(json.dumps(dict(mode=args.mode,complete=True,steps=32,
            timing_use='Profiler and event-node instrumentation; never use for performance comparison'))+'\n')
    finally:engine.shutdown()


if __name__=='__main__':main()
