import argparse
import asyncio
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'slime/alignment_a_preparation/gpu_tools'))
import full_graph_ab_experiment as experiment
from inference_benchmark import FROZEN,flush_cache_until_idle
from outer_profile_hooks import KEY

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['A','C'],required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    experiment.environment(a.mode);experiment.allow_full_configuration(a.run,a.mode)
    import torch,sglang
    samples=torch.load(FROZEN,map_location='cpu',weights_only=False)['samples'];prompts=[list(map(int,x['tokens'][:-int(x['response_length'])])) for x in samples]
    cfg=experiment.configure(experiment.base_config(),a.mode,a.run/'graph-evidence');cfg['enable_return_routed_experts']=True
    evidence=a.run/'profile-evidence';evidence.mkdir();cfg['forward_hooks'].append(dict(name='p11-outer',target_modules=['model'],before_cuda_graph_capture=True,hook_factory='outer_profile_hooks:install',config=dict(mode=a.mode,output_dir=str(evidence))))
    (a.run/'requested-server-args.json').write_text(json.dumps(cfg,indent=2)+'\n')
    engine=sglang.Engine(**cfg)
    def generate(n):return engine.generate(input_ids=prompts,sampling_params=dict(temperature=.6,max_new_tokens=n,ignore_eos=True),return_logprob=True,logprob_start_len=-1,return_routed_experts=True,routed_experts_start_len=0)
    def command(value):
        from sglang.srt.managers.io_struct import SetInternalStateReq
        result=engine.loop.run_until_complete(asyncio.wait_for(engine.tokenizer_manager.set_internal_state(SetInternalStateReq(server_args={KEY:value})),timeout=120))
        if not result or not all(result):raise RuntimeError('Outer control not acknowledged')
    try:
        for n in (1,128,1):flush_cache_until_idle(engine);generate(n)
        flush_cache_until_idle(engine);command(1);outputs=generate(1);command(2)
        deadline=time.monotonic()+120
        while not all((evidence/f'outer-rank-{rank}.json').exists() for rank in range(8)):
            if time.monotonic()>deadline:raise RuntimeError('All-rank profile reports did not arrive')
            time.sleep(.2)
        from full_graph_ab_workload import prepare
        import numpy as np
        arrays,records=prepare(outputs,prompts,1,json.loads((Path(cfg['model_path'])/'config.json').read_text()));np.savez_compressed(a.run/'workload.npz',**arrays)
        (a.run/'complete.json').write_text(json.dumps(dict(complete=True,mode=a.mode,requests=8,response_tokens=8,records=records,scope='Instrumented prefill+sampling attribution, not a latency benchmark'),indent=2)+'\n')
    finally:engine.shutdown()

if __name__=='__main__':main()
