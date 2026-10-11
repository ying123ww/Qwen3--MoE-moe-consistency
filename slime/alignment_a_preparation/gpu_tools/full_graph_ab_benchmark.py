"""P10 route return is timed; observation validation/files are outside generate."""
import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import full_graph_ab_experiment as experiment
from inference_benchmark import FROZEN,flush_cache_until_idle
from full_graph_ab_workload import prepare
from full_graph_ab_control import control


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=experiment.MODES,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();experiment.environment(args.mode)
    experiment.allow_full_configuration(args.run,args.mode)
    import torch
    import sglang
    samples=torch.load(FROZEN,map_location='cpu',weights_only=False)['samples']
    prompts=[list(map(int,s['tokens'][:-int(s['response_length'])])) for s in samples]
    assert len(prompts)==8 and sum(map(len,prompts))==1218
    config=experiment.configure(experiment.base_config(),args.mode,args.run/'graph-evidence')
    config['enable_return_routed_experts']=True
    model_config=json.loads((Path(config['model_path'])/'config.json').read_text())
    sampling=dict(temperature=.6,max_new_tokens=128,ignore_eos=True)
    manifest=dict(mode=args.mode,requested_server_args=config,sampling=sampling,
        source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest(),prompt_lengths=list(map(len,prompts)),
        route_observation=True,workload_save_inside_timing=False,
        scope='Blocking Engine.generate including actual route return; excludes load/capture/warmup/cache flush/NPZ writes/RPC audit',
        environment={k:v for k,v in __import__('os').environ.items() if k.startswith(('SGLANG_','SLIME_QWEN3','NCCL_'))})
    (args.run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.perf_counter();engine=sglang.Engine(**config)
    try:
        (args.run/'resolved-server-args.json').write_text(json.dumps(dataclasses.asdict(engine.server_args),indent=2,default=str)+'\n')
        flush_cache_until_idle(engine)
        engine.generate(input_ids=prompts[:3],sampling_params=dict(temperature=0,max_new_tokens=2,ignore_eos=True))
        (args.run/'startup.json').write_text(json.dumps(dict(startup_seconds=time.perf_counter()-started))+'\n')
        timings=[]
        for batch in range(23):
            flush_retries=flush_cache_until_idle(engine)
            if batch==3:control(engine,'reset','formal')
            started=time.perf_counter()
            outputs=engine.generate(input_ids=prompts,sampling_params=sampling,return_logprob=True,
                logprob_start_len=-1,return_routed_experts=True,routed_experts_start_len=0)
            elapsed=time.perf_counter()-started
            arrays,records=prepare(outputs,prompts,128,model_config)
            import numpy as np
            artifact=f'workload-{batch:02d}.npz'
            np.savez_compressed(args.run/artifact,**arrays)
            row=dict(batch=batch,warmup=batch<3,elapsed_s=elapsed,output_tokens=1024,
                output_tokens_per_s=1024/elapsed,cache_flush_retries=flush_retries,
                cached_tokens=[x['meta_info'].get('cached_tokens') for x in outputs],
                artifact=artifact,samples=records,completed_wall_time=time.time())
            with (args.run/'measurements.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if batch>=3:timings.append(elapsed)
            print(json.dumps(dict(mode=args.mode,batch=batch,warmup=batch<3,elapsed_s=elapsed)),flush=True)
        control(engine,'flush','formal')
        ordered=sorted(timings);position=(len(ordered)-1)*.95
        p95=ordered[int(position)]+(ordered[math.ceil(position)]-ordered[int(position)])*(position-int(position))
        median=statistics.median(timings)
        summary=dict(mode=args.mode,median_s=median,p95_s=p95,output_tokens_per_s=1024/median,
            formal_batches=20,warmup_batches=3,input_tokens=1218,output_tokens=1024,route_observation=True,
            scope=manifest['scope'])
        (args.run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    finally:engine.shutdown()


if __name__=='__main__':main()
