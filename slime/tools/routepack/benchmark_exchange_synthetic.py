"""Reproducible synthetic scale probe, explicitly not representative training data."""
import json
from pathlib import Path
import numpy as np
from profile_data import merge_topology
from benchmark_exchange import compare


def fixture(seed=19):
    rng=np.random.default_rng(seed)
    rows=[{'global_rank':r,'world_size':4,'vpp_size':1,
           'groups':{'dense_dp':list(range(4)),'ep':[r//2*2,r//2*2+1],
                     'expert_dp':list(range(r%2,4,2)),**{n:[r] for n in ('tp','expert_tp','cp','pp')}},
           'experts':{str(l):{'num_experts':4,'local_expert_indices':[r%2*2,r%2*2+1]} for l in range(48)}} for r in range(4)]
    samples=[];traces={};cells=[]
    for slot in range(8):
        for rank in range(4):
            ids=[]
            for j in range(4):
                sid=str(len(samples));ids.append(sid)
                samples.append({'sample_id':sid,'step':0,'rollout_id':int(sid),'reward_group_id':int(sid),
                                'model_version':'synthetic','expert_layout_version':'synthetic','token_count':17,
                                'response_length':1,'valid_loss_token_count':1,'layer_ids':list(range(48))})
                receiver=(rng.random((16,48)) < (0.85 if slot%2 else 0.15)).astype(np.int64)
                traces[sid]=np.stack((receiver*2,receiver*2+1),axis=-1)
            cells.append({'step':0,'group_id':rank//2,'slot':slot,'source_rank':rank,'sample_ids':ids})
    return samples,traces,{'cells':cells},merge_topology(rows)


if __name__=='__main__':
    result=compare(*fixture(),32,repeats=20,seed=0,max_candidates=500,max_iterations=20,max_swaps=20)
    result['scope']='synthetic 128 samples / 32 bins / 48 layers / EP2, not real workload evidence'
    target=Path(__file__).resolve().parents[2]/'alignment_a_preparation/routepack_b_20260912/sampled_best_cpu_benchmark.json'
    with target.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('excess','same_swap_count','cpu_wall_ms','routepack_swaps')}))
