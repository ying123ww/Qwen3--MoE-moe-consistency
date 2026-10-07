"""Build fresh128-sample native layout and seed0 candidate plans, CPU only."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
root=Path('/public/home/yingxy2024/moe-consistency/slime')
sys.path[:0]=[str(root),str(root/'tools/routepack')]
from slime.utils.dp_schedule import build_dp_schedule
from exchange import refine
from profile_data import summarize
from runtime_schedule import make_manifest,apply_schedule
out=Path(__file__).parent
argv=json.loads((out/'timing_native-01/argv.json').read_text())
def arg(name):return argv[argv.index('--'+name)+1]
keys=['hidden-size','num-attention-heads','num-query-groups','kv-channels','num-layers','ffn-hidden-size','vocab-size','moe-ffn-hidden-size','moe-router-topk','num-experts','max-tokens-per-gpu']
args=SimpleNamespace(**{k.replace('-','_'):int(arg(k)) for k in keys})
args.use_dynamic_batch_size='--use-dynamic-batch-size' in argv
args.balance_data='--balance-data' in argv
args.balance_by_flops='--balance-by-flops' in argv
args.moe_layer_freq=json.loads(arg('moe-layer-freq'))
args.multi_latent_attention=False
cfg={'dp_size':8,'cp_size':1,'vpp_size':1,'microbatch_group_size_per_vp_stage':1}
assert int(arg('global-batch-size'))==128 and args.max_tokens_per_gpu==10240
b2=root/'alignment_a_preparation/gpu_runs/autopack-real-hgx05-b1-20260914-02/b2-verified'
samples=[json.loads(s) for s in (b2/'samples.jsonl').read_text().splitlines()]
samples=sorted(samples,key=lambda s:s['prepared_sample_index'])[:128]
assert [s['prepared_sample_index'] for s in samples]==list(range(128))
tokens=[np.load(s['token_path'],allow_pickle=False).tolist() for s in samples]
lengths=list(map(len,tokens));rollouts=[s['rollout_id'] for s in samples]
actual=build_dp_schedule(args,cfg,lengths,global_batch_size=128,rollout_indices=rollouts)
partitions,micro,counts,batches=actual
topology=json.loads((b2/'topology.json').read_text())
rankmap={g:r for r,g in enumerate(topology['dense_dp_rank_to_global'])}
groupmap={r:g['group_id'] for g in topology['ep_groups'] for r in g['members']}
cells=[]
for rank in range(8):
 offset=0
 for step,count in enumerate(counts):
  for slot in range(count):
   ids=[samples[partitions[rank][i]]['sample_id'] for i in micro[rank][offset+slot]]
   globalrank=topology['dense_dp_rank_to_global'][rank]
   cells.append(dict(step=step,group_id=groupmap[globalrank],source_rank=globalrank,slot=slot,sample_ids=ids))
  offset+=count
native={'source':'fresh build_dp_schedule on exact 128 lengths and rollout IDs','cells':cells}
plans=out/'plans';plans.mkdir(exist_ok=True)
inputs=plans/'input';inputs.mkdir(exist_ok=True)
(inputs/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
(inputs/'schedule.json').write_text(json.dumps(native,indent=2)+'\n')
(inputs/'topology.json').write_text(json.dumps(topology)+'\n')
traces={s['sample_id']:np.load(s['route_path'],allow_pickle=False,mmap_mode='r') for s in samples}
metadata,records,arrays=summarize(samples,traces,native,topology,512)
np.savez_compressed(inputs/'route_loads.npz',**arrays)
(inputs/'collectives.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
mapping={s['sample_id']:s['prepared_sample_index'] for s in samples}
summary={'args':vars(args),'parallel_config':cfg,'num_microbatches':counts,'global_batch_sizes':batches,
         'sample_count':128,'total_tokens':sum(lengths),'bins':len(cells),'plans':{}}
swaps=20
for name in ('native','routepack','random'):
 target=plans/name;target.mkdir(exist_ok=True)
 if name=='native':schedule=native;report={'swaps':[]};result_arrays=arrays
 else:
  schedule,report,result_arrays=refine(samples,traces,native,topology,512,max_candidates=500,max_swaps=swaps,
        max_bin_tokens=None,mode='sampled_best' if name=='routepack' else 'random',seed=0,max_iterations=20,candidate_batch_size=25)
  if name=='routepack':swaps=len(report['swaps'])
  else:assert len(report['swaps'])==swaps
 (target/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
 (target/'exchange.json').write_text(json.dumps(report,indent=2)+'\n')
 np.savez_compressed(target/'route_loads.npz',**result_arrays)
 manifest=make_manifest(native,schedule,sample_to_global_index=mapping,source_rank_to_dp_rank=rankmap,
   total_lengths=lengths,token_sequences=tokens,rollout_indices=rollouts,pad_multiplier=512,max_tokens_per_bin=10240,
   mode='native' if name=='native' else 'candidate')
 path=target/'runtime.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
 result=apply_schedule(actual,total_lengths=lengths,token_sequences=tokens,rollout_indices=rollouts,dp_size=8,
                       max_tokens_per_bin=10240,pad_multiplier=512,path=path)
 h=result_arrays['collective_H'];E=float((h.max(axis=-1)-h.mean(axis=-1)).sum())
 summary['plans'][name]={'E':E,'swaps':len(report['swaps']),'runtime':str(path),'fresh_native_adapter_passed':True}
 print(json.dumps(summary['plans'][name]),flush=True)
(out/'plans-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
