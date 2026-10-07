"""Build explicit identity-bound runtime manifests from already-generated plans."""
import json
from pathlib import Path
import sys
import numpy as np
root=Path('/public/home/yingxy2024/moe-consistency/slime')
sys.path.insert(0,str(root/'tools/routepack'))
from runtime_schedule import make_manifest, apply_schedule
b2=root/'alignment_a_preparation/gpu_runs/autopack-real-hgx05-b1-20260914-02/b2-verified'
plans=Path(__file__).parent
samples=[json.loads(s) for s in (b2/'samples.jsonl').read_text().splitlines()]
samples.sort(key=lambda s:s['prepared_sample_index'])
assert [s['prepared_sample_index'] for s in samples]==list(range(len(samples)))
tokens=[np.load(s['token_path'],allow_pickle=False).tolist() for s in samples]
lengths=[s['token_count'] for s in samples]
rollouts=[s['rollout_id'] for s in samples]
assert all(len(t)==n for t,n in zip(tokens,lengths))
mapping={s['sample_id']:s['prepared_sample_index'] for s in samples}
topology=json.loads((b2/'topology.json').read_text())
rankmap={g:r for r,g in enumerate(topology['dense_dp_rank_to_global'])}
native=json.loads((b2/'schedule.json').read_text())
benchmark=json.loads((root/'alignment_a_preparation/gpu_runs/autopack-real-hgx05-20260914/exchange-seed0.json').read_text())
# Offline adapter smoke reference reconstructed from recorded native B2;
# runtime caller MUST run build_dp_schedule independently before apply_schedule.
partitions=[[] for _ in rankmap];micro=[[] for _ in rankmap]
steps=sorted({c['step'] for c in native['cells']});counts=[]
for step in steps:
 cells=[c for c in native['cells'] if c['step']==step]
 counts.append(len({c['slot'] for c in cells}))
 for c in sorted(cells,key=lambda c:(rankmap[c['source_rank']],c['slot'])):
  r=rankmap[c['source_rank']];ids=[mapping[s] for s in c['sample_ids']];start=len(partitions[r]);partitions[r].extend(ids);micro[r].append(list(range(start,start+len(ids))))
reference=(partitions,micro,counts,[256]*len(counts))
summary={}
for name in ('native','routepack','random'):
 target=plans/name;target.mkdir(exist_ok=True)
 candidate=native if name=='native' else json.loads((target/'schedule.json').read_text())
 manifest=make_manifest(native,candidate,sample_to_global_index=mapping,source_rank_to_dp_rank=rankmap,
  total_lengths=lengths,token_sequences=tokens,rollout_indices=rollouts,pad_multiplier=512,max_tokens_per_bin=20480,
  mode='native' if name=='native' else 'candidate')
 path=target/'runtime.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
 result=apply_schedule(reference,total_lengths=lengths,token_sequences=tokens,rollout_indices=rollouts,
  dp_size=8,max_tokens_per_bin=20480,pad_multiplier=512,path=path)
 if name=='native':
  E=benchmark['excess']['native'];swaps=0;assert result==reference
 else:
  report=json.loads((target/'exchange.json').read_text())
  with np.load(target/'route_loads.npz',allow_pickle=False) as arrays:
   h=arrays['collective_H'];E=float((h.max(axis=-1)-h.mean(axis=-1)).sum())
  swaps=len(report['swaps']);assert swaps==18
 assert E==benchmark['excess'][name]
 summary[name]={'runtime_manifest':str(path),'E':E,'swaps':swaps,'adapter_recorded_native_smoke_passed':True}
summary['note']='Smoke uses recorded native B2; actual runtime independently rebuilds native schedule and verifies exact match.'
(plans/'runtime-validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
