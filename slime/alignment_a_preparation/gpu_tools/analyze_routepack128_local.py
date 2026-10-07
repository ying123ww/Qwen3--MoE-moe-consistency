import sys,json,statistics
from pathlib import Path
import numpy as np
root=Path('/public/home/yingxy2024/moe-consistency/slime')
sys.path.insert(0,str(root/'alignment_a_preparation/gpu_tools'))
from analyze_layer_attribution import topology_and_layout,analyze
b=root/'alignment_a_preparation/gpu_runs/autopack-gpu-compare128-20260914'
groups,mapping,slots=topology_and_layout(b/'plans/input/topology.json',b/'plans/native/schedule.json')
paths={'native':b/'timing_native-01/timing.json','routepack':b/'runs/timing_routepack-01/timing.json'}
out={}; tables={}; loads={}
for name,p in paths.items():
 report,rows,keys=analyze(p,groups,mapping,slots)
 d=json.loads(p.read_text()); it=next(i for i in d['iterations'] if i[0]['phase']=='attribution')
 perrank=[]
 for r in it:
  sums={stage+'_'+kind:sum(e['elapsed_ms'] for e in r['layer_attribution']['events'] if e['stage']==stage and e['kind']==kind)/1000 for stage in ['replay_forward','replay_backward'] for kind in ['moe','experts']}
  moe=sums['replay_forward_moe']+sums['replay_backward_moe']; expert=sums['replay_forward_experts']+sums['replay_backward_experts']
  perrank.append(dict(rank=r['rank'],step_s=r['instrumented_step_s'],fb_s=r['forward_backward_s'],outside_fb_s=r['instrumented_step_s']-r['forward_backward_s'],moe_s=moe,experts_s=expert,moe_nonexpert_interval_s=moe-expert,fb_outside_observed_moe_s=r['forward_backward_s']-moe,**sums))
 agg={k:statistics.mean(x[k] for x in perrank) for k in perrank[0] if k!='rank'}
 tables[name]={(r['group_id'],r['slot'],r['layer'],r['stage'],r['kind']):r for r in rows}
 loads[name]=np.load(b/'plans'/name/'route_loads.npz')
 h=loads[name]['collective_H'];u=loads[name]['collective_U'];pe=loads[name]['collective_per_expert']
 out[name]={'rank_mean':agg,'per_rank':perrank,'formal_step':report['formal_measured_max_rank_step_s'], 'proxy':dict(E=float((h.max(-1)-h.mean(-1)).sum()),sum_H_max=float(h.max(-1).sum()),sum_H_mean=float(h.mean(-1).sum()),sum_U_max=float(u.max(-1).sum()),U_excess=float((u.max(-1)-u.mean(-1)).sum()),sum_expert_peak=float(pe.max(-1).sum()),total_H=int(h.sum()))}
assert tables['native'].keys()==tables['routepack'].keys()
# Array order is sorted(step, group_id, slot), validated by schedule construction.
collective_keys=sorted({(c['step'],c['group_id'],c['slot']) for c in json.loads((b/'plans/native/schedule.json').read_text())['cells']})
index={(g,s):i for i,(_,g,s) in enumerate(collective_keys)}
correlations={}
for stage in ['replay_forward','replay_backward']:
 for kind in ['moe','experts']:
  rows=[k for k in tables['native'] if k[3:]==(stage,kind)]
  dh=[];dt=[];native_h=[];native_t=[];nmax=[];rmax=[]
  for k in rows:
   g,s,l,_,_=k;i=index[g,s]; n=tables['native'][k];r=tables['routepack'][k]
   hn=loads['native']['collective_H'][i,l];hr=loads['routepack']['collective_H'][i,l]
   dh.extend(hr-hn);dt.extend(np.array(r['rank_ms'])-n['rank_ms']);native_h.extend(hn);native_t.extend(n['rank_ms']);nmax.append(n['max_ms']);rmax.append(r['max_ms'])
  correlations[stage+'_'+kind]=dict(pearson_delta_H_delta_rank_ms=float(np.corrcoef(dh,dt)[0,1]),pearson_native_H_rank_ms=float(np.corrcoef(native_h,native_t)[0,1]),sum_collective_max_native_s=sum(nmax)/1000,sum_collective_max_routepack_s=sum(rmax)/1000,collectives=len(rows))
out['correlations']=correlations
out['limitations']=['One attribution round per method, not independently replicated.','MoE and experts nested; do not add. Observed replay_backward is recompute forward, not all backward kernels.','MoE minus expert includes router/dispatch/combine/permutation/waits/host gaps, not isolated communication.','No aligned cross-rank timestamps: summed maxima are descriptive and are not step time.','Correlations are descriptive and include layer/shape confounding; not causal.','Formal median differs from instrumented single round; do not mix for component decomposition.']
(b/'local-attribution-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for n in paths:print(n,json.dumps({k:v for k,v in out[n].items() if k!='per_rank'}))
print('correlations',json.dumps(correlations))
