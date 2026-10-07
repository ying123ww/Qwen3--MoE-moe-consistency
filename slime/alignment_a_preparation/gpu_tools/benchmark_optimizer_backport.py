"""Actual HybridDeviceOptimizer factorial A/B; isolated classes, no dependency edits."""
import argparse,gc,hashlib,importlib.util,json,time,types,os
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pairs',type=int,default=8);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
source=Path('/public/home/yingxy2024/slime-deps/Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py')
adam=Path('/public/home/yingxy2024/moe-consistency/slime/slime/backends/megatron_utils/stateless_adam.py')
s=source.read_text();pin='                    if cpu_copy and self.pin_cpu_params:\n                        param = param.pin_memory()\n'
assert s.count(pin)==1 and s.count('self._h2d_stream.record_event().wait(torch.cuda.current_stream())')==1
spec=importlib.util.spec_from_file_location('bench_stateless',adam);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
variants={}
for name,keep_pin,keep_event in [('old',False,False),('pin_only',True,False),('event_only',False,True),('both',True,True)]:
 code=s if keep_pin else s.replace(pin,'')
 if not keep_event:code=code.replace('self._h2d_stream.record_event().wait(torch.cuda.current_stream())','self._d2h_stream.record_event().wait(torch.cuda.current_stream())')
 m=types.ModuleType('hybrid_'+name);exec(compile(code,str(source), 'exec'),m.__dict__);variants[name]=m.HybridDeviceOptimizer
sizes=[6*1024**2//4,12*1024**2//4]*args.pairs
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
rows=[];hashes={};correct=[]
for rep in range(3):
 order=list(variants) if rep%2==0 else list(reversed(variants))
 for name in order:
  torch.manual_seed(123)
  params=[torch.nn.Parameter(torch.randn(n,device='cuda',dtype=torch.bfloat16)) for n in sizes]
  for x in params:x.decoupled_grad=torch.randn(x.shape,device='cuda',dtype=torch.float32)
  opt=variants[name](params,offload_fraction=1.0,cpu_optimizer_cls=mod.StatelessAdam,gpu_optimizer_cls=mod.StatelessAdam,param_update_in_fp32=True,pin_cpu_params=True,pin_cpu_grads=True,overlap_cpu_optimizer_d2h_h2d=True,lr=1e-3)
  masters=list(opt.cpu_copys_map_gpu_param)
  pinned=[x.is_pinned() for x in masters]
  measurements=[]
  for step in range(14):
   torch.cuda.synchronize();t=time.perf_counter();opt.step();torch.cuda.synchronize();dt=time.perf_counter()-t
   if step>=4:measurements.append(dt)
  # Current-stream consumer checks completion ordering; final global sync checks final values.
  opt.step()
  observed=[x.detach().clone() for x in params]
  torch.cuda.synchronize()
  mismatches=sum(int((v!=opt.gpu_params_map_cpu_copy[x].to(device='cuda',dtype=x.dtype)).sum()) for x,v in zip(params,observed))
  digest=hashlib.sha256()
  for x in params:digest.update(x.detach().cpu().view(torch.uint16).numpy().tobytes())
  h=digest.hexdigest();hashes.setdefault(rep,{})[name]=h
  correct.append({'rep':rep,'variant':name,'consumer_vs_master_mismatches':mismatches,'final_sha256':h})
  if rep==0:
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    with torch.profiler.record_function('P1/optimizer_step'):opt.step();torch.cuda.synchronize()
   prof.export_chrome_trace(str(args.output/(name+'.json')))
  row={'rep':rep,'variant':name,'pinned_count':sum(pinned),'params':len(params),'step_s':measurements,'median_s':float(np.median(measurements))}
  rows.append(row);print(json.dumps(row),flush=True)
  del params,masters,opt,observed;x=None;gc.collect();torch.cuda.empty_cache()
result={'environment':{k:os.environ.get(k) for k in ('CUDA_LAUNCH_BLOCKING','TORCH_USE_CUDA_DSA','OMP_NUM_THREADS','CUDA_VISIBLE_DEVICES')},'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'threads':torch.get_num_threads(),'fp32_parameter_bytes':sum(sizes)*4,'pairs':args.pairs,'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'adam_sha256':hashlib.sha256(adam.read_bytes()).hexdigest(),'scope':'isolated actual optimizer, synthetic gradients, no forward/backward; timings unprofiled, device synchronized; 4 warmup/10 measured per variant/rep','rows':rows,'correctness':correct,'variant_final_bits_equal':all(len(set(v.values()))==1 for v in hashes.values())}
(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['variant_final_bits_equal'] and all(x['consumer_vs_master_mismatches']==0 for x in correct)
