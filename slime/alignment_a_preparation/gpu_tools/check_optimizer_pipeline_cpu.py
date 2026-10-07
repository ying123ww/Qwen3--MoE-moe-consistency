"""Read existing traces on CPU; test whether H2D survives into the next CPU update."""
import gzip,json,hashlib
from bisect import bisect_right
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=[]
for source in json.loads((root/'gpu_runs/p1-optimizer-exclusive-20260912.json').read_text())['results']:
 p=Path(source['path']);es=json.load(gzip.open(p))['traceEvents']
 scopes=sorted([e for e in es if e.get('name')=='Optimizer.step#StatelessAdam.step'],key=lambda e:e['ts']);starts=[e['ts'] for e in scopes]
 apis={e.get('args',{}).get('correlation'):e for e in es if e.get('cat')=='cuda_runtime'}
 count=ended=before_next=eligible=0;tails=[]
 for e in es:
  if e.get('cat')!='gpu_memcpy' or 'HtoD' not in e['name']:continue
  a=apis.get(e.get('args',{}).get('correlation'))
  if not a:continue
  i=bisect_right(starts,a['ts'])-1
  if i<0 or a['ts']+a['dur']>scopes[i]['ts']+scopes[i]['dur']+.01:continue
  count+=1;end=e['ts']+e['dur'];ended+=end<=a['ts']+a['dur'];tails.append(max(0,end-a['ts']-a['dur']))
  if i+1<len(scopes):eligible+=1;before_next+=end<=starts[i+1]
 rows.append({'trace':str(p),'sha256':source['sha256'],'h2d_count':count,'gpu_finished_before_api_return':ended,'pairs_with_next_suboptimizer':eligible,'gpu_finished_before_next_suboptimizer':before_next,'max_gpu_tail_after_api_us':max(tails)})
out=root/'gpu_runs/p1-optimizer-pipeline-cpu-20260912.json';out.write_text(json.dumps({'method':'GPU memcpy correlation -> launching CUDA API -> enclosing StatelessAdam scope -> next scope. CPU parsing only; historical traces.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':rows},indent=2)+'\n');print(out);print(json.dumps(rows,indent=2))
