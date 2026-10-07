import json
from pathlib import Path
b=Path('/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/gpu_runs')
s=json.loads((b/'repeats-20260910-state.json').read_text())
if s['active']:
 p=b/s['active'];f=p/'measurements-rank0.jsonl'
 if f.exists():
  rows=[json.loads(x) for x in f.read_text().splitlines()];s['last_rollout']=rows[-1]['rollout_id'];s['last_step_s']=rows[-1]['elapsed_s'];s['formal_steps']=max(0,len(rows)-5)
 st=p/'job-status.json'
 if st.exists():s['ray_status']=json.loads(st.read_text())['status']
print(json.dumps(s))
