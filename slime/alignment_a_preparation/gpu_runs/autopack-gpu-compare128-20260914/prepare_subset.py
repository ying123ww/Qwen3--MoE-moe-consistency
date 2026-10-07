"""Select complete first 16 prompt groups (128 samples), without shortening."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
source=Path('/tmp/ap-real-capture-20260914-02/capture/rollout.pt')
capture=source.parent/'captured'
output=Path('/tmp/ap-real-capture128-20260914/rollout.pt')
metadata_output=Path('/public/home/yingxy2024/moe-consistency/slime/alignment_a_preparation/gpu_runs/autopack-gpu-compare128-20260914/subset.json')
def filehash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
payload=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
assert isinstance(payload,dict) and set(payload)=={'samples'}
samples=payload['samples'];assert len(samples)==256
rows=[json.loads(s) for s in (capture/'samples_captured.jsonl').read_text().splitlines()]
assert len(rows)==256
selected=samples[:128];identity=[]
for i,(sample,row) in enumerate(zip(selected,rows[:128])):
 assert sample['index']==i and sample['rollout_id']==i
 fixture=row['source_fixture_record']
 assert fixture['group_index']==i//8 and fixture['repeat_index']==i%8
 tokens=np.asarray(sample['tokens'],dtype='<i8')
 assert hashlib.sha256(tokens.tobytes()).hexdigest()==row['token_hash']
 assert len(tokens)==row['token_count'] and sample['response_length']==row['response_length']
 assert sample['rollout_routed_experts'].shape==(len(tokens)-1,48,8)
 identity.append({'index':i,'source_row_1based':fixture['source_row_1based'],'prompt_group':i//8,
  'repeat':i%8,'token_count':len(tokens),'response_length':sample['response_length'],'token_hash':row['token_hash'],
  'trace_hash':row['trace_hash'],'finish_reason':row.get('finish_reason')})
assert len({x['source_row_1based'] for x in identity})==16
output.parent.mkdir(parents=True,exist_ok=True)
with output.open('xb') as f:torch.save({'samples':selected},f)
checked=torch.load(output,map_location='cpu',weights_only=True,mmap=True)
assert len(checked['samples'])==128
for a,b in zip(selected,checked['samples']):
 assert a.keys()==b.keys()
 for key in a:
  assert torch.equal(a[key],b[key]) if isinstance(a[key],torch.Tensor) else a[key]==b[key]
result={'source':str(source),'source_sha256':filehash(source),'output':str(output),'output_sha256':filehash(output),
 'source_samples':256,'samples':128,'selection':'first 128 complete samples = first 16 capture02 prompt groups x8, original order; no token/response/route truncation',
 'planned_config':{'global_batch_size':128,'rollout_batch_size':16,'n_samples_per_prompt':8,'max_tokens_per_gpu':10240},
 'tokens_total':sum(x['token_count'] for x in identity),'response_tokens_total':sum(x['response_length'] for x in identity),
 'min_total_length':min(x['token_count'] for x in identity),'max_total_length':max(x['token_count'] for x in identity),
 'roundtrip_all_fields_equal':True,'source_preserved':True,'sample_identity':identity}
metadata_output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sample_identity'},indent=2))
