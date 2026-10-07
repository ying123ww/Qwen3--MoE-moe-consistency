"""Restore original captured arrays and reconstruct original first128 fixture; no GPU."""
import hashlib,json,tarfile,shutil
from pathlib import Path
import numpy as np
import torch
root=Path('/public/home/yingxy2024/moe-consistency/slime')
archive=root/'alignment_a_preparation/gpu_runs/autopack-real-hgx05-20260914/capture02-evidence.tar.gz'
capture=Path('/tmp/ap-real-capture-20260914-02/capture/captured')
output=Path('/tmp/ap-real-capture128-20260914/rollout.pt')
with tarfile.open(archive,'r:gz') as tf:
 for m in tf:
  if not m.isfile():continue
  relative=Path(m.name)
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe archive member')
  if 'batch-0000' not in relative.parts and relative.name!='samples_captured.jsonl':continue
  target=Path('/tmp')/relative
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():continue
  with tf.extractfile(m) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
rows=[json.loads(x) for x in (capture/'samples_captured.jsonl').read_text().splitlines()][:128]
old=json.loads((root/'alignment_a_preparation/gpu_runs/autopack-gpu-compare128-20260914/subset.json').read_text())
assert len(rows)==128
samples=[]
for i,row in enumerate(rows):
 tokens=np.load(capture/row['token_path'],allow_pickle=False)
 routes=np.load(capture/row['route_path'],allow_pickle=False)
 probs=np.load(capture/row['response_logprob_path'],allow_pickle=False)
 bits=np.load(capture/row['response_logprob_bits_path'],allow_pickle=False)
 assert row['token_hash']==old['sample_identity'][i]['token_hash']==hashlib.sha256(tokens.astype('<i8').tobytes()).hexdigest()
 assert row['trace_hash']==old['sample_identity'][i]['trace_hash']==hashlib.sha256(routes.astype('<i8').tobytes()).hexdigest()
 assert np.array_equal(probs.astype('<f4').view('<u4'),bits)
 assert routes.shape==(len(tokens)-1,48,8) and len(probs)==row['response_length'] and np.isfinite(probs).all()
 samples.append({'index':i,'group_index':row['reward_group_id'] if row.get('reward_group_id') is not None else i//2,'rollout_id':i,
 'tokens':tokens.tolist(),'response_length':row['response_length'],'rollout_log_probs':probs.astype(np.float32).tolist(),
 'rollout_routed_experts':torch.from_numpy(routes.copy()),'loss_mask':[1]*row['response_length'],'reward':0.0,'status':'completed',
 'metadata':{'routepack_sample_id':row['sample_id'],'capture':str(capture),'reward_note':'placeholder unused by forward-only gate'}})
manifest=json.loads((root/'alignment_a_preparation/gpu_runs/autopack-gpu-compare128-20260914/plans/routepack/runtime.json').read_text())
identity=manifest['input_identity']
assert identity['lengths']==[len(s['tokens']) for s in samples]
assert identity['rollout_indices']==[s['rollout_id'] for s in samples]
assert identity['tokens_sha256']==[hashlib.sha256(json.dumps(s['tokens'],separators=(',',':'),ensure_ascii=True).encode()).hexdigest() for s in samples]
output.parent.mkdir(exist_ok=True)
with output.open('xb') as f:torch.save({'samples':samples},f)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
result={'path':str(output),'samples':len(samples),'source_archive':str(archive),'source_archive_sha256':digest(archive),
 'reconstructed_file_sha256':digest(output),'original_subset_sha256':old['output_sha256'],
 'original_token_and_route_hashes_all_match':True,'original_logprob_bits_all_match':True,'runtime_input_identity_exact':True,
 'scope':'Original captured arrays recovered, no new generation; serialization container hash may differ'}
(Path(__file__).parent/'restored.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
