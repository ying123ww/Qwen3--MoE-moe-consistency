"""Post-real-rollout old native replay vs recapture and storage-replacement gate."""
import json
import os
from pathlib import Path
from urllib.request import build_opener,ProxyHandler,Request
from graph_lifecycle_hooks import KEY


def log_rollout(rollout_id,args,samples,extra_metrics,rollout_time):
    import torch
    flat=[x for group in samples for x in group] if samples and isinstance(samples[0],list) else samples
    if len(flat)!=8 or any(x.response_length!=128 or len(x.rollout_log_probs)!=128 for x in flat):raise RuntimeError('P13 requires1024 real rollout target tokens')
    run=Path(os.environ['SLIME_P13_RUN_DIR']);endpoint=json.loads((run/'bootstrap/server-endpoint.json').read_text())
    host=endpoint['host'].strip('[]')
    if host in ('0.0.0.0','::'):host='127.0.0.1'
    if ':' in host:host='['+host+']'
    base=f'http://{host}:{endpoint["port"]}';opener=build_opener(ProxyHandler({}))
    def post(path,payload):
        request=Request(base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
        with opener.open(request,timeout=300) as response:
            body=response.read()
            return json.loads(body) if 'json' in response.headers.get('Content-Type','') else body.decode()
    def control(value):
        result=post('/set_internal_state',dict(server_args={KEY:value}))
        if not isinstance(result,list) or not result or not all(result):raise RuntimeError('P13 control was not accepted by every worker')
        return result
    prompts=[list(map(int,x.tokens[:-x.response_length])) for x in flat]
    def greedy():
        if post('/flush_cache',{}) is False:raise RuntimeError('P13 cache flush rejected')
        outputs=post('/generate',dict(input_ids=prompts,sampling_params=dict(temperature=0,max_new_tokens=16,ignore_eos=True),return_logprob=True,logprob_start_len=-1))
        if not isinstance(outputs,list) or len(outputs)!=8:raise RuntimeError('Incomplete replay probe outputs')
        records=[]
        for out in outputs:
            pairs=out['meta_info']['output_token_logprobs']
            if len(pairs)!=16:raise RuntimeError('Incomplete replay probe target logprobs')
            values=torch.tensor([x[0] for x in pairs],dtype=torch.float32)
            if not torch.isfinite(values).all():raise RuntimeError('Nonfinite replay probe')
            records.append(dict(tokens=[int(x[1]) for x in pairs],bits=values.view(torch.int32).tolist()))
        return records
    old=greedy();control(1)
    control(2);new=greedy();control(1)
    if old!=new:raise RuntimeError('Updated old native graph differs from recaptured reference')
    negative=None
    if int(rollout_id)==2:
        control(3);control(2);negative=greedy();control(1)
        if negative!=new:raise RuntimeError('Storage replacement recovery changed unchanged model values')
    post('/flush_cache',{})
    evidence=run/'replay-validation';evidence.mkdir(exist_ok=True)
    (evidence/f'version-{rollout_id}.json').write_text(json.dumps(dict(rollout_id=rollout_id,old_replay=old,recaptured=new,model_equal=True,replacement_recovered=negative is not None,replacement_reference=negative,scope='Actual updated engine native replay vs recapture;128 greedy probe LP bits. Real online1024 target gate is separate.'),indent=2)+'\n')
    return False
