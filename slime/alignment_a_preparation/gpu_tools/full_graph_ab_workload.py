"""Untimed canonical token/route records and comparison of actually generated workloads."""
import base64
import hashlib

import numpy as np


def prepare(outputs,prompts,response_tokens,model_config):
    if len(outputs)!=len(prompts) or not outputs:raise ValueError('Output identity coverage mismatch')
    layers=int(model_config['num_hidden_layers']);topk=int(model_config['num_experts_per_tok'])
    experts=int(model_config['num_experts']);arrays={};records=[]
    for index,(out,prompt) in enumerate(zip(outputs,prompts)):
        meta=out['meta_info'];raw=meta['output_token_logprobs']
        if meta['completion_tokens']!=response_tokens or len(raw)!=response_tokens:
            raise ValueError('Output length mismatch')
        probabilities=np.asarray([x[0] for x in raw],dtype='<f4')
        if not np.isfinite(probabilities).all():raise ValueError('Nonfinite output logprob')
        response=np.asarray([int(x[1]) for x in raw],dtype='<i8')
        tokens=np.asarray([*map(int,prompt),*response],dtype='<i8')
        if out.get('output_ids') is not None and list(out['output_ids'])!=response.tolist():
            raise ValueError('Returned token identities disagree with logprob identities')
        encoded=meta.get('routed_experts')
        if not isinstance(encoded,str):raise ValueError('Missing actual routed expert trace')
        route=np.frombuffer(base64.b64decode(encoded,validate=True),dtype='<i4')
        expected=(len(tokens)-1,layers,topk)
        if route.size!=int(np.prod(expected)):raise ValueError('Route coverage mismatch')
        route=route.reshape(expected)
        if (route<0).any() or (route>=experts).any():raise ValueError('Route expert outside logical range')
        if (np.diff(np.sort(route,axis=-1),axis=-1)==0).any():raise ValueError('Duplicate top-k expert')
        arrays.update({f'tokens_{index}':tokens,f'routes_{index}':route,f'logprob_bits_{index}':probabilities.view('<u4')})
        records.append(dict(sample=index,prompt_tokens=len(prompt),response_ids=response.tolist(),
            token_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),
            response_sha256=hashlib.sha256(response.tobytes()).hexdigest(),
            route_sha256=hashlib.sha256(route.tobytes()).hexdigest(),route_shape=list(expected),
            route_bytes=route.nbytes))
    return arrays,records


def compare(left,right,prompt_lengths):
    result=dict(response_tokens=0,response_token_mismatches=0,identical_responses=0,
        identical_route_arrays=0,shared_input_route_rows=0,ordered_route_id_mismatches=0,
        route_set_row_mismatches=0)
    with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
        for index,prompt_length in enumerate(prompt_lengths):
            x,y=a[f'tokens_{index}'],b[f'tokens_{index}']
            if x.shape!=y.shape or not np.array_equal(x[:prompt_length],y[:prompt_length]):
                raise ValueError('Compared prompt identities or response lengths changed')
            result['response_tokens']+=len(x)-prompt_length
            result['response_token_mismatches']+=int(np.count_nonzero(x[prompt_length:]!=y[prompt_length:]))
            result['identical_responses']+=int(np.array_equal(x[prompt_length:],y[prompt_length:]))
            rx,ry=a[f'routes_{index}'],b[f'routes_{index}']
            if rx.shape!=ry.shape:raise ValueError('Route observation coverage differs')
            result['identical_route_arrays']+=int(np.array_equal(rx,ry))
            divergence=np.flatnonzero(x!=y)
            shared=min(int(divergence[0]) if len(divergence) else len(x),rx.shape[0])
            # Compare routes only where the full preceding input prefix is equal.
            result['shared_input_route_rows']+=shared*rx.shape[1]
            result['ordered_route_id_mismatches']+=int(np.count_nonzero(rx[:shared]!=ry[:shared]))
            result['route_set_row_mismatches']+=int(np.count_nonzero(
                np.any(np.sort(rx[:shared],axis=-1)!=np.sort(ry[:shared],axis=-1),axis=-1)))
    return result
