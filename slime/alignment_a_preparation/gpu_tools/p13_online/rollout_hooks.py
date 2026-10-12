"""Real natural rollout with a fixed128-target gate; no teacher/synthetic substitution."""
async def generate(args,sample,sampling_params):
    from slime.rollout.sglang_rollout import generate as original
    sampling_params=dict(sampling_params);sampling_params['ignore_eos']=True
    return await original(args,sample,sampling_params)
