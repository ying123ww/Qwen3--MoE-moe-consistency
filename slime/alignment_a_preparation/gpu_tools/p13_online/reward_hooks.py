"""Controlled nonzero-GRPO diagnostic reward; not task learning quality."""
async def alternating_reward(args,sample):
    if int(args.n_samples_per_prompt)!=2:raise ValueError('Diagnostic reward requires2 samples per prompt')
    if sample.index is None:raise ValueError('Actual rollout sample index missing')
    return float(int(sample.index)%2)
