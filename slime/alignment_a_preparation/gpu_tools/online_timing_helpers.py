"""Small workload audit; retain normal slime metrics and reward behavior."""
import hashlib
import json
import math
import os
from pathlib import Path


def log_workload(rollout_id, args, samples, extra_metrics, rollout_time):
    flat = [s for group in samples for s in group] if samples and isinstance(samples[0], list) else samples
    assert len(flat) == 8
    response_tokens = [int(s.response_length) for s in flat]
    input_tokens = [len(s.tokens) for s in flat]
    for s in flat:
        assert len(s.rollout_log_probs) == s.response_length
        assert all(math.isfinite(float(v)) for v in s.rollout_log_probs)
    prompts = [list(map(int, s.tokens[:-s.response_length])) if s.response_length else list(map(int,s.tokens)) for s in flat]
    row = dict(rollout_id=rollout_id, samples=len(flat), response_tokens=sum(response_tokens),
               training_input_tokens=sum(input_tokens), response_lengths=response_tokens,
               prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
               rollout_actor_reported_s=rollout_time)
    with (Path(os.environ['ALIGN_A_RUN_DIR'])/'workload.jsonl').open('a') as handle:
        handle.write(json.dumps(row)+'\n')
    return False
