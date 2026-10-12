import asyncio
from types import SimpleNamespace
from reward_hooks import alternating_reward


def test_each_real_group_has_nonzero_centered_advantage():
    args=SimpleNamespace(n_samples_per_prompt=2)
    rewards=[asyncio.run(alternating_reward(args,SimpleNamespace(index=i))) for i in range(8)]
    assert rewards==[0.,1.]*4
    for i in range(0,8,2):
        a,b=rewards[i:i+2];mean=(a+b)/2
        assert [a-mean,b-mean]==[-.5,.5]
