import pytest
import torch
from validate_online_bits import compare


def data():
    rollout={'samples':[dict(response_length=128,tokens=[i]*130,rollout_log_probs=[-1.]*128) for i in range(8)]}
    train={'samples':[dict(rollout_position=i,tokens=[i]*130,log_probs=[-1.]*128) for i in range(8)]}
    return rollout,train


def test_complete_gate():
    a,b=data();assert compare(a,b)==dict(target_tokens=1024,bit_mismatches=0,bitwise_exact=True)


def test_positive_negative_zero_are_distinct_bits():
    a,b=data();a['samples'][0]['rollout_log_probs'][0]=0.;b['samples'][0]['log_probs'][0]=-0.
    assert compare(a,b)['bit_mismatches']==1


@pytest.mark.parametrize('kind',['tokens','missing','nonfinite','short'])
def test_invalid_scope_rejected(kind):
    a,b=data()
    if kind=='tokens':b['samples'][0]['tokens'][0]=9
    elif kind=='missing':b['samples'].pop()
    elif kind=='nonfinite':a['samples'][0]['rollout_log_probs'][0]=float('nan')
    else:a['samples'][0]['response_length']=127
    with pytest.raises(RuntimeError):compare(a,b)
