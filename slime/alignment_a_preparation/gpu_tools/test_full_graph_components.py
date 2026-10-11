import pytest

from summarize_full_graph_components import summarize


def record():
    return dict(rank=0,mode='C',last_size=4,steps=32,components=[
        dict(component='model_forward',last_replay_gpu_ms=10.,layer=-1,depth=0,path='model'),
        dict(component='attention',last_replay_gpu_ms=6.,layer=0,depth=1,path='layers.0.attention'),
        dict(component='qkv',last_replay_gpu_ms=2.,layer=0,depth=2,path='layers.0.attention.qkv')])


def test_nested_intervals_remain_separate_and_keep_provenance():
    result=summarize(record())
    assert result['components']['model_forward']['inclusive_sum_ms']==10.
    assert result['components']['attention']['inclusive_sum_ms']==6.
    assert result['components']['qkv']['inclusive_sum_ms']==2.
    assert result['components']['qkv']['layers']==[0]
    assert result['components']['qkv']['depths']==[2]
    assert result['components']['qkv']['paths']==['layers.0.attention.qkv']


@pytest.mark.parametrize('kind',['missing','duplicated','incomplete','negative','nan'])
def test_untrustworthy_profile_records_fail(kind):
    value=record()
    if kind=='missing':value['components'].pop(0)
    if kind=='duplicated':value['components'].append(dict(value['components'][0]))
    if kind=='incomplete':value['steps']=31
    if kind=='negative':value['components'][1]['last_replay_gpu_ms']=-1
    if kind=='nan':value['components'][1]['last_replay_gpu_ms']=float('nan')
    with pytest.raises(ValueError):summarize(value)
