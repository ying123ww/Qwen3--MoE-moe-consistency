import base64
import json

import numpy as np
import pytest

import full_graph_ab_experiment as experiment
from full_graph_ab_workload import compare,prepare
from full_graph_ab_control import encode,decode


def communication():
    return json.dumps({name:dict(num_sms=20,num_max_nvl_chunked_send_tokens=9,
        num_max_nvl_chunked_recv_tokens=256,num_max_rdma_chunked_send_tokens=6,
        num_max_rdma_chunked_recv_tokens=128) for name in ('normal_dispatch','normal_combine')})


@pytest.mark.parametrize('action,phase',[('reset','gate'),('flush','gate'),('reset','formal'),('flush','formal')])
def test_experiment_control_command_roundtrip(action,phase):
    assert decode(encode(action,phase))==(action,phase)


@pytest.mark.parametrize('value',[0,99,20.5,'20'])
def test_invalid_control_message_is_rejected(value):
    with pytest.raises(ValueError):decode(value)


@pytest.mark.parametrize('mode',['A','B','C'])
def test_configuration_keeps_communication_as_separate_axis(monkeypatch,mode):
    monkeypatch.setenv('SLIME_QWEN3_FIXED_DEEPEP_CONFIG',communication())
    result=experiment.configure(experiment.base_config(),mode,'/tmp/proof')
    assert result['enable_deterministic_inference']==(mode=='C')
    assert result['cuda_graph_config']['decode']['backend']=='full'
    assert result['cuda_graph_config']['decode']['bs']==[2,4]
    assert result['cuda_graph_config']['prefill']['backend']=='disabled'
    if mode=='A':assert 'deepep_config' not in result
    else:assert json.loads(result['deepep_config'])==json.loads(communication())


def test_fixed_mode_rejects_different_sms_and_preserves_explicit_valid_config(monkeypatch):
    bad=json.loads(communication());bad['normal_dispatch']['num_sms']=24
    with pytest.raises(ValueError,match='20 SM'):
        experiment.configure({'deepep_config':json.dumps(bad)},'B','/tmp/proof')
    monkeypatch.delenv('SLIME_QWEN3_FIXED_DEEPEP_CONFIG',raising=False)
    assert experiment.configure({'deepep_config':communication()},'B','/tmp/proof')['deepep_config']


def fixture(response=(10,11,12),routes=None):
    routes=np.tile(np.asarray([0,1],dtype='<i4'),(4,2,1)) if routes is None else routes
    output=dict(output_ids=list(response),meta_info=dict(completion_tokens=3,
        output_token_logprobs=[(-0.0,int(x)) for x in response],
        routed_experts=base64.b64encode(routes.tobytes()).decode()))
    return output


def test_actual_route_coverage_identity_and_logprob_bits():
    arrays,records=prepare([fixture()],[[2,3]],3,dict(num_hidden_layers=2,num_experts_per_tok=2,num_experts=4))
    assert arrays['tokens_0'].tolist()==[2,3,10,11,12]
    assert arrays['routes_0'].shape==(4,2,2)
    assert arrays['logprob_bits_0'].tolist()==[0x80000000]*3
    assert records[0]['route_shape']==[4,2,2]


@pytest.mark.parametrize('failure',['missing','duplicate','length','identity'])
def test_invalid_observation_cannot_be_reported_as_valid(failure):
    out=fixture()
    if failure=='missing':out['meta_info'].pop('routed_experts')
    if failure=='duplicate':out=fixture(routes=np.zeros((4,2,2),dtype='<i4'))
    if failure=='length':out['meta_info']['routed_experts']=base64.b64encode(np.zeros(2,dtype='<i4').tobytes()).decode()
    if failure=='identity':out['output_ids']=[91,92,93]
    with pytest.raises(ValueError):prepare([out],[[2,3]],3,dict(num_hidden_layers=2,num_experts_per_tok=2,num_experts=4))


def test_route_comparison_ignores_diverged_input_prefix_and_distinguishes_order(tmp_path):
    model=dict(num_hidden_layers=2,num_experts_per_tok=2,num_experts=4)
    a,_=prepare([fixture()],[[2,3]],3,model)
    route=a['routes_0'].copy();route[0,0]=[1,0];route[3,:,:]=[2,3]
    b,_=prepare([fixture(response=(10,19,12),routes=route)],[[2,3]],3,model)
    left=tmp_path/'left.npz';right=tmp_path/'right.npz'
    np.savez(left,**a);np.savez(right,**b)
    result=compare(left,right,[2])
    assert result['response_token_mismatches']==1
    assert result['shared_input_route_rows']==6
    assert result['ordered_route_id_mismatches']==2
    assert result['route_set_row_mismatches']==0
