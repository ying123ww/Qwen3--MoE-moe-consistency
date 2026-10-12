from types import SimpleNamespace
import torch
from training_update_hooks import cpu_master_tensors


def test_cpu_master_discovery_nested_deduplicates():
    master=torch.ones(100,dtype=torch.float32)
    half=torch.ones(10,dtype=torch.bfloat16)
    leaf=SimpleNamespace(param_groups=[{'params':[master,half]}])
    hybrid=SimpleNamespace(cpu_optimizers=[leaf],sub_optimizers=[leaf])
    root=SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=hybrid)],param_groups=[{'params':[master]}])
    assert cpu_master_tensors(root)==[master]


class Chained:
    def __init__(self,children):self.chained_optimizers=children
    @property
    def optimizer(self):raise AssertionError('Do not access ambiguous optimizer property')


def test_multi_optimizer_chain_does_not_access_ambiguous_property():
    a=torch.ones(4,dtype=torch.float32);b=torch.ones(5,dtype=torch.float32)
    root=Chained([SimpleNamespace(param_groups=[{'params':[a]}]),SimpleNamespace(param_groups=[{'params':[b]}])])
    values=cpu_master_tensors(root)
    assert {id(x) for x in values}=={id(a),id(b)}
