import json

import pytest
import run_inference_repro as repro
import runtime_diagnostics as diagnostics
import torch


@pytest.mark.parametrize('condition',['inherited','clean'])
@pytest.mark.parametrize('name',['baseline','aligned','candidate'])
def test_environment_factor_does_not_change_other_controls(condition,name):
    original=dict(SLURM_PROCID='0',PMI_RANK='0',PMIX_RANK='0',OMPI_COMM_WORLD_RANK='0',
                  OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                  SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE='1',NCCL_ALGO='Ring')
    result=repro.worker_environment(original,condition,name)
    assert result['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']==('0' if name=='baseline' else '1')
    assert result['OMP_NUM_THREADS']=='1' and result['NCCL_ALGO']=='Ring'
    assert result['CUDA_VISIBLE_DEVICES']==original['CUDA_VISIBLE_DEVICES']
    assert ('SLURM_PROCID' in result)==(condition=='inherited')
    assert ('PMI_RANK' in result)==(condition=='inherited')
    assert ('PMIX_RANK' in result)==(condition=='inherited')
    assert original['SLURM_PROCID']=='0'


def test_diagnostics_only_writes_first_warmup_and_leaves_output_intact(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(diagnostics,'capture',lambda:calls.append(1) or dict(pid=5,torch_threads=1))
    monkeypatch.setattr(diagnostics.torch.distributed,'is_initialized',lambda:False)
    callback=diagnostics.install(dict(output_dir=str(tmp_path)))
    output=object()
    assert callback(object(),(),output) is None
    assert callback(object(),(),output) is None
    assert calls==[1]
    files=list(tmp_path.glob('rank-0-*.json'))
    assert len(files)==1 and json.loads(files[0].read_text())['torch_threads']==1


def test_inner_model_hook_fires_when_eager_runner_calls_root_forward_directly(tmp_path,monkeypatch):
    monkeypatch.setattr(diagnostics,'capture',lambda:dict(torch_threads=1))
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:False)
    class Inner(torch.nn.Module):
        def forward(self,x): return x
    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model=Inner()
        def forward(self,x): return self.model(x)
    model=Outer()
    dict(model.named_modules())['model'].register_forward_hook(diagnostics.install(dict(output_dir=str(tmp_path))))
    sentinel=object()
    assert model.forward(sentinel) is sentinel
    assert len(list(tmp_path.glob('rank-0-*.json')))==1
