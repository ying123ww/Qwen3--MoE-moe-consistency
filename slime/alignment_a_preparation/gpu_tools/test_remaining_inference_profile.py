"""Scope identity/exception safety and asynchronous GPU attribution fixtures."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import profiling_annotations as annotations
from summarize_remaining_inference import summarize
import run_remaining_inference_profile as runner


@pytest.mark.parametrize('name,value',[('baseline','0'),('candidate','1')])
def test_worker_attention_mode_overrides_inherited_template(name,value):
    template=dict(SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE='1',sentinel='retained')
    result=runner.worker_environment(template,name)
    assert result['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']==value
    assert result['sentinel']=='retained' and template['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']=='1'


def test_all_four_launches_receive_the_right_attention_mode(tmp_path,monkeypatch):
    prep=tmp_path/'slime/alignment_a_preparation'
    (prep/'gpu_runs').mkdir(parents=True)
    (tmp_path/'.a-tmp').mkdir()
    prior=tmp_path/'prior'
    (prior/'exact-off').mkdir(parents=True)
    (prior/'exact-comparison.json').write_text(json.dumps(dict(model_exact=True,batch_exact=True)))
    frozen=tmp_path/'rollout.pt'
    frozen.write_bytes(b'frozen')
    (prior/'exact-off/manifest.json').write_text(json.dumps(dict(source_sha256={},
        source_rollout_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())))
    (prior/'runtime-versions.json').write_text(json.dumps(dict(torch='v',triton='v')))
    expected=tmp_path/'expected.json'
    expected.write_text('{}')
    monkeypatch.setattr(runner,'PRIOR',prior)
    monkeypatch.setattr(runner.guard,'PREP',prep)
    monkeypatch.setattr(runner.guard,'PROJECT',tmp_path)
    monkeypatch.setattr(runner.guard,'FROZEN',frozen)
    monkeypatch.setattr(runner.importlib.metadata,'version',lambda name:'v')
    monkeypatch.setattr(runner.signal,'signal',lambda *args:None)
    monkeypatch.setattr(runner,'snapshot',lambda *args:dict(cgroups=[dict(path=str(tmp_path/'memory/job_7'))]))
    monkeypatch.setattr(runner,'validate',lambda *args:None)
    monkeypatch.setattr(runner.guard,'environment',lambda *args:dict(SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE='1'))
    monkeypatch.setattr(runner.guard,'p2p_preflight',lambda *args:None)
    monkeypatch.setattr(runner.guard,'stop_worker',lambda:None)
    calls=[]
    def command(run,label,argv,env,memory):
        name=label.split('-')[-1]
        assert env['SGLANG_USE_UNIFIED_ATTENTION_FOR_DECODE']==('0' if name=='baseline' else '1')
        assert argv[argv.index('--variant')+1]==('baseline' if name=='baseline' else 'aligned')
        calls.append(label)
        output=Path(argv[argv.index('--run')+1])
        (output/'resolved-server-args.json').write_text(json.dumps(dict(cuda_graph_config={
            phase:dict(backend='disabled') for phase in ('prefill','decode')})))
    monkeypatch.setattr(runner.guard,'command',command)
    monkeypatch.setattr(runner,'collect',lambda path:[])
    monkeypatch.setattr(runner,'report',lambda *args:None)
    monkeypatch.setattr(sys,'argv',['runner','--run-id','remaining-profile-test','--expected-sources',str(expected)])
    runner.main()
    assert calls==['sanity-baseline','sanity-candidate','profile-baseline','profile-candidate']


def test_failure_report_cannot_look_like_success(tmp_path,monkeypatch):
    prep=tmp_path/'prep'
    prep.mkdir()
    run=tmp_path/'remaining-profile-failed'
    run.mkdir()
    monkeypatch.setattr(runner.guard,'PREP',prep)
    runner.failure_report(run,'sanity-baseline initialization failed')
    text=(prep/'P1_REMAINING_INFERENCE_PROFILE_20261009.md').read_text()
    assert '采样失败' in text and 'initialization failed' in text
    assert (run/'FAILURE.md').exists() and not (run/'REPORT.md').exists()


def test_scope_preserves_inputs_output_and_exception(monkeypatch):
    seen=[]
    @contextmanager
    def scope(name):
        seen.append(('enter',name))
        try: yield
        finally: seen.append(('exit',name))
    monkeypatch.setattr(annotations,'record_function',scope)
    x, residual, output=object(),object(),object()
    module=SimpleNamespace(layer_id=39)
    def original(self,a,b,*,flag):
        assert self is module and a is x and b is residual and flag==7
        return output
    assert annotations.annotate(original,'rms')(module,x,residual,flag=7) is output
    assert 'residual_rms' in seen[0][1] and seen[0][1]==seen[1][1]
    def failing(*args): raise ValueError('original failure')
    with pytest.raises(ValueError,match='original failure'):
        annotations.annotate(failing,'router_select')(module,x)
    assert seen[-1][0]=='exit'


def event(name,category,start,duration,**args):
    return dict(name=name,cat=category,ph='X',ts=start,dur=duration,pid=1,tid=2,args=args)


def test_correlates_async_kernel_to_launch_scope_and_mm_shape():
    rows=[event('step[DECODE]','user_annotation',0,20),
          event('P1/moe/layer=0','user_annotation',0,20),
          event('P1/router_logits/layer=0','user_annotation',2,8),
          event('aten::mm','cpu_op',3,6,**{'External id':7,'Input Dims':[[2,2048],[2048,128]]}),
          event('cudaLaunchKernel','cuda_runtime',5,1,correlation=9,**{'External id':7}),
          event('matmul_kernel_persistent','kernel',30,4,correlation=9,device=0,**{'External id':7})]
    result=summarize(rows)
    assert result['gpu_families']['router_matmul']['count']==1
    assert result['gpu_phase_families_s']['decode:router_matmul']==4e-6
    assert result['cpu_scope_exclusive_s']['router_logits']==8e-6
    assert result['cpu_scope_exclusive_s']['moe']==12e-6
    assert result['unlinked_persistent_matmul']==0


def test_missing_launch_does_not_guess_scope_and_gaps_are_not_cpu_time():
    rows=[event('first','kernel',0,2,device=0),
          event('P1/router_select/layer=0','user_annotation',5,3),
          event('cudaLaunchKernel','cuda_runtime',6,1,correlation=4),
          event('softmax','kernel',10,2,correlation=4,device=0),
          event('last','kernel',15,1,device=0)]
    result=summarize(rows)
    assert result['gpu_families']['router_select']['count']==1
    assert result['gpu_families']['unscoped_gpu']['count']==2
    assert result['gpu_union_s']==5e-6
    assert result['gpu_gap_s']==dict(before_next_launch=4e-6,after_launch_started=4e-6,unlinked=3e-6)


def test_unlinked_persistent_kernel_is_explicit():
    result=summarize([event('matmul_kernel_persistent','kernel',0,1,device=0)])
    assert result['unlinked_persistent_matmul']==1
