"""Decode bridge buffers must retain addresses across captured shapes."""
import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import BreakableCudaGraphBackend


def test_logits_buffer_prefixes_share_storage_and_preserve_optional_hidden_states():
    backend=object.__new__(BreakableCudaGraphBackend)
    full=LogitsProcessorOutput(next_token_logits=torch.arange(28).reshape(4,7).float(),
                              hidden_states=torch.arange(12).reshape(4,3).float())
    buffer=backend._alloc_full_buffer(full,4)
    backend._copy_output_to_buffer(full,buffer,4)
    small=LogitsProcessorOutput(next_token_logits=torch.full((2,7),17.),hidden_states=torch.full((2,3),19.))
    backend._copy_output_to_buffer(small,buffer,2)
    prefix=backend._slice_output(buffer,2)
    assert backend._output_rows(small,4)==2
    assert prefix.next_token_logits.data_ptr()==buffer.next_token_logits.data_ptr()
    assert prefix.hidden_states.data_ptr()==buffer.hidden_states.data_ptr()
    assert torch.equal(prefix.next_token_logits,small.next_token_logits)
    assert torch.equal(prefix.hidden_states,small.hidden_states)
    assert torch.equal(buffer.next_token_logits[2:],full.next_token_logits[2:])
    assert prefix.full_logits is None


def test_logits_capture_rejects_metadata_it_cannot_replay():
    backend=object.__new__(BreakableCudaGraphBackend)
    with pytest.raises(ValueError,match='non-decode'):
        backend._alloc_full_buffer(LogitsProcessorOutput(torch.ones(2,7),input_token_logprobs=torch.ones(2)),2)
    with pytest.raises(ValueError,match='customized_info'):
        backend._alloc_full_buffer(LogitsProcessorOutput(torch.ones(2,7),customized_info={'x':[1]}),2)
    with pytest.raises(ValueError,match='structure changed'):
        backend._copy_output_to_buffer(LogitsProcessorOutput(torch.ones(2,7),hidden_states=torch.ones(2,3)),
                                       LogitsProcessorOutput(torch.empty(2,7)),2)


@pytest.mark.parametrize('decode,prefill,locked,expected',[
    ('full','breakable',set(),('disabled','disabled')),
    ('full','disabled',{('decode','backend')},('disabled','disabled')),
    ('breakable','breakable',{('decode','backend')},('breakable','disabled')),
    ('full','breakable',{('prefill','backend')},('disabled','breakable')),
    ('disabled','disabled',{('decode','backend'),('prefill','backend')},('disabled','disabled')),
])
def test_normal_mode_requires_explicit_breakable_opt_in(monkeypatch,decode,prefill,locked,expected):
    from types import SimpleNamespace
    import sglang.srt.server_args as server_args
    import sglang.srt.arg_groups.overrides as overrides
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig,PhaseConfig
    args=SimpleNamespace(moe_a2a_backend='deepep',moe_runner_backend='triton',deepep_mode='normal',
                         enable_waterfill=False,_cuda_graph_config_locked=locked,
                         cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(backend=decode),prefill=PhaseConfig(backend=prefill)))
    monkeypatch.setattr(overrides,'run_post_process_pass',lambda *args: None)
    monkeypatch.setattr(server_args,'resolved_view',lambda args:args)
    server_args.ServerArgs._handle_a2a_moe(args)
    assert (args.cuda_graph_config.decode.backend,args.cuda_graph_config.prefill.backend)==expected
