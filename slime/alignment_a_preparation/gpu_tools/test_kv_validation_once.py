"""Batch KV validation must cover live buffers without per-layer scalar reads."""
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.attention.triton_backend as triton_backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def backend():
    result=object.__new__(triton_backend.TritonAttnBackend)
    result.use_unified_attention_for_decode=True
    return result


@pytest.mark.parametrize('lengths',[[0,1],[-1,3]])
def test_empty_or_negative_lengths_raise_value_error(lengths):
    model=backend();indptr=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],dtype=torch.int32)
    with pytest.raises(ValueError,match='empty KV'):
        model._validate_unified_decode_kv_lengths(indptr,2,ForwardMode.DECODE)


def test_empty_batch_and_positive_padding_are_valid():
    model=backend()
    assert model._validate_unified_decode_kv_lengths(torch.tensor([0],dtype=torch.int32),0,ForwardMode.IDLE)==0
    assert model._validate_unified_decode_kv_lengths(torch.arange(5,dtype=torch.int32),4,ForwardMode.DECODE)==4


def test_baseline_and_prefill_do_not_read_a_scalar(monkeypatch):
    model=backend()
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected scalar read')
    monkeypatch.setattr(torch.Tensor,'item',forbidden)
    assert model._validate_unified_decode_kv_lengths(torch.arange(5),4,ForwardMode.EXTEND) is None
    model.use_unified_attention_for_decode=False
    assert model._validate_unified_decode_kv_lengths(torch.arange(5),4,ForwardMode.DECODE) is None


def test_48_layer_forwards_use_one_batch_check(monkeypatch):
    model=backend();q=torch.ones(4,128);indptr=torch.tensor([0,1,18,81,210],dtype=torch.int32)
    counts=[];original=torch.Tensor.item
    def item(tensor,*args,**kwargs):counts.append(1);return original(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'item',item)
    checked=model._validate_unified_decode_kv_lengths(indptr,4,ForwardMode.DECODE)
    model.forward_metadata=SimpleNamespace(kv_indptr=indptr,kv_indices=torch.arange(210),unified_decode_kv_validated_rows=checked)
    model.dcp_size=1
    model.token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda layer:q,get_value_buffer=lambda layer:q)
    model.extend_attention_fwd_unified=lambda query,output,*args,**kwargs:output.copy_(query)
    monkeypatch.setattr(triton_backend,'logit_capping_mod',lambda *args:0.)
    layer=SimpleNamespace(tp_q_head_num=1,qk_head_dim=128,v_head_dim=128,logit_capping_method=None,logit_cap=0,
        sliding_window_size=None,k_scale=None,v_scale=None,scaling=1.,xai_temperature_len=-1,layer_id=0)
    model.page_size=64
    batch=SimpleNamespace(spec_info=None)
    for i in range(48):
        layer.layer_id=i
        assert torch.equal(model.forward_decode(q,None,None,layer,batch,save_kv_cache=False),q)
    assert len(counts)==1
    model.forward_metadata.unified_decode_kv_validated_rows=None
    with pytest.raises(RuntimeError,match='not validated'):
        model.forward_decode(q,None,None,layer,batch,save_kv_cache=False)


def setup_replay(model):
    model.kv_indptr=torch.zeros(5,dtype=torch.int32)
    def apply(**kwargs):
        model.kv_indptr[1:]=kwargs['seq_lens'].cumsum(0)
    model._apply_cuda_graph_metadata=apply
    model._translate_cuda_graph_shared_pool_locs=lambda *args:None
    model._fill_cuda_graph_swa_out_cache_loc=lambda *args:None
    model.forward_metadata=SimpleNamespace(kv_indptr=model.kv_indptr[:3],unified_decode_kv_validated_rows=2)
    model._build_cuda_graph_forward_metadata=lambda *args:SimpleNamespace(unified_decode_kv_validated_rows=None)


def test_replay_checks_live_larger_bucket_even_if_metadata_view_is_smaller():
    model=backend();setup_replay(model)
    batch=SimpleNamespace(batch_size=4,req_pool_indices=torch.arange(4),seq_lens=torch.tensor([1,1,1,0]),
                          forward_mode=ForwardMode.DECODE,spec_info=None,encoder_lens=None)
    with pytest.raises(ValueError,match='empty KV'):
        model.init_forward_metadata_out_graph(batch)
    batch.seq_lens[-1]=1
    model.init_forward_metadata_out_graph(batch)
    model.init_forward_metadata_out_graph(batch,in_capture=True)
    assert model.forward_metadata.unified_decode_kv_validated_rows==4
