"""Check actual batch-planning validation and invalid replay without timing."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    model=object.__new__(TritonAttnBackend);model.use_unified_attention_for_decode=True
    model.kv_indptr=torch.zeros(5,dtype=torch.int32,device='cuda')
    model._apply_cuda_graph_metadata=lambda **kw:model.kv_indptr[1:].copy_(kw['seq_lens'].cumsum(0))
    model._translate_cuda_graph_shared_pool_locs=lambda *a:None
    model._fill_cuda_graph_swa_out_cache_loc=lambda *a:None
    model._build_cuda_graph_forward_metadata=lambda *a:SimpleNamespace(unified_decode_kv_validated_rows=None)
    model.forward_metadata=SimpleNamespace(kv_indptr=model.kv_indptr[:3],unified_decode_kv_validated_rows=2)
    batch=SimpleNamespace(batch_size=4,req_pool_indices=None,seq_lens=torch.ones(4,dtype=torch.int32,device='cuda'),
                          forward_mode=ForwardMode.DECODE,spec_info=None,encoder_lens=None)
    model.init_forward_metadata_out_graph(batch,in_capture=True)
    assert model.forward_metadata.unified_decode_kv_validated_rows==4
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output=model.kv_indptr*2
    positives=[]
    for lengths in ([1,192,257,1000],[1,1,1,1],[17,63,129,1]):
        batch.seq_lens.copy_(torch.tensor(lengths,dtype=torch.int32,device='cuda'))
        model.init_forward_metadata_out_graph(batch)
        graph.replay();torch.cuda.synchronize()
        expected=[0]+torch.tensor(lengths).cumsum(0).mul(2).tolist()
        assert output.cpu().tolist()==expected
        positives.append(lengths)
    failures=[]
    for lengths in ([1,1,1,0],[1,-1,1,1]):
        batch.seq_lens.copy_(torch.tensor(lengths,dtype=torch.int32,device='cuda'))
        rejected=False
        try:model.init_forward_metadata_out_graph(batch)
        except ValueError:rejected=True
        if not rejected:raise RuntimeError('Invalid metadata was accepted before replay')
        torch.ones(1,device='cuda');torch.cuda.synchronize()
        failures.append(dict(lengths=lengths,rejected_before_replay=True,exception='ValueError',context_usable=True))
    result=dict(passed=True,positive_dynamic_replays=positives,invalid_metadata_cases=failures,
                scope='Actual TritonAttnBackend batch-planning validator with minimal GPU metadata; model gates run separately',
                timing_requested=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
