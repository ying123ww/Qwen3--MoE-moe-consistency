"""EP4 normal fixed-capacity graph dispatch/combine routing/zero receive complement."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from deep_ep import Buffer,Config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);dist.init_process_group('nccl')
    groups=[dist.new_group([0,1,2,3]),dist.new_group([4,5,6,7])];group=groups[rank//4];ep_rank=rank%4
    config=Config(20,9,256,6,128);nvl=config.get_nvl_buffer_size_hint(4096,4);rdma=config.get_rdma_buffer_size_hint(4096,4)
    buffer=Buffer(group,nvl,rdma)
    if buffer.group_size!=4 or buffer.runtime.get_num_rdma_ranks()!=1:raise RuntimeError('Expected single-node EP4')
    x=torch.full((2,2048),ep_rank+1,dtype=torch.bfloat16,device='cuda')
    ids=torch.empty((2,8),dtype=torch.int64,device='cuda');weights=torch.full((2,8),1/8,dtype=torch.float32,device='cuda');scale=torch.ones(1,dtype=torch.bfloat16,device='cuda')
    seed=torch.arange(8,device='cuda',dtype=torch.int64).expand(2,8);ids.copy_(seed)
    captures={};outputs={};received={};recv_ids={}
    for capacity in (16,8):
        def compute():
            nr,nrdma,ne,mask,event=buffer.get_dispatch_layout(ids,128)
            y,ri,rw,counts,handle,event=buffer.dispatch(x,num_tokens_per_rank=nr,num_tokens_per_rdma_rank=nrdma,num_tokens_per_expert=ne,is_token_in_rank=mask,topk_idx=ids,topk_weights=weights,num_worst_tokens=capacity,config=config)
            if y.shape!=(capacity,2048) or counts!=[]:raise RuntimeError('Dynamic receive count was not bypassed')
            valid=(ri>=0).any(dim=1);y.masked_fill_(~valid[:,None],0.);rw.masked_fill_(ri<0,0.)
            y.mul_(scale)
            combined,_,event=buffer.combine(y,handle,config=config)
            return combined,y,ri
        for _ in range(2):result=compute()
        torch.cuda.synchronize();dist.barrier();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):result=compute()
        captures[capacity]=graph;outputs[capacity]=result[0];received[capacity]=result[1];recv_ids[capacity]=result[2]
    cases=['all-to-rank0','balanced-local','spread-all-ranks','one-idle-sender','all-idle']
    rows=[]
    for name in cases:
        if name=='balanced-local':layout=seed+ep_rank*32;factor=1.
        elif name=='spread-all-ranks':layout=torch.tensor([0,1,32,33,64,65,96,97],device='cuda',dtype=torch.int64).expand(2,8);factor=.25
        else:layout=seed.clone();factor=1.
        if name=='one-idle-sender' and ep_rank==3:layout.fill_(-1)
        if name=='all-idle':layout.fill_(-1)
        ids.copy_(layout);scale.fill_(factor)
        valid_send=(layout>=0).any(dim=1);expected=x.clone();expected.masked_fill_(~valid_send[:,None],0.)
        actual={}
        for capacity in (16,8):
            captures[capacity].replay();torch.cuda.synchronize();dist.barrier()
            value=outputs[capacity].clone();valid_recv=(recv_ids[capacity]>=0).any(dim=1)
            mismatch=int((value.view(torch.int16)!=expected.view(torch.int16)).sum())
            if mismatch:raise RuntimeError(f'Controlled routing/combine mismatch {name} cap{capacity}')
            actual[capacity]=dict(output_bit_mismatches=mismatch,real_receive_rows=int(valid_recv.sum()),zero_receive=not bool(valid_recv.any()),physical_receive_shape=list(received[capacity].shape))
        rows.append(dict(case=name,rank=rank,ep_rank=ep_rank,physical_send_rows=2,logical_send_rows=int(valid_send.sum()),capacities=actual))
    (args.run/f'controlled-rank-{rank}.json').write_text(json.dumps(dict(rank=rank,complete=True,cases=rows,scope='Actual normal EP4 graph dispatch/combine,identity synthetic experts; complements full model and does not replace it'),indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
