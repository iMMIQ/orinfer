"""Compare index normalization/compression at high positions, including a tail."""
import argparse
from pathlib import Path
import torch
from kernels.model.qsa import index_query,index_compress,index_pending
from tools.model.flash_qsa_reference import rope
from tools.model.flash_reference_math import norm
from tools.operators.common import configure,error,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();configure()
    capacity=262144;m=17;start=capacity-m
    raw=torch.randn((m+3,5,128),device='cuda').half()
    # There are three pending members at position 262127.
    assert start%4==3
    qk=raw[3:].contiguous();pending=torch.zeros((4,128),device='cuda',dtype=torch.float16)
    for offset in range(3):pending[(start-3+offset)%4]=raw[offset,4]
    qw=torch.randn(128,device='cuda')*.05;kw=torch.randn_like(qw)*.05
    q=torch.empty((m,4,128),device='cuda',dtype=torch.float16)
    cache=torch.full((capacity//4,128),float('nan'),device='cuda',dtype=torch.float16)
    pos=torch.tensor([start],device='cuda',dtype=torch.int32)
    query=index_query(m);compress=index_compress(m,capacity);store=index_pending(m)
    query(qk,qw+1,pos,q);compress(qk,pending,kw+1,pos,cache);store(qk,pos,pending)
    expected_q=rope(norm(qk[:,:4],qw),range(start,start+m))
    pooled=raw[:,4].float().reshape(-1,4,128).mean(1).half()
    expected_k=rope(norm(pooled[:,None],kw),range(start-3,capacity,4))[:,0]
    metrics={'query':error(q,expected_q),'compressed':error(cache[(start-3)//4:],expected_k)}
    assert all(x['finite'] and x['relative_l2']<.004 for x in metrics.values()),metrics
    for token in range(capacity-4,capacity):assert torch.equal(pending[token%4],qk[token-start,4])
    assert bool(torch.isnan(cache[:(start-3)//4]).all())
    write_json(a.output/'results.json',{'complete':True,'position':start,'capacity':capacity,'errors':metrics,
                                      'pending_exact':True,'prefix_unmodified':True})


if __name__=='__main__':main()
