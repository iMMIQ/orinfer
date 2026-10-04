"""Check shared dequant scratch tails, exact reader equivalence and graph replay."""
import argparse
from pathlib import Path
import torch
from tools.operators.common import configure,write_json
from kernels.model.kv_int8 import dequant_prefill_kv,attention_prefill_int8
from kernels.model.attention_prefill_staged import attention_prefill_staged


def run(output,context=2048,tq=32,async_stages=0):
    configure();assert context>=2048 and tq in (32,64)
    k=torch.randint(-128,128,(context,4,256),device='cuda',dtype=torch.int8);v=torch.empty_like(k).random_(-128,128)
    ks=torch.rand(context,4,4,device='cuda',dtype=torch.float16)*.1;vs=torch.rand_like(ks)*.1
    ks[0].fill_(516);vs[1].fill_(2**-24)
    ko=torch.empty(context,1024,device='cuda',dtype=torch.float16);vo=torch.empty_like(ko)
    length=torch.zeros(1,device='cuda',dtype=torch.int32)
    pad=32 if async_stages else 1
    dq=dequant_prefill_kv(context,pad_to=pad).torch_function
    q=torch.randn(1,tq,6144,device='cuda',dtype=torch.float16);gate=torch.randn_like(q)
    pos=torch.zeros(1,tq,device='cuda',dtype=torch.int32)
    y=torch.empty(1,tq,24,256,device='cuda',dtype=torch.float16);yr=torch.empty_like(y)
    fast=attention_prefill_staged(1,tq,context,kv_layout='token_major',block_m=tq,num_stages=async_stages,interior_mask=bool(async_stages),contiguous_queries=bool(async_stages)).torch_function
    ref=attention_prefill_int8(1,tq,context,block_m=tq).torch_function
    def chain():dq(k,v,ks,vs,length,ko,vo);fast(q,ko,vo,gate,pos,length,y)
    chain();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):chain()
    lengths=sorted(set([0,1,145,1023,1024,1025,2048,context]))
    for n in lengths:
        ko.fill_(7);vo.fill_(7);length.fill_(n);pos.copy_(torch.arange(max(0,n-tq),max(0,n-tq)+tq,device='cuda',dtype=torch.int32)[None,:])
        if n==0:pos.fill_(-1)
        k.random_(-128,128);v.random_(-128,128);q.mul_(.95)
        graph.replay();ref(q,k,v,gate,pos,length,yr,ks,vs);torch.cuda.synchronize()
        for codes,scales,out in [(k,ks,ko),(v,vs,vo)]:
            expected=(codes[:n].float()*scales[:n].repeat_interleave(64,-1).float()).clamp(-65504,65504).half().reshape(n,1024)
            assert torch.equal(out[:n],expected),('dequant',n)
            padded=(n+pad-1)//pad*pad
            assert torch.equal(out[n:padded],torch.zeros_like(out[n:padded])),('padding not zero',n)
            assert torch.equal(out[padded:],torch.full_like(out[padded:],7)),('tail overwritten',n)
        assert torch.equal(y,yr),('attention changed',n,float((y.float()-yr.float()).abs().max()))
    write_json(output/'result.json',dict(status='passed',bit_exact_attention=True,bit_exact_dequant=True,changed_input_graph_replay=True,lengths=lengths,query_tokens=tq,async_stages=async_stages,seed=20261002))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--context',type=int,default=2048);p.add_argument('--query-tokens',type=int,default=32);p.add_argument('--async-stages',type=int,choices=[0,1,2],default=0);a=p.parse_args();run(a.output,a.context,a.query_tokens,a.async_stages)
