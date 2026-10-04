"""Validate fused INT8 KV writers/readers, tails, page mappings and graph replay."""
import argparse
import json
from pathlib import Path
import torch
from tools.operators.common import configure, write_json
from kernels.vision.bridge import full_prepare_mrope
from kernels.model.kv_int8 import full_prepare_mrope_int8, attention_prefill_int8, paged_attention_partials_int8, dequant_pairs_probe
from kernels.model.attention_prefill_staged import attention_prefill_staged
from kernels.model.attention_decode_staged import paged_attention_partials_gqa_staged


def relative(actual, expected):
    assert bool(actual.isfinite().all())
    a,e=actual.float(),expected.float()
    return float((a-e).norm()/e.norm().clamp_min(1e-20))


def run(output):
    configure(); torch.manual_seed(20261002)
    codes=torch.arange(-128,128,device='cuda',dtype=torch.int32).repeat(6).reshape(-1,2)
    scales=torch.tensor([2**-24,2**-14,.03125,1.,515.5,516.],device='cuda',dtype=torch.float16).repeat_interleave(128)
    packed=(codes[:,0]&255)|((codes[:,1]&255)<<8)
    decoded=torch.empty(len(packed),2,device='cuda',dtype=torch.float16)
    dequant_pairs_probe(len(packed)).torch_function(packed,scales,decoded)
    expected=(codes.float()*scales.float()[:,None]).clamp(-65504,65504).half()
    torch.cuda.synchronize(); assert torch.equal(decoded,expected), 'Packed converter disagrees'
    pages,context,rows=4,512,17
    section=(11,11,10)
    x=torch.randn(rows,14336,device='cuda',dtype=torch.float16)
    x[0,13312:13568]=65504
    x[1,13312:13568]=-65504
    x[2,13312:13376]=0
    x[3,13312:13376]=2**-24
    wq=torch.randn(256,device='cuda',dtype=torch.float16)*.1
    wk=torch.randn_like(wq)*.1
    angles=torch.arange(context,device='cuda')[:,None]* (1/(1e7**(torch.arange(0,64,2,device='cuda')/64)))
    rotary=torch.cat((angles.cos(),angles.sin()),1).half()
    req=torch.zeros(rows,device='cuda',dtype=torch.int32)
    pos=torch.tensor([0,1,63,64,126,127,128,129,145,255,256,257,383,384,385,510,511],device='cuda',dtype=torch.int32)
    table=torch.tensor([[2,0,3,1]],device='cuda',dtype=torch.int32)
    status=torch.zeros(1,device='cuda',dtype=torch.int32)
    mrope=torch.arange(context,device='cuda',dtype=torch.int32)[:,None].repeat(1,3)
    q=torch.empty(rows,24,256,device='cuda',dtype=torch.float16); g=torch.empty_like(q)
    qr=torch.empty_like(q); gr=torch.empty_like(g)
    k=torch.zeros(pages,128,4,256,device='cuda',dtype=torch.float16); v=torch.zeros_like(k)
    ki=torch.zeros_like(k,dtype=torch.int8); vi=torch.zeros_like(v,dtype=torch.int8)
    ks=torch.zeros(pages,128,4,4,device='cuda',dtype=torch.float16); vs=torch.zeros_like(ks)
    ref=full_prepare_mrope(pages,context,section,max_position=context).torch_function
    writer=full_prepare_mrope_int8(pages,context,section,max_position=context).torch_function
    ref(x,wq,wk,rotary,req,pos,table,status,mrope,qr,gr,k,v)
    writer(x,wq,wk,rotary,req,pos,table,status,mrope,q,g,ki,vi,ks,vs)
    torch.cuda.synchronize()
    assert torch.equal(q,qr) and torch.equal(g,gr), 'Q/Gate changed'
    def dq(code,scale): return (code.float()*scale.repeat_interleave(64,-1).float()).clamp(-65504,65504).half()
    kd,vd=dq(ki,ks),dq(vi,vs)
    for orig,code,scale in [(k,ki,ks),(v,vi,vs)]:
        expected=(orig.float()/scale.repeat_interleave(64,-1).float().clamp_min(2**-24)).round().clamp(-127,127).to(torch.int8)
        assert torch.equal(code,expected), 'RNE codes disagree'
    metrics=dict(q_exact=True,gate_exact=True,pack_codes_exact=True,k_relative_l2=relative(kd,k),v_relative_l2=relative(vd,v))
    # Writer graph rewrites the same slots from changed projection data.
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): writer(x,wq,wk,rotary,req,pos,table,status,mrope,q,g,ki,vi,ks,vs)
    x.mul_(.75);graph.replay();ref(x,wq,wk,rotary,req,pos,table,status,mrope,qr,gr,k,v)
    torch.cuda.synchronize(); assert torch.equal(q,qr) and torch.equal(g,gr)
    status.fill_(1);before=ki.clone();before_q=q.clone();graph.replay();torch.cuda.synchronize()
    assert torch.equal(ki,before) and torch.equal(q,before_q);status.zero_()
    # Random complete cache includes nonidentity pages and awkward split boundaries.
    ki.random_(-127,128);vi.random_(-127,128);ks.uniform_(.002,.05);vs.uniform_(.002,.05)
    kd,vd=dq(ki,ks),dq(vi,vs)
    length=torch.tensor([145],device='cuda',dtype=torch.int32)
    positions=torch.tensor([144],device='cuda',dtype=torch.int32)
    qd=torch.randn(1,24,256,device='cuda',dtype=torch.float16)
    mi=torch.empty(1,24,8,device='cuda');li=torch.empty_like(mi);oi=torch.empty(1,24,8,256,device='cuda')
    mr=torch.empty_like(mi);lr=torch.empty_like(mi);orr=torch.empty_like(oi)
    decoder=paged_attention_partials_int8(pages,pages).torch_function
    refdecoder=paged_attention_partials_gqa_staged(pages,pages).torch_function
    decoder(qd,ki,vi,table,length,positions,mi,li,oi,ks,vs)
    refdecoder(qd,kd,vd,table,length,positions,mr,lr,orr)
    torch.cuda.synchronize(); metrics['decode_relative_l2']=relative(oi,orr)
    print('decode metrics',metrics, 'maxima', float(mi.max()), float(mr.max()), 'denom',float(li.max()),float(lr.max()),flush=True)
    assert metrics['decode_relative_l2']<.002
    assert torch.allclose(mi,mr,atol=.002,rtol=.002) and torch.allclose(li,lr,atol=.002,rtol=.002)
    graph2=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2):decoder(qd,ki,vi,table,length,positions,mi,li,oi,ks,vs)
    qd.mul_(1.25);length.fill_(512);positions.fill_(511);graph2.replay()
    refdecoder(qd,kd,vd,table,length,positions,mr,lr,orr);torch.cuda.synchronize()
    assert relative(oi,orr)<.002
    length.zero_();graph2.replay();torch.cuda.synchronize();assert torch.equal(oi,torch.zeros_like(oi))
    tq=32; qp=torch.randn(1,tq,6144,device='cuda',dtype=torch.float16);gate=torch.randn_like(qp)
    pp=torch.arange(113,145,device='cuda',dtype=torch.int32)[None,:];length.fill_(145)
    y=torch.empty(1,tq,24,256,device='cuda',dtype=torch.float16);yr=torch.empty_like(y)
    # Identity contiguous view is the production single-request prefill contract.
    prefill=attention_prefill_int8(1,tq,context,kv_layout='token_major').torch_function
    refprefill=attention_prefill_staged(1,tq,context,kv_layout='token_major').torch_function
    prefill(qp,ki,vi,gate,pp,length,y,ks,vs)
    refprefill(qp,kd,vd,gate,pp,length,yr);torch.cuda.synchronize()
    metrics['prefill_relative_l2']=relative(y,yr);assert metrics['prefill_relative_l2']<.002
    graph3=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph3):prefill(qp,ki,vi,gate,pp,length,y,ks,vs)
    qp.mul_(.8); pp.add_(367);length.fill_(512);graph3.replay()
    refprefill(qp,kd,vd,gate,pp,length,yr);torch.cuda.synchronize();assert relative(y,yr)<.002
    metrics.update(packed_converter_exhaustive=True,status='passed',changed_input_graph_replay=True,invalid_write_guard=True,seed=20261002)
    write_json(output/'result.json',metrics);print(json.dumps(metrics),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
