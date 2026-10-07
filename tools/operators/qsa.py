"""Validate native QSA at compression boundaries through the actual 256k limit."""
import argparse
from pathlib import Path
import torch
from kernels.model import qsa
from tools.model.flash_next.reference.qsa import index,select,attention,quantize_kv,dequantize_kv
from tools.operators.common import configure,environment,error,export_kernel,write_json,benchmark


def selection_ops(m,capacity):
    blocks=(capacity+3)//4;segments=(blocks+1023)//1024
    def empty(shape,dtype=torch.int32):return torch.empty(shape,device='cuda',dtype=dtype)
    scores=empty((m,blocks),torch.float32);prefix=empty((m,));remaining=empty((m,))
    hist=empty((m,segments,256));counts=empty((m,2,segments));offsets=empty((m,2,segments));greater=empty((m,))
    selected=empty((m,2051))
    ops=[]
    for shift in (24,16,8,0):
        ops.extend([(qsa.radix_histogram(m,capacity,shift),(scores,prefix,None,hist)),
                    (qsa.radix_choose(m,capacity,shift),(hist,prefix,remaining,None))])
    ops.extend([(qsa.selection_counts(m,capacity),(scores,prefix,None,counts)),
                (qsa.selection_offsets(m,capacity),(counts,offsets,greater,None,selected)),
                (qsa.selection_scatter(m,capacity),(scores,prefix,offsets,greater,None,selected))])
    return scores,selected,ops


def checked(x,y,limit=.004):
    metric=error(x,y)
    assert metric['finite'] and metric['relative_l2']<limit,metric
    return metric


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();configure();report={'complete':False,'environment':environment(),'cases':[]}
    # Chunk crosses old pending groups, several new groups and an incomplete tail.
    capacity=8192;total=2060;m=9;start=2047
    raw=torch.randn((total,5,128),device='cuda').half()
    qw=torch.randn(128,device='cuda')*.05;kw=torch.randn_like(qw)*.05
    refq,refk=index(raw,qw,kw)
    pending=raw[start-4:start,4].clone()
    # Ring ordering is absolute token modulo four.
    for token in range(start-4,start):pending[token%4]=raw[token,4]
    cache=torch.full(((capacity+3)//4,128),float('nan'),device='cuda',dtype=torch.float16)
    cache[:start//4]=refk[:start//4]
    position=torch.tensor([start],device='cuda',dtype=torch.int32)
    query=torch.empty((m,4,128),device='cuda',dtype=torch.float16)
    chunk=raw[start:start+m].contiguous()
    prep=qsa.index_query(m);compress=qsa.index_compress(m,capacity);store=qsa.index_pending(m)
    def prepare():
        prep(chunk,qw+1,position,query);compress(chunk,pending,kw+1,position,cache);store(chunk,position,pending)
    saved=pending.clone();prepare();torch.cuda.synchronize()
    report['compression']={'query':checked(query,refq[start:start+m]),
        'cache':checked(cache[:(start+m)//4],refk[:(start+m)//4])}
    for token in range(start+m-4,start+m):assert torch.equal(pending[token%4],raw[token,4])
    # Restore all index state before graph; replay with changed input and position.
    pending.copy_(saved)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):prepare()
    pending.copy_(saved);graph.replay();torch.cuda.synchronize()
    checked(cache[:(start+m)//4],refk[:(start+m)//4])
    export_kernel(compress,a.output/'compress')
    for capacity,positions in [(2053,[2047,2048,2049,2051,2052]),
                               (8192,[4095,4096,8189]),(32768,[32760,32761]),
                               (262144,[262140,262141,262142,262143])]:
        # consecutive queries consume the Position scalar; arbitrary first position.
        first=positions[0];m=len(positions)
        positions=list(range(first,first+m))
        pos=torch.tensor([first],device='cuda',dtype=torch.int32)
        blocks=(capacity+3)//4
        iq=torch.randn((m,4,128),device='cuda').half()
        ck=torch.randn((blocks,128),device='cuda').half()
        scores,selected,ops=selection_ops(m,capacity)
        score=qsa.index_scores(m,capacity)
        ops=[(score,(iq,ck,pos,scores))]+[(k,tuple(pos if x is None else x for x in args)) for k,args in ops]
        def run_selection():
            for kernel,args in ops:kernel(*args)
        run_selection();torch.cuda.synchronize()
        expect=select(iq,ck,positions)
        for actual,reference in zip(selected,expect):
            assert torch.equal(actual[actual>=0].sort().values,reference[reference>=0].sort().values),(capacity,actual,reference)
            assert len(actual[actual>=0].unique())==len(actual[actual>=0])
        # Ties are deterministic and must still select exactly min(512,complete).
        iq.zero_();run_selection();torch.cuda.synchronize();tie=select(iq,ck,positions)
        assert torch.equal(selected,tie)
        iq.normal_();run_selection();torch.cuda.synchronize()
        # Full KV allocation really addresses the end of 262144, not a tiny alias.
        key=torch.randn((capacity,2,256),device='cuda').half()
        value=torch.randn_like(key)
        # Only the reference input is FP16; native persistent storage is INT8.
        ki=torch.empty_like(key,dtype=torch.int8);vi=torch.empty_like(value,dtype=torch.int8)
        ks=torch.empty((capacity,2,4),device='cuda',dtype=torch.float16);vs=torch.empty_like(ks)
        zero=torch.zeros(1,device='cuda',dtype=torch.int32)
        kv=qsa.kv_store(capacity,capacity);kv(key,value,zero,ki,vi,ks,vs)
        refki,refks=quantize_kv(key);refvi,refvs=quantize_kv(value)
        assert torch.equal(ki,refki) and torch.equal(vi,refvi)
        assert torch.equal(ks,refks) and torch.equal(vs,refvs)
        key_ref=dequantize_kv(ki,ks);value_ref=dequantize_kv(vi,vs)
        query=torch.randn((m,24,256),device='cuda').half()
        gate=torch.randn_like(query);out=torch.empty_like(query)
        maximum=torch.empty((m,24,8),device='cuda');den=torch.empty_like(maximum)
        partial=torch.empty((m,24,8,256),device='cuda')
        sparse=qsa.sparse_attention(m,capacity);merge=qsa.sparse_merge(m)
        def run():
            run_selection();sparse(query,ki,vi,ks,vs,selected,pos,maximum,den,partial);merge(maximum,den,partial,gate,out)
        run();torch.cuda.synchronize();metric=checked(out,attention(query,key_ref,value_ref,gate,selected))
        saved_out=out.clone()
        timing,g=benchmark(run,repetitions=3)
        iq.mul_(.9);query.mul_(.8);pos.fill_(first-1);g.replay();torch.cuda.synchronize()
        expected=select(iq,ck,list(range(first-1,first-1+m)))
        for actual,reference in zip(selected,expected):
            assert torch.equal(actual[actual>=0].sort().values,reference[reference>=0].sort().values)
        changed=checked(out,attention(query,key_ref,value_ref,gate,selected))
        assert not torch.equal(out,saved_out)
        report['cases'].append({'capacity':capacity,'positions':positions,'error':metric,
            'changed_graph_error':changed,'kv_dtype':'int8','kv_group':64,
            'kv_store_exact':True,'fp16_kv_output_error':error(out,attention(query,key,value,gate,selected)),
            'exact_selection':True,'ties':True,'timing':timing})
        write_json(a.output/'results.json',report)
        print('QSA passed',capacity,timing['median_ms'],flush=True)
        del key,value,ki,vi,ks,vs,refki,refvi,refks,refvs,key_ref,value_ref,query,gate,out,g,ops,partial,maximum,den
    report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
