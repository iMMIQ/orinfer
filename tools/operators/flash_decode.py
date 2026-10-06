"""Validate GPU greedy and compare exact dense INT8 single-token candidates."""
import argparse
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compile-cache',type=Path)
    p.add_argument('--greedy-only',action='store_true');a=p.parse_args()
    if a.compile_cache:shutil.copytree(a.compile_cache/'0.1.15',a.output/'cache'/'0.1.15',dirs_exist_ok=True)
    import torch
    from kernels.model.greedy import greedy_partials,greedy_merge
    from kernels.model.int8_projection import int8_projection,int8_gemv
    from tools.operators.common import configure,benchmark,write_json,export_kernel
    configure();report={'complete':False,'greedy':[],'dense':[]}
    for vocab in (1,65,1024,1025,248320):
        x=torch.randn((1,vocab),device='cuda');blocks=(vocab+1023)//1024
        vals=torch.empty(blocks,device='cuda');ids=torch.empty(blocks,device='cuda',dtype=torch.int32)
        bad=torch.empty_like(ids);out=torch.empty(2,device='cuda',dtype=torch.int32)
        part,merge=greedy_partials(vocab),greedy_merge(vocab)
        def run():part(x,vals,ids,bad);merge(vals,ids,bad,out)
        timing,graph=benchmark(run,repetitions=5)
        assert out.tolist()==[int(x.argmax()),0]
        x.fill_(-3);x[0,vocab-1]=7;x[0,0]=7;graph.replay();torch.cuda.synchronize()
        assert out.tolist()==[0,0],'Argmax tie rule changed'
        for value in (float('nan'),float('inf'),float('-inf')):
            x[0,-1]=value;graph.replay();torch.cuda.synchronize();assert out[1].item()==1
        report['greedy'].append({'vocab':vocab,'changed_graph_and_nonfinite':True,'timing':timing})
        export_kernel(part,a.output/f'greedy-partials-{vocab}');export_kernel(merge,a.output/f'greedy-merge-{vocab}')
        write_json(a.output/'results.json',report)
    if a.greedy_only:
        report['complete']=True;write_json(a.output/'results.json',report);return
    for n,k in [(65,128),(48,2560),(512,2560),(6144,2560),(10240,2560),(2560,6144),(248320,2560)]:
        x=torch.randint(-127,128,(1,k),device='cuda',dtype=torch.int8)
        w=torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8)
        ws=torch.rand(n,device='cuda').half()*.002;scale=torch.tensor([.003],device='cuda').half()
        expected=torch.empty((1,n),device='cuda',dtype=torch.float32);actual=torch.empty_like(expected)
        reference=int8_projection(1,n,k,'float32')
        rt,_=benchmark(lambda:reference(x,w,ws,scale,expected),repetitions=5)
        # Independent exact-integer check at representative output rows.
        rows=sorted(set([0,n//2,n-1]))
        integer=x.cpu().to(torch.int64)@w[rows].cpu().to(torch.int64).T
        oracle=(integer.float()*ws[rows].cpu().float())*scale.cpu().float()
        assert torch.equal(expected[:,rows].cpu(),oracle)
        case={'shape':[n,k],'tensorcore':rt,'candidates':[]}
        for bn in (4,8,16,32):
            reference(x,w,ws,scale,expected)
            kernel=int8_gemv(n,k,'float32',bn)
            def run():kernel(x.view(torch.int32),w.view(torch.int32),ws,scale,actual)
            ct,graph=benchmark(run,repetitions=5)
            assert torch.equal(actual,expected),(n,k,bn,float((actual-expected).abs().max()))
            x.neg_();scale.mul_(.5);graph.replay();reference(x,w,ws,scale,expected);torch.cuda.synchronize()
            assert torch.equal(actual,expected),'Changed-input graph mismatch'
            x.neg_();scale.mul_(2)
            case['candidates'].append({'block_n':bn,'exact':True,'timing':ct})
            export_kernel(kernel,a.output/f'dp4a-{n}-{k}-{bn}')
            print('dense',n,k,bn,'TC',rt['median_ms'],'DP4A',ct['median_ms'],flush=True)
        report['dense'].append(case);write_json(a.output/'results.json',report)
        del x,w,ws,scale,actual,expected
    report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
