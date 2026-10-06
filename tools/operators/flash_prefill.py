"""Tune larger prefill row tiles against the original exact projection outputs."""
import argparse
from pathlib import Path
import torch
from kernels.model.int8_projection import int8_projection
from kernels.model.hyperconnection import hc_projection
from tools.operators.common import configure,benchmark,error,write_json,export_kernel


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();configure()
    report={'complete':False,'cases':[]}
    for kind,n,k in [('int8',6144,2560),('int8',512,2560),('int8',2560,6144),
                     ('hc',320,10240),('hc',10240,320),('hc',4,10240)]:
        m=128
        x=torch.randint(-127,128,(m,k),device='cuda',dtype=torch.int8) if kind=='int8' else (torch.randn((m,k),device='cuda')*.2).half()
        weight=torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8) if kind=='int8' else (torch.randn((n,k),device='cuda')*.03).bfloat16()
        ws=(torch.rand(n,device='cuda')*.002).half();scale=(torch.rand(m,device='cuda')*.003).half()
        expected=torch.empty((m,n),device='cuda',dtype=torch.float16);actual=torch.empty_like(expected)
        def build(bm,bn):return int8_projection(m,n,k,block_m=bm) if kind=='int8' else hc_projection(m,n,k,block_m=bm,dtype='float16',block_n=bn)
        def launch(kernel,out):
            if kind=='int8':kernel(x,weight,ws,scale,out)
            else:kernel(x,weight,out)
        ref=build(16,64);rt,_=benchmark(lambda:launch(ref,expected),repetitions=10)
        case={'kind':kind,'N':n,'K':k,'reference':rt,'candidates':[]}
        choices=[(32,64),(64,64)] if kind=='int8' else [(32,32),(32,64),(64,64)]
        for bm,bn in choices:
            launch(ref,expected);kernel=build(bm,bn)
            timing,graph=benchmark(lambda:launch(kernel,actual),repetitions=10)
            assert torch.equal(actual,expected),error(actual,expected)
            x.neg_();graph.replay();launch(ref,expected);torch.cuda.synchronize()
            assert torch.equal(actual,expected),error(actual,expected)
            x.neg_()
            export_kernel(kernel,a.output/f'{kind}-{n}-{k}-{bm}-{bn}')
            case['candidates'].append({'block_m':bm,'block_n':bn,'exact':True,'timing':timing})
            print('prefill',kind,n,k,bm,bn,'old',rt['median_ms'],'new',timing['median_ms'],flush=True)
        report['cases'].append(case);write_json(a.output/'results.json',report)
    report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
