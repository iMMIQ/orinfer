"""Validate parallel HC projections, tails and stable-address graph updates."""
import argparse
from pathlib import Path

import torch

from kernels.model.hyperconnection import hc_injection, hc_down_partial, hc_down_finish
from tools.operators.common import configure, benchmark, error, export_kernel, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();configure();report={'complete':False,'cases':[]}
    for m,n,k,dtype in ((1,4,10240,'float16'),(3,7,129,'float16'),
                        (8,4,10240,'float16'),(3,7,129,'bfloat16')):
        x=torch.randn((m,k),device='cuda').to(getattr(torch,dtype))
        w=(torch.randn((n,k),device='cuda')/k**.5).bfloat16()
        guarded=torch.full((m+1,n),91.,device='cuda',dtype=x.dtype);out=guarded[:m]
        kernel=hc_injection(m,n,k,dtype)
        def run():kernel(x,w,out)
        timing,graph=benchmark(run,repetitions=8)
        oracle=(x.bfloat16().double() @ w.double().T).to(x.dtype)
        metric=error(out,oracle);assert metric['finite'] and metric['relative_l2']<.001,metric
        original=out.clone();saved_x=x.clone();saved_w=w.clone()
        x.zero_();graph.replay();torch.cuda.synchronize();assert bool((out==0).all())
        x.copy_(saved_x);w.zero_();graph.replay();torch.cuda.synchronize();assert bool((out==0).all())
        w.copy_(saved_w);out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
        assert torch.equal(out,original) and bool((guarded[-1]==91).all())
        report['cases'].append({'kind':'injection','rows':m,'outputs':n,'inner':k,'dtype':dtype,
            'fp64_error':metric,'changed_activation_and_weight_graph':True,'tail_guard':True,
            'timing':timing,'export':export_kernel(kernel,a.output/f'inject-{m}-{n}-{k}-{dtype}')})
        write_json(a.output/'results.json',report)
    for m,n,k in ((1,320,10240),(2,17,1024),(3,17,1024),(4,320,10240),
                   (7,17,1024),(8,320,10240)):
        splits=4
        x=torch.randn((m,k),device='cuda').half()
        w=(torch.randn((n,k),device='cuda')/k**.5).bfloat16()
        storage=torch.full((splits*m*n+17,),91.,device='cuda')
        partial=storage[:splits*m*n].view(splits,m,n)
        guarded=torch.full((m+1,n),91.,device='cuda',dtype=torch.float16);out=guarded[:m]
        project=hc_down_partial(m,n,k,splits);finish=hc_down_finish(m,n,splits)
        def run():project(x,w,partial);finish(partial,out)
        timing,graph=benchmark(run,repetitions=8)
        down=(x.bfloat16().double() @ w.double().T).half()
        oracle=torch.nn.functional.silu((down/4).half().float()).half()
        metric=error(out,oracle);assert metric['finite'] and metric['relative_l2']<.003,metric
        original=out.clone();saved_x=x.clone();saved_w=w.clone()
        partial.fill_(float('nan'));out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
        assert torch.equal(out,original)
        x.zero_();graph.replay();torch.cuda.synchronize();assert bool((out==0).all())
        x.copy_(saved_x);w.zero_();graph.replay();torch.cuda.synchronize();assert bool((out==0).all())
        w.copy_(saved_w);graph.replay();torch.cuda.synchronize();assert torch.equal(out,original)
        assert bool((guarded[-1]==91).all()) and bool((storage[-17:]==91).all())
        report['cases'].append({'kind':'down','rows':m,'outputs':n,'inner':k,'splits':splits,
            'fp64_error':metric,'partial_overwrite_and_changed_graph':True,'tail_guard':True,
            'timing':timing,'exports':[export_kernel(project,a.output/f'down-{m}-{n}-partial'),
                                      export_kernel(finish,a.output/f'down-{m}-{n}-finish')]})
        write_json(a.output/'results.json',report)
    report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
