"""WY selectable FP16 compensation, declared-policy and FP32/64 diagnostics."""
import argparse
import gc
import json
from pathlib import Path
import shutil

import torch
from common import benchmark,configure,environment,error,export_kernel,identity,write_json
from op14_gdn_chunk_wy import inputs,reference,fp64_subset
from kernels.operators.op14_gdn_chunk_wy import launch
from kernels.model.gdn_compensated import gdn_chunk_wy_compensated


def declared(a,k,v,g,beta,transform_low,input_low):
    aa=torch.tril(a)
    kk=k.repeat_interleave(3,dim=1).float()
    active=beta!=0
    kk=torch.where(active[...,None],kk,0.)
    vv=torch.where(active[...,None],v.float(),0.)
    gg=torch.where(active,g,0.)
    bk=(beta[...,None]*kk)*gg.exp()[...,None]
    bv=beta[...,None]*vv
    ahi=aa.half().float();alo=(aa-ahi).half().float()
    khi=bk.half().float();klo=(bk-khi).half().float()
    vhi=bv.half().float();vlo=(bv-vhi).half().float()
    w=ahi@khi;u=ahi@vhi
    if transform_low:
        w=alo@khi+w;u=alo@vhi+u
    if input_low:
        w=ahi@klo+w;u=ahi@vlo+u
    return w,u


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--value-tile',type=int,choices=(32,64),default=32)
    ap.add_argument('--policy',choices=('high','retain-transform','retain-input'),default='high')
    ap.add_argument('--quick',action='store_true');ap.add_argument('--validation-only',action='store_true')
    ap.add_argument('--cached-scalars',action='store_true')
    args=ap.parse_args();configure()
    transform_low=args.policy=='retain-transform';input_low=args.policy=='retain-input'
    sources=('kernels/model/gdn_compensated.py','tools/operators/op14_gdn_chunk_wy.py',
             'kernels/operators/op14_gdn_chunk_wy.py','tools/operators/gdn_reference.py')
    for path in sources:
        dest=args.output/'dependencies'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
    report=dict(status='running',environment=environment(),sources=[identity(p) for p in sources],
                policy=args.policy,flags=[transform_low,input_low],value_tile=args.value_tile,
                cached_scalars=args.cached_scalars,cases=[],
                scope='Synthetic WY: declared FP16 operand policy checks implementation; '
                      'FP32/64 drift diagnostic, FP32 W/U output, no model quality/TPS claim')
    specs=[(1,512,64,'random'),(1,513,64,'random'),(3,129,64,'pad_garbage'),(1,129,64,'beta0')]
    if not args.quick:
        specs += [(1,2048,64,'random'),(1,8192,64,'random'),(3,129,64,'beta1'),
                  (3,129,64,'identity'),(3,129,64,'strong'),(1,129,16,'random'),(1,129,32,'random')]
    kernels={};paired_kernels={}
    for batch,tokens,bt,mode in specs:
        if bt not in kernels:
            kernels[bt]=gdn_chunk_wy_compensated(bt,args.value_tile,
                compensate_transform=transform_low,compensate_inputs=input_low,cached_scalars=args.cached_scalars)
            paired_kernels[bt]=gdn_chunk_wy_compensated(bt,args.value_tile)
            export_kernel(kernels[bt],args.output/f'BT{bt}')
            export_kernel(paired_kernels[bt],args.output/f'paired-BT{bt}')
        a,k,v,g,beta=inputs(batch,tokens,bt,'float16',mode)
        expected=declared(a,k,v,g,beta,transform_low,input_low)
        fp32=reference(a,k,v,g,beta)
        w,u=torch.empty_like(expected[0]),torch.empty_like(expected[1])
        def run():
            launch(kernels[bt],a,k,v,g,beta,w,u,stream=torch.cuda.current_stream().cuda_stream)
        run();torch.cuda.synchronize()
        implementation={n:error(x,y) for n,x,y in zip(('W','U'),(w,u),expected)}
        assert all(m['finite'] and m['relative_l2']<2e-5 for m in implementation.values()),implementation
        drift={n:error(x,y) for n,x,y in zip(('W','U'),(w,u),fp32)}
        assert all(m['finite'] for m in drift.values())
        subset=fp64_subset(a,k,v,g,beta,w,u) if mode not in ('beta0','pad_garbage') else None
        if args.validation_only:
            timing=None;graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run()
        else:timing,graph=benchmark(run,repetitions=4)
        saved_beta=beta.clone();saved_w,saved_u=w.clone(),u.clone()
        beta.zero_();w.fill_(float('nan'));u.fill_(float('nan'))
        graph.replay();torch.cuda.synchronize()
        assert bool((w==0).all() and (u==0).all())
        beta.copy_(saved_beta);graph.replay();torch.cuda.synchronize()
        assert torch.equal(w,saved_w) and torch.equal(u,saved_u)
        valid=tokens-(k.shape[2]-1)*bt
        if valid<bt:
            assert bool((w[:,:,-1,valid:]==0).all() and (u[:,:,-1,valid:]==0).all())
        rec=dict(B=batch,T=tokens,BT=bt,mode=mode,timing=timing,implementation=implementation,
                 fp32_drift=drift,fp64_subset=subset,graph_zero_restore_bitwise=True)
        if not args.validation_only:
            rec['paired_current_timing'],paired_graph=benchmark(lambda:launch(
                paired_kernels[bt],a,k,v,g,beta,w,u,stream=torch.cuda.current_stream().cuda_stream),repetitions=4)
            del paired_graph
        report['cases'].append(rec);write_json(args.output/'result.json',report)
        print(json.dumps(dict(T=tokens,BT=bt,mode=mode,ms=None if timing is None else timing['median_ms'],
                              fp32_drift=drift)),flush=True)
        del a,k,v,g,beta,w,u,expected,fp32,graph,saved_beta,saved_w,saved_u
        gc.collect()
    report['status']='passed';write_json(args.output/'result.json',report)


if __name__=='__main__':main()
