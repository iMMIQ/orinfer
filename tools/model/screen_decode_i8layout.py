"""Actual M1 paired old vs lossless I8-fragment layout, same W4 math."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from common import benchmark,configure,environment,error,export_kernel,identity,write_json
from kernels.model.w4_decode_register_mma import w4_decode_register_mma
from kernels.model.w4_decode_i8layout import w4_decode_i8layout
from tools.quantization.w4_i8_pack import pack_array


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--vector-load',action='store_true')
    ap.add_argument('--byte-permute',action='store_true')
    ap.add_argument('--vector-words',type=int,choices=(2,4),default=2)
    ap.add_argument('--production-split',action='store_true')
    ap.add_argument('--validation-only',action='store_true')
    args=ap.parse_args();configure()
    assert not args.byte_permute or args.vector_load
    assert args.vector_words==2 or args.vector_load
    if args.vector_load:
        from kernels.model.w4_decode_i8layout_vector import w4_decode_i8layout_vector
        factory=w4_decode_i8layout_vector
    else:factory=w4_decode_i8layout
    report=dict(status='running',environment=environment(),model=identity(args.model),cases=[],sources=[],
                scope='Actual captured M1 activations; paired original and lossless alternative packed weights; not full-model decode or quality')
    for name in ['kernels/model/w4_decode_i8layout.py','kernels/model/w4_decode_register_mma.py','tools/quantization/w4_i8_pack.py','kernels/operators/op03_ffn_gate_up.py']:
        p=Path(name);shutil.copyfile(p,args.output/p.name);report['sources'].append(identity(p))
    report['vector_load']=args.vector_load
    report['byte_permute']=args.byte_permute
    report['vector_words']=args.vector_words
    report['production_split']=args.production_split
    report['validation_only']=args.validation_only
    if args.production_split:
        from kernels.operators.op32_split_k_merge import split_k_merge
        p=Path('kernels/operators/op32_split_k_merge.py');shutil.copyfile(p,args.output/p.name);report['sources'].append(identity(p))
    if args.vector_load:
        p=Path('kernels/model/w4_decode_i8layout_vector.py');shutil.copyfile(p,args.output/p.name);report['sources'].append(identity(p))
    m=json.loads(args.model.read_text());bs={b['name']:b for b in m['buffers']}
    root=Path('artifacts/experimental-vllm/activations')
    metadata=[json.loads(p.read_text()) for p in sorted(root.glob('*.json'))]
    for name,act in [('L0_GateUp','mlp.gate_up_proj'),('L0_Down','mlp.down_proj')]:
        tensors=[]
        for suffix,dtype in [('_P',np.int32),('_S',np.float16),('_Z',np.int8)]:
            b=bs[name+suffix];tensors.append(torch.from_numpy(np.fromfile(args.model.parent/b['data']['file'],dtype=dtype).reshape(b['shape'])))
        old_p,s,z=tensors
        native=pack_array(old_p.numpy(),verify=True)
        pp=torch.from_numpy(native.view(np.int32)).cuda();old_p=old_p.cuda();s=s.cuda();z=z.cuda()
        n,ng=s.shape;k=ng*128
        meta=next(x for x in metadata if x['mode']=='decode' and '.layers.0.' in x['kind'] and x['kind'].endswith(act))
        x=torch.load(root/meta['file'],map_location='cpu',weights_only=True).half().cuda();assert x.shape==(1,k)
        split=8 if args.production_split and name=='L0_Down' else 1
        tile_n=64 if split==8 else 128
        dtype='float32' if split==8 else 'float16'
        out=torch.empty((1,n),device='cuda',dtype=torch.float16);reference=torch.empty_like(out)
        old_partial=torch.empty((split,n),device='cuda',dtype=getattr(torch,dtype)) if split>1 else reference
        new_partial=torch.empty_like(old_partial) if split>1 else out
        merge=split_k_merge(1,N=n,SPLIT=split) if split>1 else None
        old=w4_decode_register_mma(n,k,SPLIT=split,output_dtype=dtype,TILE_N=tile_n)
        options={'byte_permute':args.byte_permute,'vector_words':args.vector_words} if args.vector_load else {}
        new=factory(n,k,SPLIT=split,output_dtype=dtype,TILE_N=tile_n,**options)
        def baseline():
            stream=torch.cuda.current_stream().cuda_stream
            old.adapter.func(x,old_p,s,z,old_partial,stream=stream)
            if merge is not None:merge.adapter.func(old_partial,reference,stream=stream)
        def run():
            stream=torch.cuda.current_stream().cuda_stream
            new.adapter.func(x,pp,s,z,new_partial,stream=stream)
            if merge is not None:merge.adapter.func(new_partial,out,stream=stream)
        if args.validation_only:
            baseline();run();torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run()
            bt=timing=None
        else:bt,_=benchmark(baseline,repetitions=12);timing,graph=benchmark(run,repetitions=12)
        metric=error(out,reference);assert torch.equal(out,reference),metric
        assert torch.equal(new_partial,old_partial)
        saved=x.clone();initial=out.clone();x.zero_();out.fill_(float('nan'))
        graph.replay();torch.cuda.synchronize();assert bool((out==0).all())
        x.copy_(saved);graph.replay();torch.cuda.synchronize();assert torch.equal(out,initial)
        export_kernel(new,args.output/name)
        if merge is not None:export_kernel(merge,args.output/(name+'-merge'))
        row=dict(weight=name,N=n,K=k,activation=meta,timing=timing,baseline_timing=bt,error=metric,
                 outputs_identical=True,full_packed_roundtrip=True,packed_bytes=pp.numel()*4,graph_zero_restore=True,
                 split=split,tile_n=tile_n,partial_dtype=dtype,partials_identical=True,merge_included=merge is not None)
        report['cases'].append(row);write_json(args.output/'result.json',report)
        print(json.dumps({key:value for key,value in row.items() if key!='activation'}),flush=True)
    report['status']='passed';write_json(args.output/'result.json',report)


if __name__=='__main__':main()
