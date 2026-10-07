"""Screen exact LUT4 FFN tiles using resident model weights and an integer oracle.

This is projection diagnostics, not model TPS or BF16 quality acceptance.
Optional captures contain the real prefill A8 inputs and their FP16 row scales.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path


def model_identity(model):
    raw=(model/'cache/model.json').read_bytes()
    descriptor=json.loads(raw)
    root=Path(os.environ.get('ORINFER_EXECUTION_CACHE',str(Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'orinfer/packages')))
    package=root/descriptor['execution_package']/'package.json'
    if not package.exists():package=model/'cache/packages'/descriptor['execution_package']/'package.json'
    package_raw=package.read_bytes()
    if hashlib.sha256(package_raw).hexdigest()!=descriptor['execution_package']:
        raise ValueError('Operator package identity mismatch')
    return hashlib.sha256(raw+package_raw+(model/'config.json').read_bytes()).hexdigest()


def packed_codes(pp, n, k):
    """Yield logical U4 rows from native INT8-fragment physical packing."""
    import numpy as np
    j=np.arange(k,dtype=np.int64)[None,:]
    packed=pp.cpu().numpy().view(np.uint32)
    for begin in range(0,n,256):
        i=np.arange(begin,min(begin+256,n),dtype=np.int64)[:,None]
        lanes=(i%64//16)*32+(i%8)*4+(j%16//4)
        words=(j%128//32)*2+(i%16//8)
        shifts=(j%32//16)*16+(j%4)*4
        code=((packed[i//64,j//128,lanes,words]>>shifts)&15).astype(np.uint8)
        yield begin, i, j, code


def decode_w8(pp, step, coef, n, k):
    """Independent decoder of the short-M approximate LUT4 codebook."""
    import numpy as np
    lookup=coef.cpu().numpy().view(np.uint32)
    coarse=step.cpu().numpy().view(np.uint8)
    decoded=np.empty((n,k),dtype=np.int8)
    for begin,i,j,code in packed_codes(pp,n,k):
        value=((lookup[i,j//128]>>((code%4)*8))&255).astype(np.int16)
        value+=(code//4).astype(np.int16)*coarse[i,j//128].astype(np.int16)-128
        if value.min() < -128 or value.max() > 127:raise ValueError('Invalid LUT4 codebook')
        decoded[begin:begin+len(i)]=value
    return decoded


def decode_strict_w8(pp, s, z, ws):
    """Long-M W8: FP16 dequantization, FP32 division, ties-even INT8 rounding."""
    import numpy as np
    scale=s.cpu().numpy();zero=z.cpu().numpy();row=ws.cpu().numpy()
    n,groups=scale.shape;k=groups*128
    decoded=np.empty((n,k),dtype=np.int8)
    for begin,i,j,code in packed_codes(pp,n,k):
        weight=(code.astype(np.int16)-zero[i,j//128]).astype(np.float16)*scale[i,j//128]
        value=np.rint(weight.astype(np.float32)/row[i].astype(np.float32))
        decoded[begin:begin+len(i)]=np.clip(value,-127,127).astype(np.int8)
    return decoded


def main():
    import numpy as np
    import torch
    from safetensors import safe_open
    from kernels.model.w4a8_lut4 import w4a8_lut4
    from tools.operators.common import benchmark, configure, export_kernel, write_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--activations',type=Path)
    p.add_argument('--rows',type=int,default=512)
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--families',nargs='+',choices=['GateUp','Down'],default=['GateUp','Down'])
    p.add_argument('--bm',type=int,nargs='+',choices=[64,128,256],default=[64,128,256])
    p.add_argument('--bn',type=int,nargs='+',choices=[64,128],default=[64,128])
    p.add_argument('--stages',type=int,nargs='+',choices=[1,2],default=[1,2])
    p.add_argument('--grid-orders',nargs='+',choices=['nfirst','mfirst'],default=['nfirst'])
    p.add_argument('--min-blocks',type=int,nargs='+',choices=[1,2,3,4],default=[1])
    p.add_argument('--warp-m',type=int,nargs='+',choices=[1,2,4],default=[1])
    a=p.parse_args()
    if a.rows<=0:raise ValueError('rows must be positive')
    configure()
    torch.manual_seed(20261002)
    # Establish a live context before TileLang compiles its launch ABI.
    torch.empty(1,device='cuda')
    weights=a.model/'cache/weights'
    index=json.loads((weights/'model.safetensors.index.json').read_text())['weight_map']
    def tensor(name):
        with safe_open(str(weights/index[name]),framework='pt',device='cpu') as source:
            return source.get_tensor(name).cuda()
    fingerprint=model_identity(a.model)
    if a.activations:
        captured=json.loads((a.activations/'capture.json').read_text())
        if captured['fingerprint']!=fingerprint:
            raise ValueError('Capture belongs to another model/operator package')
    report=dict(status='running',kind='lut4_ffn',seed=20261002,fingerprint=fingerprint,rows=a.rows,cases=[],
                scope='LUT4 tile screening against exact integer reconstruction; not end-to-end throughput or quality.')
    for family in a.families:
        base=f'L{a.layer}_{family}'
        pp,s,step,coef,ws=[tensor(base+suffix) for suffix in ['_P','_S','_Step','_LUT4','_WS']]
        n,groups=s.shape;k=groups*128
        decoded=decode_w8(pp,step,coef,n,k)
        w8=torch.from_numpy(decoded).cuda()
        if a.activations:
            prefix=a.activations/f'{base}_m{a.rows}'
            raw=prefix.with_suffix('.a8').read_bytes();scales=prefix.with_suffix('.scale.f16').read_bytes()
            expected=captured['inputs'][prefix.name]
            if hashlib.sha256(raw).hexdigest()!=expected['activation_sha256'] or hashlib.sha256(scales).hexdigest()!=expected['scale_sha256']:
                raise ValueError('Captured activation payload changed')
            aq=torch.from_numpy(np.frombuffer(raw,dtype=np.int8).copy().reshape(a.rows,k)).cuda()
            asc=torch.from_numpy(np.frombuffer(scales,dtype=np.float16).copy()).cuda()
            origin='real model prefill A8 capture'
        else:
            aq=torch.randint(-32,33,(a.rows,k),device='cuda',dtype=torch.int8)
            asc=torch.full((a.rows,),0.01,device='cuda',dtype=torch.float16)
            origin='random implementation input; timing requires whole-model confirmation'
        # Every 128-term FP32 dot is an exact integer (bound < 2**24).
        # Accumulate those exact partials in INT32, avoiding a full-K FP32
        # reduction that can silently round before the final conversion.
        integer=torch.zeros((a.rows,n),device='cuda',dtype=torch.int32)
        if k*128*128 >= 2**31:raise ValueError('Full projection can overflow the INT32 oracle')
        for begin in range(0,k,128):
            partial=aq[:,begin:begin+128].float()@w8[:,begin:begin+128].float().T
            integer+=partial.to(torch.int32)
        golden=(integer.float()*asc[:,None].float()*ws[None,:].float()).half()
        del decoded,w8,integer,partial
        choices=[]
        for bm in a.bm:
            for bn in a.bn:
                for stages,grid_order,min_blocks,warp_m in itertools.product(a.stages,a.grid_orders,a.min_blocks,a.warp_m):
                    if bn//16*warp_m*32>512:
                        continue
                    epilogue=stages==2
                    kernel=w4a8_lut4(a.rows,n,k,bm,bn,stages,coalesced_epilogue=epilogue,grid_order=grid_order,min_blocks=min_blocks,warp_m=warp_m)
                    guarded=torch.full((a.rows*n+256,),123.,device='cuda',dtype=torch.float16)
                    result=guarded[:-256].view(a.rows,n)
                    def run():
                        kernel.adapter.func(aq,pp,s,step,coef,asc,ws,result,stream=torch.cuda.current_stream().cuda_stream)
                    timing,graph=benchmark(run,repetitions=8)
                    assert torch.equal(result,golden),(family,bm,bn,stages,'integer oracle mismatch')
                    assert bool((guarded[-256:]==123.).all()),'Output guard changed'
                    saved=aq.clone();aq.zero_();result.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
                    assert bool((result==0).all()),'Graph ignored changed input'
                    aq.copy_(saved);graph.replay();torch.cuda.synchronize()
                    assert torch.equal(result,golden) and bool((guarded[-256:]==123.).all())
                    key=f'{base}-m{a.rows}-bm{bm}-bn{bn}-s{stages}-{grid_order}-resident{min_blocks}-warpM{warp_m}'
                    export_kernel(kernel,a.output/key)
                    entry=dict(family=family,n=n,k=k,bm=bm,bn=bn,stages=stages,coalesced_epilogue=epilogue,grid_order=grid_order,min_blocks=min_blocks,warp_m=warp_m,
                               activation_origin=origin,timing=timing,integer_oracle_equal=True,
                               tail_guard=True,graph_zero_restore=True,export=key)
                    choices.append(entry);report['cases'].append(entry);write_json(a.output/'result.json',report)
                    print(key,round(timing['median_ms'],4),'ms',flush=True)
                    del kernel,graph,guarded,result,saved
        finalists=sorted(choices,key=lambda c:c['timing']['median_ms'])[:3]
        for choice in finalists:
            kernel=w4a8_lut4(a.rows,n,k,choice['bm'],choice['bn'],choice['stages'],
                            coalesced_epilogue=choice['coalesced_epilogue'],grid_order=choice['grid_order'],min_blocks=choice['min_blocks'],warp_m=choice['warp_m'])
            result=torch.empty_like(golden)
            def run():
                kernel.adapter.func(aq,pp,s,step,coef,asc,ws,result,stream=torch.cuda.current_stream().cuda_stream)
            choice['screen_timing']=choice['timing']
            choice['timing'],graph=benchmark(run,repetitions=32)
            assert torch.equal(result,golden)
            choice['finalist_rechecked']=True
            del kernel,result,graph
        best=min(finalists,key=lambda c:c['timing']['median_ms'])
        # Verify the chosen tile's partial final block and a real captured
        # replay that changes only its last valid row.
        tail_a=torch.cat([aq,aq[:1]],dim=0);tail_scale=torch.cat([asc,asc[:1]],dim=0)
        tail_expected=torch.cat([golden,golden[:1]],dim=0)
        tail_kernel=w4a8_lut4(a.rows+1,n,k,best['bm'],best['bn'],best['stages'],coalesced_epilogue=best['coalesced_epilogue'],grid_order=best['grid_order'],min_blocks=best['min_blocks'],warp_m=best['warp_m'])
        tail_guard=torch.full(((a.rows+1)*n+256,),123.,device='cuda',dtype=torch.float16)
        tail_out=tail_guard[:-256].view(a.rows+1,n)
        def tail_run():
            tail_kernel.adapter.func(tail_a,pp,s,step,coef,tail_scale,ws,tail_out,stream=torch.cuda.current_stream().cuda_stream)
        tail_run();torch.cuda.synchronize()
        assert torch.equal(tail_out,tail_expected) and bool((tail_guard[-256:]==123.).all())
        tail_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(tail_graph):tail_run()
        tail_a[-1].zero_();tail_out.fill_(float('nan'));tail_graph.replay();torch.cuda.synchronize()
        assert torch.equal(tail_out[:-1],golden) and bool((tail_out[-1]==0).all())
        tail_a[-1].copy_(aq[0]);tail_graph.replay();torch.cuda.synchronize()
        assert torch.equal(tail_out,tail_expected) and bool((tail_guard[-256:]==123.).all())
        best['nonaligned_tail_rows']=a.rows+1
        export_kernel(tail_kernel,a.output/(best['export']+'-tail'))
        write_json(a.output/'result.json',report)
        del tail_a,tail_scale,tail_expected,tail_kernel,tail_guard,tail_out,tail_graph
        best['selected']=True
        write_json(a.output/'result.json',report)
        print('best',family,best['export'],best['timing']['median_ms'],flush=True)
        del pp,s,step,coef,ws,aq,asc,golden
    report['status']='passed';write_json(a.output/'result.json',report)


if __name__=='__main__':main()
