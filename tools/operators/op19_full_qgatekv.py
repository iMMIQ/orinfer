"""Real layer3/35 W4A16 full attention projection verification and bounded timing."""
import argparse
import json
from pathlib import Path
from tools.reference import SOURCE
import re
import shutil
import subprocess
import sys
import time
import torch
import tilelang.language as T
from safetensors import safe_open
from common import configure, environment, error, export_kernel, identity, tensor_sha, write_json, benchmark
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/projections'))
import decode_common as real
from kernels.operators.op19_full_qgatekv import full_qgatekv
N, K = 14336, 5120
ROWS = (1,2,3,4,5,7,8,511,512,513,2048,8192)


def freeze(output):
    files=['kernels/operators/op19_full_qgatekv.py', 'kernels/operators/op03_ffn_gate_up.py', 'tools/operators/op19_full_qgatekv.py', 'tools/operators/common.py', 'tools/projections/decode_common.py', 'kernels/operators/op20_full_prepare.py', 'tools/operators/op20_full_prepare.py', 'artifacts/reference/reference-lock.json']
    frozen=[]
    for relative in files:
        src=ROOT/relative;dest=output/'measurement-source'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        frozen.append(identity(dest))
    src=(SOURCE / 'model_executor/models/qwen3_next.py')
    dest=output/'reference-source/qwen3_next.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    native=src.read_text()
    assert 'q_gate.view(*orig_shape, self.num_heads, -1)' in native and 'torch.chunk(q_gate, 2, dim=-1)' in native
    assert 'gate = torch.sigmoid(gate)' in native
    frozen.append(identity(dest))
    dest=output/'reference-source/config.json';shutil.copyfile(real.MODEL.parent/'config.json',dest)
    cfg=json.loads(dest.read_text())['text_config']
    assert (cfg['hidden_size'],cfg['num_attention_heads'],cfg['num_key_value_heads'],cfg['head_dim'])==(K,24,4,256)
    frozen.append(identity(dest))
    return frozen


def weights(layer):
    start=time.perf_counter();fields={key:[] for key in ('weight_packed','weight_scale','weight_zero_point')};sources=[]
    with safe_open(str(real.MODEL),framework='pt',device='cpu') as f:
        for part,n in [('q_proj',12288),('k_proj',1024),('v_proj',1024)]:
            raw={}
            for key in (*fields,'weight_shape'):
                name=f'model.language_model.layers.{layer}.self_attn.{part}.{key}'
                value=f.get_tensor(name);raw[key]=value
                sources.append(dict(name=name,shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_sha(value)))
            assert raw['weight_shape'].tolist()==[n,K]
            assert raw['weight_packed'].shape==(n,K//8) and raw['weight_packed'].dtype==torch.int32
            assert raw['weight_scale'].shape==(n,K//128) and raw['weight_scale'].dtype==torch.bfloat16
            assert raw['weight_zero_point'].shape==(n//8,K//128)
            for key in fields:fields[key].append(raw[key])
    raw={key:torch.cat(values).contiguous() for key,values in fields.items()}
    source_scale=raw['weight_scale'];raw['weight_scale']=source_scale.half()
    assert torch.equal(raw['weight_scale'].bfloat16(),source_scale)
    p,s,z,q,unpack_s=real.logical(raw)
    assert p.shape==(N,K//2) and s.shape==z.shape==(N,K//128)
    shifts=torch.arange(8,dtype=torch.int64)*4
    # Recreate every original packed INT32 word, including sign bit, independently.
    packed_roundtrip=(q.reshape(N,K//8,8).long()<<shifts).sum(-1).to(torch.int32)
    zero_roundtrip=(z.reshape(N//8,8,K//128).long()<<shifts[None,:,None]).sum(1).to(torch.int32)
    assert torch.equal(packed_roundtrip,raw['weight_packed'])
    assert torch.equal(zero_roundtrip,raw['weight_zero_point'])
    assert torch.equal(p&15,q[:,::2]) and torch.equal(p>>4,q[:,1::2])
    assert bool(((z>=0)&(z<=15)).all())
    info=dict(layer=layer,source_tensors=sources,source_read_hash_unpack_s=time.perf_counter()-start,
        logical_unpack_s=unpack_s,packed_codes_bitwise_verified=True,zero_roundtrip_verified=True,
        scale_exact_BF16_FP16_roundtrip=True,logical_hashes={key:tensor_sha(v) for key,v in [('P',p),('S',s),('Z',z)]},
        roundtrip_hashes=dict(weight_packed=tensor_sha(packed_roundtrip),weight_zero_point=tensor_sha(zero_roundtrip)))
    return p,s,z,q,info


def activations(layer):
    source_layer=0 if layer==3 else 32;found=[];prefill=None
    for path in sorted(real.ACTIVATIONS.glob('*in_proj_qkvz.json')):
        info=json.loads(path.read_text())
        if f'layers.{source_layer}.' not in info['kind']:continue
        source=real.ACTIVATIONS/info['file'];assert identity(source)['sha256']==info['file_sha256']
        value=torch.load(source,map_location='cpu',weights_only=True)
        assert value.dtype==torch.float16 and value.shape[1]==K
        if info['mode']=='prefill':assert value.shape==(512,K);prefill=(value,path,info)
        else:found.append((info['computed_tokens_before'],value,path,info))
    found.sort(key=lambda row:row[0]);assert len(found)==8 and prefill is not None
    cat=torch.cat([row[1] for row in found]);rows=[]
    for m in ROWS:
        small=m<=8
        value=cat[:m].contiguous() if small else prefill[0].repeat((m+511)//512,1)[:m].contiguous()
        origin=(f'actual layer{source_layer} GDN normalized-hidden projection INPUT used with full-attention layer{layer} weights; '
            'not a full-attention model trace; '+('consecutive M1 decode rows stacked, not simultaneous model batch' if small else
            ('captured 512-row prefill prefix' if m<=512 else 'repeat captured 512-row prefill; not genuine long-context activation')))
        rows.append((m,value,dict(origin=origin,source_layer=source_layer,metadata=[str(v[2]) for v in found[:m]] if small else str(prefill[1]),
            source_file_sha256=[v[3]['file_sha256'] for v in found[:m]] if small else prefill[2]['file_sha256'],tensor_sha256=tensor_sha(value))))
    return rows


def export(kernel,folder,bm):
    files=export_kernel(kernel,folder);host=(folder/'host.txt').read_text();source=(folder/'kernel.cu').read_text()
    abi=[s.strip().replace('.data_ptr()','') for s in re.search(r'arg_values\s*=\s*([^\n]+)',host).group(1).split(',')]
    assert abi==['A','C','P','S','Z','M'],abi
    signature=re.search(r'extern "C" __global__ void (\w+)\(([^)]*)\);',source)
    cfg={key:int(re.search(r'config\.'+key+r' = (\d+)',host).group(1)) for key in ('gridDimX','blockDimX','sharedMemBytes')}
    manifest=dict(schema_version=1,target='sm_87',toolchain=environment(),abi_order=abi,abi_types=['pointer']*5+['i32'],
        source_signature=signature.group(0),symbol=signature.group(1),grid=[cfg['gridDimX'],f'ceildiv(M,{bm})',1],
        block=[cfg['blockDimX'],1,1],shared_memory_bytes=cfg['sharedMemBytes'],cooperative=False,dynamic_dimension='M',workspace_bytes=0,
        launch_attribute=dict(MAX_DYNAMIC_SHARED_SIZE_BYTES=cfg['sharedMemBytes'],check_return_code=True),
        tensor_api_order=['A','P','S','Z','X'],generated_output_name='C equals op20 X',
        tensors={'A':['M',K,'f16','tight MK'],'P':[N,K//2,'u8','NK adjacent low/high U4'],'S':[N,40,'f16','NG group128'],
            'Z':[N,40,'i8 numeric 0..15','NG group128'],'C':['M',N,'f16','24(Q256,raw_gate256),K1024,V1024; tight token-major']},artifact=files)
    for name,flag in [('resources.txt','--dump-resource-usage'),('sass.txt','--dump-sass')]:
        result=subprocess.run(['/usr/local/cuda/bin/cuobjdump',flag,str(folder/'kernel.cubin')],capture_output=True,text=True)
        (folder/name).write_text(result.stdout+result.stderr);manifest[name]=dict(exit_code=result.returncode,**identity(folder/name))
    write_json(folder/'abi.json',manifest);return manifest


def check(out,ref):
    blocks={'all':(out,ref)}
    qg=out[:,:12288].reshape(-1,24,512);rg=ref[:,:12288].reshape(-1,24,512)
    blocks.update(Q=(qg[:,:,:256],rg[:,:,:256]),raw_gate=(qg[:,:,256:],rg[:,:,256:]),K=(out[:,12288:13312],ref[:,12288:13312]),V=(out[:,13312:],ref[:,13312:]))
    errors={key:error(a,b) for key,(a,b) in blocks.items()}
    assert all(v['finite'] and v['relative_l2']<=.002 for v in errors.values()),errors
    return errors


def graph_check(run,a,p,s,z,out,qcpu,scpu,zcpu,weights_changed):
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    expected=out.clone()
    def restored():
        torch.cuda.synchronize();assert torch.equal(out,expected)
    def zero():
        torch.cuda.synchronize();assert bool((out==0).all())
    out.fill_(float('nan'));graph.replay();restored()
    saved=a.clone();a.zero_();graph.replay();zero();a.copy_(saved);graph.replay();restored()
    flags=dict(poison_restore=True,changed_X_restore=True)
    if weights_changed:
        saved_s=s.clone();s.zero_();graph.replay();zero();s.copy_(saved_s);graph.replay();restored()
        saved_p=p.clone();p.copy_((z.to(torch.uint8)|(z.to(torch.uint8)<<4)).repeat_interleave(64,dim=1))
        graph.replay();zero();p.copy_(saved_p);graph.replay();restored()
        saved_z=z.clone();z.zero_();graph.replay();torch.cuda.synchronize()
        modified=real.dequant(qcpu,scpu,torch.zeros_like(zcpu)).float();ref=a.float()@modified.T
        flags['Z_mutation_error']=check(out,ref);z.copy_(saved_z);graph.replay();restored()
        flags.update(changed_P_restore=True,changed_S_restore=True,changed_Z_restore=True)
    return flags


def op20_connection(out,layer,output):
    from kernels.operators.op20_full_prepare import full_prepare
    from op20_full_prepare import math_reference
    with safe_open(str(real.MODEL),framework='pt',device='cpu') as f:
        wq=f.get_tensor(f'model.language_model.layers.{layer}.self_attn.q_norm.weight').half().cuda()
        wk=f.get_tensor(f'model.language_model.layers.{layer}.self_attn.k_norm.weight').half().cuda()
    freq=1/(1e7**(torch.arange(0,64,2,device='cuda',dtype=torch.float32)/64))
    angle=torch.arange(128,device='cuda',dtype=torch.float32)[:,None]*freq[None,:]
    cache=torch.cat((angle.cos(),angle.sin()),-1).half();pos=torch.tensor([63],device='cuda',dtype=torch.int32)
    req=torch.zeros(1,device='cuda',dtype=torch.int32);pages=req.reshape(1,1);status=req.clone()
    q=torch.empty((1,24,256),device='cuda',dtype=torch.float16);gate=torch.empty_like(q)
    k=torch.full((1,128,4,256),float('nan'),device='cuda',dtype=torch.float16);v=torch.empty_like(k)
    kernel=full_prepare(1,1,1,128,128)
    kernel(out,wq,wk,cache,req,pos,pages,status,q,gate,k,v,stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize();ref=math_reference(out,pos,wq,wk,cache)
    assert torch.equal(gate,ref[1]) and torch.equal(v[0,63],ref[3][0])
    errors=dict(Q=error(q,ref[0]),K=error(k[0,63],ref[2][0]))
    assert all(v['finite'] and v['relative_l2']<=.001 for v in errors.values()),errors
    files=export_kernel(kernel,output/'compiled/op20-connection')
    return dict(direct_same_output_buffer=True,no_copy_or_sigmoid=True,gate_V_bitexact=True,error=errors,artifact=files)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);parser.add_argument('--layers',default='3,35')
    parser.add_argument('--rows',default='');parser.add_argument('--candidates',default='shared16,register16,shared64,register64')
    args=parser.parse_args();configure();locked=json.loads((ROOT/'artifacts/reference/reference-lock.json').read_text())
    lock=next(v for v in locked['files'] if v['name']=='model.safetensors');assert real.MODEL.stat().st_size==lock['bytes']
    report=dict(environment=environment(),checkpoint=dict(lock,path=str(real.MODEL),identity_policy='reuse locked whole-file SHA; verify byte size and read-tensor hashes'),
        source_freeze=freeze(args.output),quantization='lossless repack of checkpoint AWQ, not new quantization',effective_weight_bits=4.1875,
        resident_weight_bytes=N*(K//2+40*3),workspace_bytes=0,candidates=[],layers=[],selected={})
    def save():write_json(args.output/'results.json',report)
    kernels={};selected_rows=set(map(int,args.rows.split(','))) if args.rows else set(ROWS)
    for layer in map(int,args.layers.split(',')):
        p0,s0,z0,qcpu,info=weights(layer);report['layers'].append(info);save()
        start=time.perf_counter();p,s,z=p0.cuda(),s0.cuda(),z0.cuda();torch.cuda.synchronize();info['H2D_s']=time.perf_counter()-start
        start=time.perf_counter();b32=real.dequant(qcpu,s0,z0).float();torch.cuda.synchronize();info['reference_halfdequant_s']=time.perf_counter()-start
        inputs=activations(layer)
        if not kernels:
            for name in args.candidates.split(','):
                bm=int(name[-2:]);impl=name[:-2];row=dict(name=name,BM=bm,BN=64,BK=128,stages=2,threads=128,screen=[])
                report['candidates'].append(row);save();start=time.perf_counter();kernel=full_qgatekv(T.dynamic('M'),implementation=impl,BM=bm)
                row['prepare_s']=time.perf_counter()-start;row['abi']=export(kernel,args.output/'compiled'/name,bm);kernels[name]=kernel
                for m,cpu_a,origin in inputs:
                    if m not in ((1,8) if bm==16 else (512,)):continue
                    a=cpu_a.cuda();out=torch.empty((m,N),device='cuda',dtype=torch.float16)
                    run=lambda:kernel(a,p,s,z,out,stream=torch.cuda.current_stream().cuda_stream)
                    start=time.perf_counter();run();torch.cuda.synchronize();first=time.perf_counter()-start
                    errors=check(out,a.float()@b32.T);timing,_=benchmark(run,repetitions=8 if m<=8 else 3,calls_per_replay=1)
                    row['screen'].append(dict(M=m,error=errors,first_launch_host_s=first,**timing));save()
                    del a,out
            for phase,m in [('decode',1),('prefill',512)]:
                report['selected'][phase]=min((v['median_ms'],row['name']) for row in report['candidates'] for v in row['screen'] if v['M']==m)[1]
            save()
        info['shapes']=[]
        for m,cpu_a,origin in inputs:
            if m not in selected_rows:continue
            name=report['selected']['decode' if m<=8 else 'prefill'];kernel=kernels[name];a=cpu_a.cuda();out=torch.empty((m,N),device='cuda',dtype=torch.float16)
            run=lambda:kernel(a,p,s,z,out,stream=torch.cuda.current_stream().cuda_stream)
            start=time.perf_counter();ref=a.float()@b32.T;torch.cuda.synchronize();ref_s=time.perf_counter()-start
            run();torch.cuda.synchronize();errors=check(out,ref);flags=graph_check(run,a,p,s,z,out,qcpu,s0,z0,m==1)
            timing,_=benchmark(run,repetitions=8 if m<=8 else 3,calls_per_replay=1)
            budget=.23 if m<=8 else 1.35*m/512
            result=dict(M=m,kernel=name,activation=origin,error=errors,reference_matmul_s=ref_s,graph=flags,
                output_bytes=m*N*2,workspace_bytes=0,budget_ms=budget,budget_ratio=timing['median_ms']/budget,**timing)
            if m==1:info['op20_connection']=op20_connection(out,layer,args.output)
            info['shapes'].append(result);save();print(json.dumps(dict(layer=layer,**result)),flush=True);del a,out,ref
        info['extremes']=[]
        kernel=kernels[report['selected']['decode']]
        for label,cpu_a in [('zero',torch.zeros((1,K),dtype=torch.float16)),('alternating_8',torch.where(torch.arange(K)%2==0,8.,-8.).half().reshape(1,K)),
                ('single_32',torch.nn.functional.pad(torch.tensor([[32.]],dtype=torch.float16),(0,K-1)))]:
            a=cpu_a.cuda();out=torch.empty((1,N),device='cuda',dtype=torch.float16);kernel(a,p,s,z,out,stream=torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize();errors=check(out,a.float()@b32.T);info['extremes'].append(dict(case=label,error=errors));save();del a,out
        del p,s,z,b32,qcpu,p0,s0,z0
    report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();report['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
    report['export_recheck']=[dict(path=entry['path'],matches=identity(entry['path'])['sha256']==entry['sha256']) for row in report['candidates'] for entry in row['abi']['artifact']['files']]
    assert all(v['matches'] for v in report['export_recheck']);save()


if __name__=='__main__':main()
