"""Real layer0/32 W4A16 GDN QKV/Z correctness, bounded screening and AOT evidence."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import torch
import tilelang.language as T
from safetensors import safe_open
from common import configure, environment, error, export_kernel, identity, tensor_sha, write_json, benchmark
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/projections'))
import decode_common as real
from kernels.operators.op06_gdn_qkvz import gdn_qkvz


def weights(layer):
    started = time.perf_counter()
    fields = {field: [] for field in ('weight_packed', 'weight_scale', 'weight_zero_point')}
    sources = []
    with safe_open(str(real.MODEL), framework='pt', device='cpu') as handle:
        for part in real.PARTS['gdn_qkvz']:
            for field in fields:
                name = f'model.language_model.layers.{layer}.{part}.{field}'
                tensor = handle.get_tensor(name)
                sources.append(dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=tensor_sha(tensor)))
                fields[field].append(tensor)
    raw = {field: torch.cat(values).contiguous() for field, values in fields.items()}
    half = raw['weight_scale'].half()
    assert torch.equal(half.bfloat16(), raw['weight_scale'])
    raw['weight_scale'] = half
    p, s, z, q, unpack_s = real.logical(raw)
    assert p.shape == (16384, 2560) and s.shape == z.shape == (16384, 40)
    assert bool(((z >= 0) & (z <= 15)).all())
    assert torch.equal(p & 15, q[:, ::2]) and torch.equal(p >> 4, q[:, 1::2])
    return p, s, z, q, dict(layer=layer, source_tensors=sources,
        source_read_hash_unpack_s=time.perf_counter()-started, logical_unpack_s=unpack_s,
        logical_hashes={key:tensor_sha(value) for key,value in [('P',p),('S',s),('Z',z)]},
        packed_codes_bitwise_verified=True)


def activations(layer):
    found = []
    prefill = None
    for path in sorted(real.ACTIVATIONS.glob('*in_proj_qkvz.json')):
        info = json.loads(path.read_text())
        if f'layers.{layer}.' not in info['kind']:
            continue
        source = real.ACTIVATIONS / info['file']
        assert identity(source)['sha256'] == info['file_sha256']
        value = torch.load(source, map_location='cpu', weights_only=True)
        assert value.dtype == torch.float16 and value.shape[1] == 5120
        if info['mode'] == 'prefill':
            assert value.shape == (512,5120)
            prefill = (value, path, info)
        else:
            found.append((info['computed_tokens_before'],value,path,info))
    found.sort(key=lambda row:row[0])
    assert len(found) == 8 and prefill is not None
    cat = torch.cat([row[1] for row in found])
    rows = []
    for m in (1,2,3,4,5,7,8):
        value = cat[:m].contiguous()
        rows.append((m,value,dict(origin='consecutive real M1 decode inputs stacked; not simultaneous model batch',
            metadata=[str(row[2]) for row in found[:m]], source_file_sha256=[row[3]['file_sha256'] for row in found[:m]],
            positions=[row[0] for row in found[:m]], tensor_sha256=tensor_sha(value))))
    for m in (511,512,513,2048,8192):
        value = prefill[0].repeat((m+511)//512,1)[:m].contiguous()
        rows.append((m,value,dict(origin='actual 512-row captured prefill prefix' if m<=512 else
            'repeat actual 512-row prefill; shape/tail validation, not genuine long-context activation',
            metadata=str(prefill[1]), source_file_sha256=prefill[2]['file_sha256'],tensor_sha256=tensor_sha(value))))
    return rows


def export(kernel, folder, bm):
    files = export_kernel(kernel,folder)
    host = (folder/'host.txt').read_text()
    source = (folder/'kernel.cu').read_text()
    abi = [item.strip().replace('.data_ptr()','') for item in re.search(r'arg_values\s*=\s*([^\n]+)',host).group(1).split(',')]
    assert set(abi[:-1]) == {'A','P','S','Z','QKV','ZOUT'} and abi[-1] == 'M', abi
    signature = re.search(r'extern "C" __global__ void (\w+)\(([^)]*)\);', source)
    cfg = {key:int(re.search(r'config\.'+key+r' = (\d+)',host).group(1)) for key in ('gridDimX','blockDimX','sharedMemBytes')}
    env = environment()
    manifest = dict(schema_version=1,target='sm_87',
        toolchain={key:str(env[key]) for key in ('torch','tilelang','cuda')},environment=env,abi_order=abi,
        abi_types=['pointer']*6+['i32'],source_signature=signature.group(0),symbol=signature.group(1),
        grid=[cfg['gridDimX'],f'ceildiv(M,{bm})',1],block=[cfg['blockDimX'],1,1],
        shared_memory_bytes=cfg['sharedMemBytes'],cooperative=False,dynamic_dimension='M',workspace_bytes=0,
        launch_attribute={'MAX_DYNAMIC_SHARED_SIZE_BYTES':cfg['sharedMemBytes'],'check_return_code':True},
        tensors={'A':['M',5120,'f16','tight MK'],'P':[16384,2560,'u8','NK adjacent low/high U4'],
        'S':[16384,40,'f16','NG group128'],'Z':[16384,40,'i8 numeric 0..15','NG group128'],
        'QKV':['M',10240,'f16','tight Q2048/K2048/V6144'],'ZOUT':['M',6144,'f16','tight 48heads x128']},artifact=files)
    for name,flag in [('resources.txt','--dump-resource-usage'),('sass.txt','--dump-sass')]:
        result = subprocess.run(['/usr/local/cuda/bin/cuobjdump',flag,str(folder/'kernel.cubin')],capture_output=True,text=True)
        (folder/name).write_text(result.stdout+result.stderr)
        manifest[name] = dict(exit_code=result.returncode,**identity(folder/name))
    write_json(folder/'abi.json',manifest)
    return manifest


def validate_graph(run,a,p,s,z,qkv,zout,qcpu,scpu,zcpu,ref,check_weights):
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    expected=(qkv.clone(),zout.clone())
    def assert_expected():
        torch.cuda.synchronize()
        assert torch.equal(qkv,expected[0]) and torch.equal(zout,expected[1])
    def assert_zero():
        torch.cuda.synchronize()
        assert bool((qkv==0).all() and (zout==0).all())
    qkv.fill_(float('nan'));zout.fill_(float('nan'));graph.replay();assert_expected()
    saved=a.clone();a.zero_();graph.replay();assert_zero();a.copy_(saved);graph.replay();assert_expected()
    flags=dict(poison_restore=True,changed_X_restore=True)
    if check_weights:
        saved_s=s.clone();s.zero_();graph.replay();assert_zero();s.copy_(saved_s);graph.replay();assert_expected()
        saved_p=p.clone();p.copy_((z.to(torch.uint8)|(z.to(torch.uint8)<<4)).repeat_interleave(64,dim=1))
        graph.replay();assert_zero();p.copy_(saved_p);graph.replay();assert_expected()
        saved_z=z.clone();z.zero_();graph.replay();torch.cuda.synchronize()
        modified=real.dequant(qcpu,scpu,torch.zeros_like(zcpu)).float()
        modified_ref=a.float()@modified.T
        errors={'QKV':error(qkv,modified_ref[:,:10240]),'ZOUT':error(zout,modified_ref[:,10240:])}
        assert all(item['finite'] and item['relative_l2']<=.002 for item in errors.values()),errors
        z.copy_(saved_z);graph.replay();assert_expected()
        flags.update(changed_P_restore=True,changed_S_restore=True,changed_Z_restore=True,Z_mutation_error=errors)
    return flags


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--layers',default='0,32');parser.add_argument('--rows',default='')
    parser.add_argument('--candidates',default='shared16,register16,shared64,register64')
    args=parser.parse_args();configure()
    report=dict(environment=environment(),checkpoint=str(real.MODEL),checkpoint_sha256=real.checkpoint_sha256(real.MODEL),
        quantization='lossless repack of existing checkpoint; no new quantization',effective_weight_bits=4.1875,
        resident_weight_bytes=43909120,workspace_bytes=0,candidates=[],layers=[],failures=[],selected={})
    def save():write_json(args.output/'results.json',report)
    kernels={}
    for layer in map(int,args.layers.split(',')):
        p0,s0,z0,q,info=weights(layer);report['layers'].append(info);save()
        start=time.perf_counter();p,s,z=p0.cuda(),s0.cuda(),z0.cuda();torch.cuda.synchronize();info['H2D_s']=time.perf_counter()-start
        start=time.perf_counter();b32=real.dequant(q,s0,z0).float();torch.cuda.synchronize();info['reference_halfdequant_s']=time.perf_counter()-start
        inps=activations(layer)
        if not kernels:
            for name in args.candidates.split(','):
                bm=16 if name.endswith('16') else 64;implementation=name[:-2]
                candidate=dict(name=name,BM=bm,BN=64,BK=128,stages=2,threads=128,screen=[]);report['candidates'].append(candidate);save()
                try:
                    start=time.perf_counter();kernel=gdn_qkvz(T.dynamic('M'),implementation=implementation,BM=bm)
                    candidate['prepare_s']=time.perf_counter()-start;candidate['abi']=export(kernel,args.output/'compiled'/name,bm);kernels[name]=kernel
                    for m,cpu_a,origin in inps:
                        if m not in ((1,8) if bm==16 else (512,)):continue
                        a=cpu_a.cuda();qkv=torch.empty((m,10240),device='cuda',dtype=torch.float16);zout=torch.empty((m,6144),device='cuda',dtype=torch.float16)
                        run=lambda:kernel(a,p,s,z,qkv,zout,stream=torch.cuda.current_stream().cuda_stream)
                        start=time.perf_counter();run();torch.cuda.synchronize();first=time.perf_counter()-start
                        ref=a.float()@b32.T
                        errors={'QKV':error(qkv,ref[:,:10240]),'ZOUT':error(zout,ref[:,10240:])}
                        assert all(item['finite'] and item['relative_l2']<=.002 for item in errors.values()),errors
                        timing,_=benchmark(run,repetitions=8 if m<=8 else 3,calls_per_replay=1)
                        candidate['screen'].append(dict(M=m,error=errors,first_launch_host_s=first,**timing));save()
                        del a,qkv,zout,ref
                except Exception as exc:
                    report['failures'].append(dict(candidate=name,exception=repr(exc),traceback=traceback.format_exc()));save();print(traceback.format_exc(),flush=True)
            for phase,m in [('decode',1),('prefill',512)]:
                valid=[(shape['median_ms'],candidate['name']) for candidate in report['candidates'] for shape in candidate['screen'] if shape['M']==m]
                assert valid,phase
                report['selected'][phase]=min(valid)[1]
            save()
        info['shapes']=[]
        for m,cpu_a,origin in inps:
            if args.rows and m not in set(map(int,args.rows.split(','))):continue
            name=report['selected']['decode' if m<=8 else 'prefill'];kernel=kernels[name]
            a=cpu_a.cuda();qkv=torch.empty((m,10240),device='cuda',dtype=torch.float16);zout=torch.empty((m,6144),device='cuda',dtype=torch.float16)
            run=lambda:kernel(a,p,s,z,qkv,zout,stream=torch.cuda.current_stream().cuda_stream)
            start=time.perf_counter();ref=a.float()@b32.T;torch.cuda.synchronize();reference_s=time.perf_counter()-start
            run();torch.cuda.synchronize();errors={'QKV':error(qkv,ref[:,:10240]),'ZOUT':error(zout,ref[:,10240:])}
            assert all(item['finite'] and item['relative_l2']<=.002 for item in errors.values()),errors
            flags=validate_graph(run,a,p,s,z,qkv,zout,q,s0,z0,ref,m==1)
            timing,_=benchmark(run,repetitions=8 if m<=8 else 3,calls_per_replay=1)
            result=dict(M=m,kernel=name,activation=origin,error=errors,reference_matmul_s=reference_s,
                graph=flags,output_bytes=m*16384*2,workspace_bytes=0,**timing)
            info['shapes'].append(result);save();print(json.dumps(dict(layer=layer,**result)),flush=True)
            del a,qkv,zout,ref
        del p,s,z,b32,q,p0,s0,z0
    report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();save()


if __name__=='__main__':main()
