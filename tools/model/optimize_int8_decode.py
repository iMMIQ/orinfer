"""Build a quality-priority INT8 FFN operator package on existing W4 weights.

The architecture's registered slot order stays intact: post norm emits A8,
GateUp uses INT8 MMA, SwiGLU emits A8, and Down uses FP32 split-K partials.
Other projections and all recurrent state keep their existing precision.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.model.prepare import file_hash, write_json
from tools.model.upgrade_batching import bind
from tools.operators.abi import parse_host


def upgrade(model, destination, report, mode):
    from kernels.model.w4a8_decode import w4a8_decode
    from kernels.model.residual_norm_a8 import residual_norm_a8
    from kernels.operators.op30_activation_quantization import swiglu_activation_quantization
    from tools.operators.common import configure, export_kernel
    configure()
    data = json.loads((model/'cache/model.json').read_text())
    metadata = data['metadata']
    cache_root = Path(os.environ.get('ORIN_OPERATOR_CACHE',
        str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))/'orin-llm/operators')))
    installed = cache_root/data['operator_package']
    origin = installed if installed.exists() else model/'cache/operators'/data['operator_package']
    if file_hash(origin/'package.json') != data['operator_package']:
        raise ValueError('Source operator package digest mismatch')
    package = json.loads((origin/'package.json').read_text())
    config = json.loads((model/'config.json').read_text())
    text = config.get('text_config',config)
    h,f = text['hidden_size'],text['intermediate_size']
    buffers = {b['name']:b for b in metadata['buffers']}
    for layer in range(text['num_hidden_layers']):
        for family,n,k in [('GateUp',2*f,h),('Down',h,f)]:
            prefix = f'L{layer}_{family}'
            if buffers[prefix+'_P']['layout'] != 'u4_warp_n64_k128_mma_i8':
                raise ValueError('INT8 decode requires the lossless I8-fragment layout')
            if buffers[prefix+'_P']['shape'] != [n//64,k//128,128,8]:
                raise ValueError('Packed projection shape mismatch')
            if buffers[prefix+'_S']['shape'] != [n,k//128] or buffers[prefix+'_WS']['shape'] != [n]:
                raise ValueError('Projection quantization metadata shape mismatch')
    if not package.get('batch_profiles'):
        raise ValueError('Build a batch package before adding INT8 decode')
    if mode == 'group':
        name = 'DecodeDownAS'
        if name in buffers:
            raise ValueError('Model already contains INT8 decode workspace')
        spec = dict(name=name,dtype='f16',shape=[128,f//128],layout='contiguous',
                    alignment=256,access='read_write',data=None)
        metadata['buffers'].append(spec)
        data['buffer_scopes'][name] = 'workspace'
    destination.mkdir(parents=True)
    for path in model.iterdir():
        if path.is_file():
            shutil.copyfile(path,destination/path.name)
    cache = destination/'cache';cache.mkdir()
    shutil.copytree(model/'cache/weights',cache/'weights',copy_function=os.link)
    operator = cache/'operators'/'.building'
    shutil.copytree(origin,operator,copy_function=os.link)
    kernels = {k['name']:k for k in package['kernels']}
    exports = {}
    def compile_kernel(name,factory):
        print('compile',name,flush=True)
        out = operator/'int8-decode-aot'/name
        export_kernel(factory(),out)
        host = parse_host((out/'host.txt').read_text())
        if len(host)!=1:
            raise ValueError('Expected one kernel export')
        def identity(filename):
            path = out/filename
            return dict(file=str(path.relative_to(operator)),sha256=file_hash(path))
        exports[name] = dict(**host[0],module=identity('kernel.cubin'),source=identity('kernel.cu'),host_abi=identity('host.txt'))
    compile_kernel('post-norm',lambda: residual_norm_a8(h,text['rms_norm_eps'],threads=256,write_normalized=False))
    compile_kernel('swiglu',lambda: swiglu_activation_quantization(f,128 if mode=='group' else None,
                                                                threads=128 if mode=='group' else 512))
    for family,n,k in [('GateUp',2*f,h),('Down',h,f)]:
        for rows in [1,2,4,8,None]:
            # M>=32 reuses one W4 load across two MMA row tiles. The small-M
            # variants omit all inactive second-tile computation at compile time.
            for tile_m in ([16,32] if rows is None else [16]):
                key = f'{family}-m{rows or "dynamic"}-tile{tile_m}'
                compile_kernel(key,lambda rows=rows,n=n,k=k,family=family,tile_m=tile_m: w4a8_decode(
                    rows,n,k,1 if family=='GateUp' else 8,mode=mode,TILE_N=64 if tile_m==16 else 128,
                    TILE_M=tile_m,output_dtype='float16' if family=='GateUp' else 'float32',
                    activation_group=128 if mode=='group' and family=='Down' else None))
    # MTP verification reuses short prefill bindings. Use the same FFN law
    # there, including recurrent prefill used by the continuous scheduler.
    programs = [('decode',1,'decode')]
    programs.extend((f'batch_m{m}',m,'batch') for m in package['batch_profiles'])
    programs.extend((f'prefill_m{p["tokens"]}',p['tokens'],p['kind'])
                    for p in package['prefill_profiles'] if p['kind'] in ('sequence','recurrent'))
    for program,rows,kind in programs:
        if rows not in (1,2,4,8,16,32,64,128):
            raise ValueError('Unsupported short-profile row count')
        dims = dict(rows=rows,M=rows)
        for layer,layer_kind in enumerate(text['layer_types']):
            gdn = layer_kind=='linear_attention'
            first = {'decode':9 if gdn else 7,'batch':7 if gdn else 4,
                     'sequence':11 if gdn else 7,'recurrent':9 if gdn else 7}[kind]
            post = f'L{layer}_PostWeight'
            entries = [
                ('post-norm',dict(X='Mix',R='R1',W=post,Y='Norm',RO='R0',Q='TemporaryA8',S='TemporaryAS')),
                (f'GateUp-m{rows if rows<=8 else "dynamic"}-tile{16 if rows<=16 else 32}',
                    dict(A='TemporaryA8',PP=f'L{layer}_GateUp_P',S=f'L{layer}_GateUp_S',Z=f'L{layer}_GateUp_Z',
                         WS=f'L{layer}_GateUp_WS',AS='TemporaryAS',O='GateUp')),
                ('swiglu',dict(X='GateUp',Mask='UnusedA8Mask',Q='TemporaryA8',S='DecodeDownAS' if mode=='group' else 'TemporaryAS')),
                (f'Down-m{rows if rows<=8 else "dynamic"}-tile{16 if rows<=16 else 32}',
                    dict(A='TemporaryA8',PP=f'L{layer}_Down_P',S=f'L{layer}_Down_S',Z=f'L{layer}_Down_Z',
                         WS=f'L{layer}_Down_WS',AS='DecodeDownAS' if mode=='group' else 'TemporaryAS',O='Partial')),
            ]
            for slot,(key,pointers) in enumerate(entries,first):
                name = f'{program}/layer{layer}/k{slot}'
                if name not in kernels:
                    raise ValueError('Missing registered FFN slot')
                kernels[name] = bind(exports[key],name,pointers,dims)
    package['kernels'] = list(kernels.values())
    package['buffer_contracts'] = [{k:v for k,v in b.items() if k!='data'} for b in metadata['buffers']]
    raw = (json.dumps(package,ensure_ascii=False,indent=2)+'\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    new = operator/'package.new.json';new.write_bytes(raw);new.replace(operator/'package.json')
    operator.rename(operator.parent/digest)
    data['operator_package'] = digest
    write_json(cache/'model.json',data)
    report.mkdir(parents=True,exist_ok=True)
    write_json(report/'upgrade.json',dict(operator_package=digest,weight_bytes=metadata['weight_bytes'],
        weight_representation='unchanged',persistent_weight_bytes_added=0,
        workspace_bytes_added=128*(f//128)*2 if mode=='group' else 0,ffn_mode=mode))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--model-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['group','row'],default='group')
    args = parser.parse_args()
    destination = args.model_output.absolute()
    if destination.exists():
        parser.error('Destination already exists')
    staging = destination.with_name('.'+destination.name+'.building-'+uuid.uuid4().hex)
    try:
        upgrade(args.model.resolve(strict=True),staging,args.output,args.mode)
        cli = Path(__file__).resolve().parents[2]/'target/release/orin-llm'
        subprocess.run([str(cli),'plan-model',str(staging)],check=True,stdout=subprocess.DEVNULL)
        staging.rename(destination)
    except BaseException:
        if staging.exists():shutil.rmtree(staging)
        raise
    print('INT8 DECODE PACKAGE READY',destination,flush=True)


if __name__=='__main__':
    main()
