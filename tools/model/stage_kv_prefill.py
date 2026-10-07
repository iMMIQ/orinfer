"""Publish INT8 KV with one shared, demand-mapped FP16 prefill workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.model.publication import atomic_model, clone_model, commit_package, load_model

SCRATCH=('PrefillK','PrefillV')


def add_workspace(wrapper):
    meta=wrapper['metadata'];kv=meta.get('kv_cache',{})
    if wrapper['architecture']!='qwen3_5' or not kv.get('direct_prefill') or not kv.get('demand_mapping') or kv.get('prefill_workspace'):
        raise ValueError('Expected direct demand-mapped INT8 KV without prefill scratch')
    payloads=[b for b in meta['buffers'] if b['name'].endswith(('KPages','VPages'))]
    context=meta['max_context']
    if not payloads or any(b['dtype']!='i8' or b['shape']!=[context//128,128,4,256] for b in payloads):
        raise ValueError('Expected group64 INT8 KV geometry')
    names={b['name'] for b in meta['buffers']}
    if names.intersection(SCRATCH):raise ValueError('Scratch name collision')
    kv['prefill_workspace']={n:2048 for n in SCRATCH}
    for n in SCRATCH:
        meta['buffers'].append(dict(name=n,dtype='f16',shape=[1,context,1024],layout='token-major-prefill-kv-scratch',alignment=256,access='read_write'))
        wrapper['buffer_scopes'][n]='workspace'
    return wrapper


def publish(source,destination,engine,output):
    from tools.operators.common import configure,export_kernel
    from tools.operators.abi import parse_host,evaluate
    from kernels.model.kv_int8 import dequant_prefill_kv
    from kernels.model.attention_prefill_staged import attention_prefill_staged
    configure()
    source=source.resolve(strict=True);destination=destination.absolute()
    if destination.exists() or destination.is_symlink() or destination.resolve().is_relative_to(source):
        raise ValueError('Destination must be new and outside immutable source')
    w, old, package = load_model(source)
    w = add_workspace(w)
    context = w['metadata']['max_context']
    with atomic_model(destination, engine, command='validate-model') as staging:
        pkg = clone_model(source, staging, old)
        exports={}
        def exported(key,factory):
            if key not in exports:
                print('compile prefill KV',key,flush=True)
                directory=output/key;export_kernel(factory(),directory)
                assets={}
                for name,file in [('source','kernel.cu'),('host_abi','host.txt'),('module','kernel.cubin')]:
                    data=(directory/file).read_bytes();digest=hashlib.sha256(data).hexdigest()
                    relative='kernels/'+digest+Path(file).suffix
                    if not (pkg/relative).exists():(pkg/relative).write_bytes(data)
                    assets[name]=dict(file=relative,sha256=digest)
                exports[key]=(assets,parse_host((directory/'host.txt').read_text())[0])
            return exports[key]
        def bind(name,export,args):
            assets,host=export;dims=dict(batch=1)
            result=dict(name=name,args=[],symbol=host['symbol'],cooperative=False,**assets)
            for formal in host['ordered_arguments']:
                expr=formal['value']
                if expr.endswith('.data_ptr()'):result['args'].append(dict(kind='buffer',name=args[expr]))
                elif formal['ctype'] in ('ctypes.c_int','ctypes.c_int32'):
                    result['args'].append(dict(kind='i32',value=int(evaluate(expr,dims))))
                else:raise ValueError('Unexpected scalar ABI')
            launch=host['launch_expressions']
            for field,prefix in [('grid','gridDim'),('block','blockDim')]:
                result[field]=[int(evaluate(launch[prefix+a],dims)) for a in 'XYZ']
            result['shared_memory_bytes']=int(evaluate(launch['sharedMemBytes'],dims))
            return result
        kernels=[];count=0
        for binding in package['kernels']:
            match=re.fullmatch(r'prefill_m(512|2048)/layer(\d+)/k5',binding['name'])
            if not match or not any(a.get('name')=='FullQ' for a in binding['args']):kernels.append(binding);continue
            rows=int(match[1])
            original=parse_host((old/binding['host_abi']['file']).read_text())[0]
            args={f['value']:a['name'] for f,a in zip(original['ordered_arguments'],binding['args']) if a['kind']=='buffer'}
            dqargs={k:args[k] for k in ['K.data_ptr()','V.data_ptr()','KS.data_ptr()','VS.data_ptr()','Lengths.data_ptr()']}
            dqargs.update({'KO.data_ptr()':SCRATCH[0],'VO.data_ptr()':SCRATCH[1]})
            kernels.append(bind(binding['name'].rsplit('/',1)[0]+'/k4',exported('dequant',lambda:dequant_prefill_kv(context)),dqargs))
            args.update({'K.data_ptr()':SCRATCH[0],'V.data_ptr()':SCRATCH[1]})
            kernels.append(bind(binding['name'],exported('attention-'+str(rows),lambda:attention_prefill_staged(1,rows,context,kv_layout='token_major',block_m=32 if rows==512 else 64)),args))
            count+=1
        if count!=32:raise ValueError('Expected two chunk profiles across 16 attention layers')
        package['kernels']=kernels
        package['buffer_contracts']=[{k:v for k,v in b.items() if k!='data'} for b in w['metadata']['buffers']]
        digest = commit_package(staging, pkg, w, package)
    return dict(model=str(destination),execution_package=digest,prefill_workspace_capacity_bytes=4096*context)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--engine',type=Path,default=Path('target/release/orinfer'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(publish(a.model,a.destination,a.engine,a.output),indent=2))
