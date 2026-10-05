"""Publish a prepared model with direct KV reads and demand-mapped storage.

Weights and cubins are immutable hardlinks. For INT8 storage, run through
 tools/operators/run.sh to compile write/read kernels before publishing.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.model.publication import atomic_model, clone_model, commit_package, load_model


def transform(wrapper, package):
    """Require identity page tables and preserve stable recipe slots."""
    meta = wrapper['metadata']
    buffers = {b['name']: b for b in meta['buffers']}
    if wrapper['architecture'] != 'qwen3_5' or not {'Kcontig', 'Vcontig', 'Pages'} <= buffers.keys():
        raise ValueError('Expected Qwen3_5 with contiguous FP16 KV workspace')
    removed = {'Kcontig', 'Vcontig'}
    kernels = []
    for binding in package['kernels']:
        names = {a.get('name') for a in binding['args'] if a['kind'] == 'buffer'}
        if 'Kcontig' in names and 'Pages' in names:
            if not re.fullmatch(r'prefill_m\d+/layer\d+/k4', binding['name']):
                raise ValueError('Unexpected KV gather recipe')
            continue
        if names & removed:
            layer = re.search(r'/layer(\d+)/k5$', binding['name'])
            if not layer:
                raise ValueError('Unexpected contiguous attention recipe')
            for arg in binding['args']:
                if arg.get('name') in removed:
                    arg['name'] = f'L{layer[1]}_' + ('KPages' if arg['name'] == 'Kcontig' else 'VPages')
        kernels.append(binding)
    package['kernels'] = kernels
    meta['buffers'] = [b for b in meta['buffers'] if b['name'] not in removed]
    for name in removed:
        wrapper['buffer_scopes'].pop(name)
    payloads = {b['name']: b for b in meta['buffers'] if b['name'].endswith(('KPages','VPages'))}
    if not payloads:
        raise ValueError('No paged KV buffers')
    for b in payloads.values():
        if b['dtype'] != 'f16' or b['shape'][1:] != [128,4,256] or b['shape'][0] * b['shape'][1] != meta['max_context']:
            raise ValueError('Invalid KV payload')
    meta['kv_cache'] = dict(direct_prefill=True, demand_mapping=True,
        buffers={name: __import__('math').prod(b['shape'][2:])*2 for name,b in payloads.items()})
    meta['reset_buffers'] = [n for n in meta['reset_buffers'] if n not in removed]
    return wrapper, package




def optimize(source, destination, engine, storage='fp16', output=None):
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Destination already exists')
    if destination.resolve().is_relative_to(source):
        raise ValueError('Destination must be outside the immutable source model')
    wrapper, old_package, package = load_model(source)
    # Demand mapping and direct prefill require token-major identity page order.
    completed = subprocess.run([str(engine.resolve()), 'plan-model', str(source)],
                               check=True, capture_output=True, text=True)
    original = json.loads(completed.stdout)['manifest']
    # Read just the safetensors header and Pages payload, without loading weights.
    index = json.loads((source/'cache/weights/model.safetensors.index.json').read_text())
    import struct
    page_buffer = next(b for b in original['buffers'] if b['name']=='Pages')
    shard = source/'cache/weights'/index['weight_map'][page_buffer['data']['tensor']]
    with shard.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length))
        tensor = header[page_buffer['data']['tensor']]
        start, stop = tensor['data_offsets']
        stream.seek(8+length+start)
        pages = stream.read(stop-start)
    expected = struct.pack('<'+'i'*(original['max_context']//128), *range(original['max_context']//128))
    if pages != expected or hashlib.sha256(pages).hexdigest()!=page_buffer['data']['sha256']:
        raise ValueError('Direct KV requires a verified identity page table')
    wrapper, package = transform(wrapper, package)
    with atomic_model(destination, engine, command='validate-model') as staging:
        pkg = clone_model(source, staging, old_package)
        # Break the package.json hardlink before writing new contracts.
        (pkg/'package.json').unlink()
        if storage == 'int8':
            if output is None: raise ValueError('INT8 export needs --output')
            quantize_package(wrapper, package, old_package, pkg, output, json.loads((source / 'config.json').read_text()))
        package['buffer_contracts'] = [{k:v for k,v in b.items() if k!='data'} for b in wrapper['metadata']['buffers']]
        digest = commit_package(staging, pkg, wrapper, package)
    return dict(model=str(destination),operator_package=digest,storage=storage,
                kv_capacity_bytes=sum(wrapper['metadata']['kv_cache']['buffers'].values())*original['max_context'])


def quantize_package(wrapper, package, old_package, destination, output, config):
    from tools.operators.common import configure, export_kernel
    from tools.operators.abi import parse_host, evaluate
    from kernels.model.kv_int8 import (full_prepare_mrope_int8,
        attention_prefill_int8, paged_attention_partials_int8)
    configure()
    meta = wrapper['metadata']
    context = meta['max_context']
    pages = context // 128
    text = config['text_config']
    if (text['num_attention_heads'],text['num_key_value_heads'],text['head_dim'],text['rms_norm_eps']) != (24,4,256,1e-6) or text['rope_parameters']['partial_rotary_factor'] != .25:
        raise ValueError('INT8 KV kernels require the 27B attention/norm/RoPE geometry')
    sections = tuple(text['rope_parameters']['mrope_section'])
    payloads = [b for b in meta['buffers'] if b['name'].endswith(('KPages','VPages'))]
    for b in payloads:
        b['dtype'] = 'i8'
        b['layout'] = 'paged-token-head-dim-int8-group64'
        meta['kv_cache']['buffers'][b['name']] //= 2
        scale = copy.deepcopy(b)
        scale['name'] = b['name'] + 'Scale'
        scale['dtype'] = 'f16'
        scale['shape'][-1] //= 64
        scale['layout'] = 'paged-token-head-group64-scale-fp16'
        meta['buffers'].append(scale)
        meta['reset_buffers'].append(scale['name'])
        wrapper['buffer_scopes'][scale['name']] = 'sequence'
        meta['kv_cache']['buffers'][scale['name']] = 32
    compiled = {}
    for i, binding in enumerate(package['kernels']):
        names = {a.get('name') for a in binding['args'] if a['kind']=='buffer'}
        if not any(n and n.endswith('KPages') for n in names): continue
        rows = int(re.search(r'_m(\d+)',binding['name'])[1]) if '_m' in binding['name'] else 1
        if 'MRopePositions' in names: key = ('prepare',)
        elif 'Pages' in names: key = ('decode', None if binding['name'].startswith('decode/') else rows)
        else: key = ('prefill',rows)
        if key not in compiled:
            print('compile INT8 KV',key,flush=True)
            if key[0]=='prepare': kernel=full_prepare_mrope_int8(pages,context,sections,max_position=context)
            elif key[0]=='decode': kernel=paged_attention_partials_int8(pages,pages,queries=key[1])
            else: kernel=attention_prefill_int8(1,rows,context,kv_layout='token_major',block_m=32 if rows==512 else 64)
            export = output/('-'.join(map(str,key)))
            export_kernel(kernel,export)
            assets = {}
            for name,file in [('source','kernel.cu'),('host_abi','host.txt'),('module','kernel.cubin')]:
                data=(export/file).read_bytes(); digest=hashlib.sha256(data).hexdigest()
                relative='kernels/'+digest+Path(file).suffix
                if not (destination/relative).exists(): (destination/relative).write_bytes(data)
                assets[name]=dict(file=relative,sha256=digest)
            compiled[key]=(assets,(export/'host.txt').read_text())
        assets,new_host=compiled[key]
        old=parse_host((old_package/binding['host_abi']['file']).read_text())[0]
        new=parse_host(new_host)[0]
        arguments={formal['value']:actual for formal,actual in zip(old['ordered_arguments'],binding['args'])}
        kname=next(n for n in names if n and n.endswith('KPages'))
        vname=next(n for n in names if n and n.endswith('VPages'))
        arguments['KS.data_ptr()']=dict(kind='buffer',name=kname+'Scale')
        arguments['VS.data_ptr()']=dict(kind='buffer',name=vname+'Scale')
        dims=dict(rows=rows,batch=1,contexts=1,tokens=context,pages=pages)
        updated=copy.deepcopy(binding)
        updated['args']=[]
        for formal in new['ordered_arguments']:
            expression=formal['value']
            if expression.endswith('.data_ptr()'):
                updated['args'].append(arguments[expression])
            else:
                if formal['ctype'] not in ('ctypes.c_int','ctypes.c_int32'): raise ValueError('Unsupported scalar')
                updated['args'].append(dict(kind='i32',value=int(evaluate(expression,dims))))
        launch=new['launch_expressions']
        updated['symbol']=new['symbol']
        for field,prefix in [('grid','gridDim'),('block','blockDim')]:
            updated[field]=[int(evaluate(launch[prefix+a],dims)) for a in 'XYZ']
        updated['shared_memory_bytes']=int(evaluate(launch['sharedMemBytes'],dims))
        updated.update(assets)
        package['kernels'][i]=updated


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--engine',type=Path,default=Path('target/release/orin-llm'))
    p.add_argument('--storage',choices=['fp16','int8'],default='fp16')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    print(json.dumps(optimize(a.model,a.destination,a.engine,a.storage,a.output),indent=2))

if __name__=='__main__': main()
