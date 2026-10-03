"""Offline single-W4 FFN LUT4/I8-layout candidate from a multi-plan manifest.

Reuses measured AOT exports: short-M LUT4, long-M strict row-W8 expansion,
and production split-K M1 reader. No Python or permanent W8 online. Original
artifacts stay immutable; unused legacy kernels are removed from this manifest.
"""
import argparse
import copy
import errno
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

from tools.operators.abi import evaluate, parse_host
from tools.quantization.w4_i8_pack import LAYOUT, pack_array
from tools.quantization.w4_i8_lut4 import prepare


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--short-exports',type=Path,required=True)
    ap.add_argument('--expand-exports',type=Path,required=True)
    ap.add_argument('--decode-exports',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resume-weights',action='store_true',help='Recover completed packed weights after a terminal assembly failure')
    args=ap.parse_args()
    previous=None
    if args.resume_weights:
        previous=json.loads((args.output/'assembly.json').read_text())
        assert previous['status']=='building' and len(previous['weights'])==128
        assert previous['source_manifest_sha256']==digest(args.model)
    else:
        assert not args.output.exists(),args.output
        args.output.mkdir(parents=True)
    source=args.model.parent
    model=json.loads(args.model.read_text())
    assert {x['chunk_tokens'] for x in model['prefill_plans']}=={512,2048,8192}
    old_buffers={b['name']:copy.deepcopy(b) for b in model['buffers']}
    old_kernels={k['name']:k for k in model['kernels']}
    records=[]

    def progress():
        (args.output/'assembly.json').write_text(json.dumps(dict(status='building',
            source_manifest_sha256=digest(args.model),weights=records),indent=2)+'\n')

    def original_array(name,dtype):
        b=old_buffers[name];p=source/b['data']['file']
        assert digest(p)==b['data']['sha256'],p
        return np.fromfile(p,dtype=dtype).reshape(b['shape'])

    def write_array(name,a,layout):
        rel=Path('native-i8')/(name+'.bin');p=args.output/rel;p.parent.mkdir(exist_ok=True)
        a.tofile(p)
        b=dict(name=name,dtype={np.dtype('int32'):'i32',np.dtype('int8'):'i8'}[a.dtype],
               shape=list(a.shape),layout=layout,alignment=256,access='read',
               data=dict(file=str(rel),sha256=digest(p)))
        return b

    def existing_array(name,dtype,shape,layout):
        rel=Path('native-i8')/(name+'.bin');p=args.output/rel
        assert p.stat().st_size==math.prod(shape)*np.dtype(dtype).itemsize,p
        return dict(name=name,dtype='i32' if dtype==np.int32 else 'i8',shape=list(shape),
                    layout=layout,alignment=256,access='read',data=dict(file=str(rel),sha256=digest(p)))

    replacements={}
    for layer in range(64):
        for kind in ['GateUp','Down']:
            base=f'L{layer}_{kind}'
            if previous is not None:
                record=previous['weights'][len(records)]
                assert record['weight']==base and record['source_sha256']==old_buffers[base+'_P']['data']['sha256']
                replacements[base+'_P']=existing_array(base+'_P',np.int32,old_buffers[base+'_P']['shape'],LAYOUT)
                assert replacements[base+'_P']['data']['sha256']==record['candidate_sha256']
                shape=old_buffers[base+'_S']['shape']
                model['buffers'].extend([existing_array(base+'_LUT4',np.int32,shape,'lut4_row_group128_u8x4'),
                                         existing_array(base+'_Step',np.int8,shape,'lut4_row_group128_u8_step')])
                records.append(record);progress();continue
            p=original_array(base+'_P',np.int32)
            assert old_buffers[base+'_P']['layout']=='u4_warp_n64_k128_mma_f16'
            native=pack_array(p,verify=True).view(np.int32)
            replacements[base+'_P']=write_array(base+'_P',native,LAYOUT)
            s=original_array(base+'_S',np.float16)
            z=original_array(base+'_Z',np.int8)
            ws=original_array(base+'_WS',np.float16)
            table,step,reference,approximate=prepare(s,z,ws)
            model['buffers'].extend([write_array(base+'_LUT4',table.view(np.int32),'lut4_row_group128_u8x4'),
                                     write_array(base+'_Step',step.view(np.int8),'lut4_row_group128_u8_step')])
            records.append(dict(weight=base,packed_bytes=native.nbytes,added_metadata_bytes=table.nbytes+step.nbytes,
                                packed_roundtrip=True,source_sha256=old_buffers[base+'_P']['data']['sha256'],
                                candidate_sha256=replacements[base+'_P']['data']['sha256']))
            progress();print(json.dumps(records[-1]),flush=True)
            del p,native,s,z,ws,table,step,reference,approximate
    model['buffers']=[replacements.get(b['name'],b) for b in model['buffers']]

    def link_file(original,relative,expected):
        p=args.output/relative;p.parent.mkdir(parents=True,exist_ok=True)
        assert digest(original)==expected,original
        if not p.exists():
            try:os.link(original,p)
            except OSError as error:
                if error.errno not in (errno.EPERM,errno.EACCES,errno.EXDEV):raise
                shutil.copyfile(original,p)
        else:assert digest(p)==expected,p

    for b in model['buffers']:
        if b.get('data') and not b['data']['file'].startswith('native-i8/'):
            link_file(source/b['data']['file'],b['data']['file'],b['data']['sha256'])

    added=[]
    added_by_name={}
    copied_exports={}
    def new_kernel(old,kind,base,bindings):
        family='GateUp' if base.endswith('_GateUp') else 'Down'
        root={'lut4':args.short_exports/f'L0_{family}-m256n128s2',
              'expand':args.expand_exports/f'L0_{family}',
              'decode':args.decode_exports/f'L0_{family}'}[kind]
        key=(kind,family)
        if key not in copied_exports:
            rel=Path('aot-native')/(kind+'-'+family)
            dest=args.output/rel;dest.mkdir(parents=True)
            refs={}
            for field,file in [('module','kernel.cubin'),('source','kernel.cu'),('host_abi','host.txt')]:
                shutil.copyfile(root/file,dest/file)
                refs[field]=dict(file=str(rel/file),sha256=digest(dest/file))
            launches=parse_host((root/'host.txt').read_text());assert len(launches)==1
            launch=launches[0];expr=launch['launch_expressions']
            template=dict(**refs,symbol=launch['symbol'],grid=[evaluate(expr['gridDim'+s],{}) for s in 'XYZ'],
                block=[evaluate(expr['blockDim'+s],{}) for s in 'XYZ'],
                shared_memory_bytes=evaluate(expr['sharedMemBytes'],{}),cooperative=False)
            order=[]
            for a in launch['ordered_arguments']:
                assert a['ctype']=='ctypes.c_void_p' and a['value'].endswith('.data_ptr()'),a
                order.append(a['value'].removesuffix('.data_ptr()'))
            copied_exports[key]=(template,order)
        template,order=copied_exports[key]
        name=old['name']+'_'+kind+'_native'
        k=dict(copy.deepcopy(template),name=name,args=[dict(kind='buffer',name=bindings[x]) for x in order])
        if kind in ['expand','decode']:
            assert k['grid']==old['grid'] and k['block']==old['block']
        if name in added_by_name:assert added_by_name[name]==k
        else:added.append(k);added_by_name[name]=k
        return dict(kind='kernel',name=name)

    counts={}
    for phase,ops in model['programs'].items():
        result=[];i=0;counts[phase]=dict(fused=0,expand=0,decode=0)
        while i<len(ops):
            op=ops[i];k=old_kernels.get(op.get('name')) if op['kind']=='kernel' else None
            packed=[a['name'] for a in k['args'] if a['kind']=='buffer' and
                    (a['name'].endswith('_GateUp_P') or a['name'].endswith('_Down_P'))] if k else []
            if not packed:result.append(op);i+=1;continue
            assert len(packed)==1
            base=packed[0].removesuffix('_P')
            bindings=dict(PP=base+'_P',S=base+'_S',Z=base+'_Z',WS=base+'_WS')
            if 'w8_expand' in k['name']:
                if phase=='prefill_m512':
                    following=old_kernels[ops[i+1]['name']]
                    assert following['symbol']=='kernel_kernel'
                    names=[a['name'] for a in following['args']]
                    assert names[:4]==['TemporaryA8','TemporaryAS','TemporaryW8',base+'_WS']
                    bindings.update(A='TemporaryA8',AS='TemporaryAS',C=names[4],Z=base+'_Step',Coef=base+'_LUT4')
                    result.append(new_kernel(k,'lut4',base,bindings));i+=2;counts[phase]['fused']+=1
                else:
                    bindings['W8']='TemporaryW8'
                    result.append(new_kernel(k,'expand',base,bindings));i+=1;counts[phase]['expand']+=1
            elif 'decode_gateup' in k['name'] or 'decode_down' in k['name']:
                bindings.update(A='Norm' if base.endswith('_GateUp') else 'Activated',
                                O='GateUp' if base.endswith('_GateUp') else 'Partial')
                result.append(new_kernel(k,'decode',base,bindings));i+=1;counts[phase]['decode']+=1
            else:raise ValueError(f'Unhandled native-layout reader: {phase}/{k["name"]}')
        model['programs'][phase]=result
    assert counts['prefill_m512']['fused']==128 and counts['decode']['decode']==128,counts
    for p in ['prefill','prefill_m2048','prefill_m8192']:assert counts[p]['expand']==128,counts
    used={op['name'] for ops in model['programs'].values() for op in ops if op['kind']=='kernel'}
    model['kernels']=[k for k in model['kernels'] if k['name'] in used]+added
    assert len({k['name'] for k in model['kernels']})==len(model['kernels'])
    for k in model['kernels']:
        for field in ['module','source','host_abi']:
            r=k[field]
            if not r['file'].startswith('aot-native/'):link_file(source/r['file'],r['file'],r['sha256'])
    sizes={'i8':1,'u8':1,'f16':2,'i32':4,'f32':4,'i64':8}
    model['weight_bytes']=sum(math.prod(b['shape'])*sizes[b['dtype']] for b in model['buffers'] if b['access']=='read')
    added_bytes=sum(r['added_metadata_bytes'] for r in records)
    assert model['weight_bytes']==json.loads(args.model.read_text())['weight_bytes']+added_bytes
    model['weight_scope']='Single W4 copy; native I8-fragment FFN packing with original S/Z/WS; FFN LUT4 short-M metadata counted; strict temporary row-W8/A8 long-M; M1 W4A16 byte-permute vector reader, down split-K8; embedding/head unchanged; not BF16/FP8 quality acceptance'
    (args.output/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    (args.output/'assembly.json').write_text(json.dumps(dict(status='assembled',source_manifest_sha256=digest(args.model),
        manifest_sha256=digest(args.output/'model.json'),weights=records,program_changes=counts,
        added_metadata_bytes=added_bytes,weight_bytes=model['weight_bytes'],
        effective_weight_bits=8*model['weight_bytes']/model['weight_parameters'],
        scope='Offline assembled candidate; model execution and quality unverified'),indent=2)+'\n')
    shutil.copyfile(__file__,args.output/'assemble_lut4.py')
    for filename in ['tools/quantization/w4_i8_pack.py','tools/quantization/w4_i8_lut4.py','tools/operators/abi.py']:
        shutil.copyfile(filename,args.output/Path(filename).name)
    print(json.dumps(dict(status='assembled',weight_bytes=model['weight_bytes'],added_metadata_bytes=added_bytes)),flush=True)


if __name__=='__main__':main()
