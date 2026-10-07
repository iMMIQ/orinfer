"""Publish dense SM87 prefill pipelines and larger native MTP warm profiles.

Only operator bindings and bounded workspaces change; resident weights retain
their existing single W4 representation. The source directory stays immutable.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.model.publication import atomic_model, clone_model, commit_package, load_model
from tools.model.screen_prefill_ffn import model_identity
from tools.operators.abi import parse_host, evaluate


def publish(source, destination, engine, output, warm_sizes=(64,128,512), attention_screen=None):
    from tools.operators.common import configure, export_kernel
    from kernels.model.attention_prefill_staged import attention_prefill_staged
    from kernels.model.kv_int8 import dequant_prefill_kv
    from kernels.model.w4_small_m import w4_small_m
    from kernels.model import speculation, control
    configure()
    source=source.resolve(strict=True);destination=destination.absolute()
    if destination.exists() or destination.resolve().is_relative_to(source):
        raise ValueError('Destination must be new and outside source')
    wrapper, old, package = load_model(source)
    meta = wrapper['metadata']
    if wrapper['architecture']!='qwen3_5' or not meta['kv_cache'].get('prefill_workspace'):
        raise ValueError('Requires prepared demand-mapped INT8 KV with prefill scratch')
    context=meta['max_context'];spec=meta['mtp'];buffers={b['name']:b for b in meta['buffers']}
    attention_choice=None
    if attention_screen is not None:
        screen=json.loads((attention_screen/'result.json').read_text())
        selected=[r for r in screen['cases'] if r.get('selected') and r.get('finalist_rechecked')]
        if (screen['status']!='passed' or screen['fingerprint']!=model_identity(source)
                or screen['capacity']!=context or screen['rows']!=2048 or len(selected)!=1):
            raise ValueError('Attention screen must validate this model and the 2048-row capacity ABI')
        attention_choice=selected[0]
        if ({r['length'] for r in attention_choice['metadata_checks']}!={0,1,145,1025,screen['context']}
                or not all(r['guard'] and r['empty_query'] and r['error']['finite']
                           and r['error']['relative_l2']<.002 for r in attention_choice['metadata_checks'])
                or not attention_choice['graph_restore_error']['finite']
                or attention_choice['graph_restore_error']['relative_l2']>=.002):
            raise ValueError('Incomplete attention metadata/graph checks')
    if buffers['FullQ']['shape'][1:] != [24,256]:
        raise ValueError('This attention package requires Q24/KV4, head_dim=256')
    if spec and warm_sizes:
        geometry={'MtpEmbedding':[5120],'MtpFullX':[14336],'MtpGateUpResult':[34816]}
        if any(buffers[n]['shape'][1:] != shape for n,shape in geometry.items()):
            raise ValueError('MTP warm profiles require the 27B projection geometry')
    kernels = {k['name']: k for k in package['kernels']}
    with atomic_model(destination, engine, command='validate-model') as staging:
        pkg = clone_model(source, staging, old)
        exports={}
        def exported(key,factory):
            if key not in exports:
                print('compile',key,flush=True)
                directory=output/key;export_kernel(factory(),directory);assets={}
                for field,file in [('module','kernel.cubin'),('source','kernel.cu'),('host_abi','host.txt')]:
                    data=(directory/file).read_bytes();digest=hashlib.sha256(data).hexdigest()
                    path='kernels/'+digest+Path(file).suffix
                    if not (pkg/path).exists():(pkg/path).write_bytes(data)
                    assets[field]=dict(file=path,sha256=digest)
                exports[key]=(assets,parse_host((directory/'host.txt').read_text())[0])
            return exports[key]
        def arguments(binding):
            host=parse_host((old/binding['host_abi']['file']).read_text())[0]
            return {f['value']:a['name'] for f,a in zip(host['ordered_arguments'],binding['args']) if a['kind']=='buffer'}
        def bind(name,export,args,rows):
            assets,host=export;dims=dict(rows=rows,M=rows,batch=1)
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
        def attention(rows,bm):
            if rows==2048 and attention_choice is not None:
                return exported('attention-'+str(rows),lambda:attention_prefill_staged(
                    1,rows,context,kv_layout='token_major',block_m=attention_choice['bm'],
                    block_n=attention_choice['bn'],threads=attention_choice['threads'],
                    num_stages=attention_choice['stages'],exp_mode=attention_choice['exp_mode'],
                    interior_mask=True,contiguous_queries=True))
            return exported('attention-'+str(rows),lambda:attention_prefill_staged(1,rows,context,kv_layout='token_major',block_m=bm,num_stages=1,interior_mask=True,contiguous_queries=True))
        pad=attention_choice['bn'] if attention_choice is not None else 32
        dq=exported('dequant-pad'+str(pad),lambda:dequant_prefill_kv(context,pad_to=pad))
        for name,binding in list(kernels.items()):
            if name.startswith(('prefill_m512/layer','prefill_m2048/layer')) and name.endswith('/k5') and any(a.get('name')=='FullQ' for a in binding['args']):
                rows=int(name.split('/')[0][9:]);kernels[name]=bind(name,attention(rows,64),arguments(binding),rows)
                dname=name[:-1]+'4';kernels[dname]=bind(dname,dq,arguments(kernels[dname]),rows)
        if spec and warm_sizes:
            if not spec.get('hidden_ring'):raise ValueError('Requires bounded MTP hidden ring')
            maximum=max(warm_sizes)
            if len(set(warm_sizes))!=len(warm_sizes) or any(n<32 or n%32 or n>meta['chunk_tokens'] for n in warm_sizes):raise ValueError('Warm sizes must be distinct multiples of 32 fitting the main chunk')
            for name in ['MtpInput','MtpPositions','MtpEmbedding','MtpCondition','MtpConcat','MtpHidden','MtpNorm','MtpR0','MtpR1','MtpFullX','MtpFullQ','MtpGate','MtpMixer','MtpMix','MtpGateUpResult','MtpActivated']:
                buffers[name]['shape'][0]=max(buffers[name]['shape'][0],maximum)
            buffers['MtpPartial']['shape'][1]=max(buffers['MtpPartial']['shape'][1],maximum)
            ring=buffers[spec['hidden_ring']]['shape'][0]
            op=lambda name:importlib.import_module('kernels.operators.op'+name)
            for rows in warm_sizes:
                if any(p['tokens']==rows for p in spec['warm_plans']):raise ValueError('Warm profile already exists')
                replacements={
                    0:('gather',lambda:speculation.gather_target_hidden(rows,5120,ring,ring=True)),
                    1:('prepare',lambda:control.prepare(rows)),
                    4:('fc',lambda:w4_small_m(rows,5120,10240,8,'float32',TILE_N=128)),
                    5:('merge',lambda:op('32_split_k_merge').split_k_merge(rows)),
                    6:('norm',lambda:op('02_residual_norm').residual_norm(rows,residual_dtype='float32',output_residual_dtype='float32')),
                    7:('in',lambda:w4_small_m(rows,14336,5120,TILE_N=128)),
                    9:('dequant',None),10:('attention',None),
                    11:('out',lambda:w4_small_m(rows,5120,6144,8,'float32',TILE_N=128)),
                    12:('merge',lambda:op('32_split_k_merge').split_k_merge(rows)),
                    13:('norm',lambda:op('02_residual_norm').residual_norm(rows,residual_dtype='float32',output_residual_dtype='float32')),
                    14:('gateup',lambda:w4_small_m(rows,34816,5120,TILE_N=128)),
                    16:('down',lambda:w4_small_m(rows,5120,17408,8,'float32',TILE_N=64)),
                    17:('merge',lambda:op('32_split_k_merge').split_k_merge(rows)),
                    18:('advance',lambda:control.advance(rows)),
                }
                for index in range(19):
                    template=kernels[f'mtp_warm_m16/body/k{index}'];name=f'mtp_warm_m{rows}/body/k{index}';args=arguments(template)
                    if index==9:
                        args={k+'.data_ptr()':v for k,v in dict(K='MtpKPages',V='MtpVPages',KS='MtpKPagesScale',VS='MtpVPagesScale',Lengths='MtpSeqLength',KO='PrefillK',VO='PrefillV').items()};export=dq
                    elif index==10:
                        args={k+'.data_ptr()':v for k,v in dict(Q='MtpFullQ',K='PrefillK',V='PrefillV',Gate='MtpGate',Positions='MtpPositions',Lengths='MtpSeqLength',Y='MtpMixer').items()};export=attention(rows,64 if rows>=128 else 32)
                    elif index in replacements:
                        key,factory=replacements[index];export=exported(f'mtp-{rows}-{key}',factory)
                    else:export=({k:template[k] for k in ['module','source','host_abi']},parse_host((old/template['host_abi']['file']).read_text())[0])
                    kernels[name]=bind(name,export,args,rows)
                for index in range(4):
                    template=kernels[f'mtp_head_m16/body/k{index}'];name=f'mtp_head_m{rows}/body/k{index}'
                    export=exported(f'mtp-{rows}-finalnorm',lambda:op('23_final_norm').final_norm(rows,1,last_row=True)) if index==0 else ({k:template[k] for k in ['module','source','host_abi']},parse_host((old/template['host_abi']['file']).read_text())[0])
                    kernels[name]=bind(name,export,arguments(template),1)
                spec['warm_plans'].append(dict(tokens=rows,program=f'mtp_warm_m{rows}',head_program=f'mtp_head_m{rows}'))
        package['kernels']=list(kernels.values());package['buffer_contracts']=[{k:v for k,v in b.items() if k!='data'} for b in meta['buffers']]
        digest = commit_package(staging, pkg, wrapper, package)
    return dict(model=str(destination),execution_package=digest,mtp_warm_sizes=list(warm_sizes),
                attention_choice=attention_choice)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--engine',type=Path,default=Path('target/release/orinfer'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warm-sizes',type=int,nargs='*',default=[64,128,512])
    p.add_argument('--attention-screen',type=Path)
    a=p.parse_args()
    print(json.dumps(publish(a.model,a.destination,a.engine,a.output,tuple(a.warm_sizes),a.attention_screen),indent=2))
