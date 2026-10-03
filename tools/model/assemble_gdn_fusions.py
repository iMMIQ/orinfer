"""Prefill-only measured WY and gatednorm/A8 overrides, one immutable model.

Actual generated ABI maps all arguments. The intermediate MixerIn survives
for full-attention prefill and decode; only the 48 checked GDN chains fuse.
"""
import argparse
import copy
import errno
import json
import os
from pathlib import Path
import shutil
from tools.operators.abi import evaluate,parse_host
from tools.model.assemble_gdn_precision import digest,identity


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--wy-exports',type=Path,required=True)
    ap.add_argument('--gatednorm-exports',type=Path,required=True)
    ap.add_argument('--gatednorm-threads',type=int,choices=(128,256,512),default=512)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();assert not args.output.exists(),args.output
    wy_report=json.loads((args.wy_exports/'result.json').read_text())
    norm_report=json.loads((args.gatednorm_exports/'result.json').read_text())
    assert wy_report['status']==norm_report['status']=='passed'
    assert wy_report['policy']=='high' and wy_report['flags']==[False,False]
    assert {c['T'] for c in wy_report['cases']} >= {512,513,2048,8192}
    assert {c['rows'] for c in norm_report['cases'] if c['threads']==args.gatednorm_threads} >= {512,513,2048,8192}
    source=args.model.parent;original=json.loads(args.model.read_text());model=copy.deepcopy(original)
    old_kernels={k['name']:k for k in model['kernels']}
    assert {p['chunk_tokens'] for p in model['prefill_plans']}=={512,2048,8192}
    args.output.mkdir(parents=True)
    hashes={};copied={}
    def preserve(ref):
        src=source/ref['file'];dst=args.output/ref['file']
        if src not in hashes:hashes[src]=digest(src)
        assert hashes[src]==ref['sha256'],src
        if ref['file'] in copied:
            assert copied[ref['file']]==ref['sha256'];return
        dst.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(src,dst)
        except OSError as error:
            if error.errno not in (errno.EPERM,errno.EACCES,errno.EXDEV):raise
            shutil.copyfile(src,dst);assert digest(dst)==ref['sha256']
        copied[ref['file']]=ref['sha256']
    for buffer in model['buffers']:
        if buffer.get('data'):preserve(buffer['data'])
    exports={}
    for name,root in [('wy_high',args.wy_exports/'BT64'),
                      ('gatednorm_a8',args.gatednorm_exports/f'threads{args.gatednorm_threads}')]:
        rel=Path('aot-gdn-fusions')/name;dest=args.output/rel;dest.mkdir(parents=True)
        refs={}
        for field,filename in [('module','kernel.cubin'),('source','kernel.cu'),('host_abi','host.txt')]:
            shutil.copyfile(root/filename,dest/filename)
            refs[field]=dict(file=str(rel/filename),sha256=digest(dest/filename))
        launch=parse_host((root/'host.txt').read_text());assert len(launch)==1
        exports[name]=(refs,launch[0])
    def bindings(old):
        launch=parse_host((source/old['host_abi']['file']).read_text());assert len(launch)==1
        order=launch[0]['ordered_arguments'];assert len(order)==len(old['args'])
        return {a['value']:bound for a,bound in zip(order,old['args'])}
    def check_mixer_consumers(ops):
        """Every MixerIn writer has exactly one immediate A8 consumer.

        Reject unknown reads, aliasing pointer arguments, and copy/zero nodes.
        Thus removing a GDN writer and its quantizer leaves no stale read;
        full-attention writers still initialize MixerIn before their A8 reads.
        """
        pending = None
        counts = {'gdn': 0, 'attention': 0, 'a8': 0}
        for index, op in enumerate(ops):
            if pending is not None:
                assert index == pending + 1 and op['kind'] == 'kernel'
            if op['kind'] != 'kernel':
                assert 'MixerIn' not in (op.get('source'), op.get('destination')), op
                continue
            kernel = old_kernels[op['name']]
            if not any(a.get('name') == 'MixerIn' for a in kernel['args']):
                assert pending is None, op
                continue
            family = Path(kernel['module']['file']).parent.name
            bound = bindings(kernel)
            accesses = [key for key, value in bound.items()
                        if value == dict(kind='buffer', name='MixerIn')]
            if family == 'gatednorm' or family.startswith('attentionprefill_staged'):
                assert pending is None and accesses == ['Y.data_ptr()'], op
                pending = index
                counts['gdn' if family == 'gatednorm' else 'attention'] += 1
            else:
                assert family == 'a8_6144' and pending is not None, op
                assert accesses == ['X.data_ptr()'], op
                pending = None
                counts['a8'] += 1
        assert pending is None and counts == {'gdn': 48, 'attention': 16, 'a8': 64}, counts
        return counts
    added={};counts={}
    def replace(old,family,bound):
        refs,launch=exports[family];expr=launch['launch_expressions']
        variables={name:a['value'] for name,a in bound.items() if a['kind']=='i32'}
        name=old['name']+'_'+family
        new=dict(copy.deepcopy(refs),name=name,symbol=launch['symbol'],cooperative=False,
                 args=[copy.deepcopy(bound[a['value']]) for a in launch['ordered_arguments']],
                 grid=[evaluate(expr['gridDim'+axis],variables) for axis in 'XYZ'],
                 block=[evaluate(expr['blockDim'+axis],variables) for axis in 'XYZ'],
                 shared_memory_bytes=evaluate(expr['sharedMemBytes'],variables))
        if name in added:assert added[name]==new
        else:added[name]=new
        return dict(kind='kernel',name=name)
    for phase,ops in model['programs'].items():
        if not phase.startswith('prefill'):continue
        check_mixer_consumers(ops)
        result=[];index=0;counts[phase]=dict(wy=0,gatednorm_a8=0)
        while index<len(ops):
            op=ops[index]
            old=old_kernels[op['name']] if op['kind']=='kernel' else None
            family=Path(old['module']['file']).parent.name if old else None
            if family=='wy_compensated32':
                result.append(replace(old,'wy_high',bindings(old)))
                counts[phase]['wy']+=1;index+=1
            elif family=='gatednorm':
                following=ops[index+1];assert following['kind']=='kernel'
                quant=old_kernels[following['name']]
                assert Path(quant['module']['file']).parent.name=='a8_6144'
                first,second=bindings(old),bindings(quant)
                assert first['Y.data_ptr()']==second['X.data_ptr()']==dict(kind='buffer',name='MixerIn')
                assert first['rows']==second['rows']
                bound={k:v for k,v in first.items() if k!='Y.data_ptr()'}
                bound.update({k:second[k] for k in ('Q.data_ptr()','S.data_ptr()')})
                result.append(replace(old,'gatednorm_a8',bound))
                counts[phase]['gatednorm_a8']+=1;index+=2
            else:result.append(op);index+=1
        assert counts[phase]==dict(wy=48,gatednorm_a8=48),counts
        model['programs'][phase]=result
    used={op['name'] for ops in model['programs'].values() for op in ops if op['kind']=='kernel'}
    model['kernels']=[k for k in model['kernels'] if k['name'] in used]+list(added.values())
    for kernel in model['kernels']:
        for field in ('module','source','host_abi'):
            if not kernel[field]['file'].startswith('aot-gdn-fusions/'):preserve(kernel[field])
    assert model['buffers']==original['buffers'] and model['weight_bytes']==original['weight_bytes']
    assert model['programs']['decode']==original['programs']['decode']
    decode_names={op['name'] for op in original['programs']['decode'] if op['kind']=='kernel'}
    assert {k['name']:k for k in model['kernels'] if k['name'] in decode_names}=={
        k['name']:k for k in original['kernels'] if k['name'] in decode_names}
    manifest=args.output/'model.json';manifest.write_text(json.dumps(model,indent=2)+'\n')
    record=dict(status='assembled',source_manifest=identity(args.model),manifest=identity(manifest),
                wy_report=identity(args.wy_exports/'result.json'),gatednorm_report=identity(args.gatednorm_exports/'result.json'),
                changes=counts,weight_bytes=model['weight_bytes'],buffers_and_decode_identical=True,
                scope='WY high FP16 operands with FP32 W/U; fused GDN gatednorm/A8, '
                      'same weights/residency; model performance/quality pending')
    (args.output/'assembly.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copyfile(__file__,args.output/Path(__file__).name)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
