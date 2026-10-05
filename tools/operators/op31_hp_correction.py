"""Offline HP primitive and complete masked-A8/W4->W8 chain validation."""
import argparse
import gc
import re
import shutil
import sys
import time
from pathlib import Path
from tools.reference import ACTIVATIONS as REFERENCE_ACTIVATIONS

import torch
from common import benchmark, configure, environment, error, export_kernel, identity, tensor_sha, write_json
from abi import parse_host
from kernels.operators.op31_hp_correction import hp_correction, launch, validate_indices, masked_base_gemm
from kernels.operators.op30_activation_quantization import activation_quantization, launch as quant_launch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'kernels/projections'))
sys.path.insert(0, str(ROOT / 'tools/projections'))
import candidates
import screen
ROWS = (1,2,3,4,5,7,8,511,512,513,2048,8192)


def export(kernel, directory, metadata, report):
    output = export_kernel(kernel, directory)
    host=(directory/'host.txt').read_text(); cuda=(directory/'kernel.cu').read_text()
    entry=re.search(r'extern "C" __global__ void (\w+)\(([^;]+)\);',cuda)
    info={**metadata,'sm':87,'toolchain':report['environment'],'exports':output,
          'entry_symbol':entry.group(1),'ordered_arguments':[x.strip() for x in entry.group(2).split(',')],
          'actual_host_launches':parse_host(host),'cooperative':False,'dynamic_M':True,
          'stream':'explicit caller current capture stream','workspace_bytes':0}
    write_json(directory/'abi.json',info)
    report['kernels'].append(info)


def stream(): return torch.cuda.current_stream().cuda_stream

def reference(a,idx,side,base,dtype):
    return (base + a[:,idx.long()].float() @ side.float().T).to(dtype)


def checked(out,ref):
    e=error(out,ref)
    assert e['finite'] and e['relative_l2']<=.002,e
    return e


def hp_run(kernel,a,idx,side,base,out):
    launch(kernel,a,idx,side,base,out,stream=stream(),base_is_masked=True)


def mutations(graph,run,a,idx,side,base,out):
    original=[x.clone() for x in (a,idx,side,base)]
    expected=out.clone(); results={}
    def check(name):
        out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
        results[name]=checked(out,reference(a,idx,side,base,out.dtype))
    a.mul_(-.75);check('A');a.copy_(original[0])
    side.mul_(-.5);check('W_hp');side.copy_(original[2])
    if idx.numel():
        idx.copy_(torch.remainder(idx+1,a.shape[1]));validate_indices(idx,a.shape[1]);check('Idx');idx.copy_(original[1])
    base.add_(.125);check('Base');base.copy_(original[3])
    out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
    assert torch.equal(out,expected),'graph restore not deterministic'
    # Unrelated invocation on independent output, then original replay.
    other=torch.empty_like(out);hp_run(run,a,idx,side,base,other)
    out.zero_();graph.replay();torch.cuda.synchronize();assert torch.equal(out,expected)
    results['poison_restore_and_independent_output']=True
    return results


def captures(layer,report):
    folder=REFERENCE_ACTIVATIONS; records=[]
    for path in sorted(folder.glob('*.json')):
        import json
        m=json.loads(path.read_text())
        if f'layers.{layer}.' in m['kind'] and m['kind'].endswith('down_proj'):
            p=folder/m['file'];t=torch.load(p,map_location='cpu',weights_only=True).half()
            src={'metadata':identity(path),'file':identity(p),'tensor_sha256':tensor_sha(t),'mode':m['mode'],'computed_tokens_before':m['computed_tokens_before']}
            assert src['file']['sha256']==m['file_sha256'] and src['tensor_sha256']==m['tensor_sha256']
            report['input_sources'].append(src);records.append((m,t))
    pre=next(t for m,t in records if m['mode']=='prefill' and len(t)==512)
    dec=torch.cat([t for m,t in sorted(records,key=lambda x:x[0]['computed_tokens_before']) if m['mode']=='decode'][:8])
    return pre,dec


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    configure();outdir=args.output
    report={'environment':environment(),'scope':'isolated HP correction and two-layer full down chain; no whole-model quality acceptance',
            'checkpoint':{'path':screen.MODEL,'full_sha256':screen.checkpoint_sha256(screen.MODEL),'hash_origin':'Supplied checkpoint hashed once; read tensor hashes verified'},
            'weight_reference':'same community AWQ W4 half((q-z)*s), not official BF16 weights',
            'primitive':[],'edge_cases':[],'chains':[],'kernels':[],'input_sources':[],'layers':[],
            'quality_limitation':'historical layer0 M512 position9 target271 probability .770 -> .0635; HP32 must not enable/accept model policy',
            'budget':{'down_512_ms':1.8,'down_2048_ms':7.2,'down_8192_ms':28.8,'historical_extra_512_ms':.322}}
    def save():write_json(outdir/'results.json',report)
    dependencies=['kernels/projections/candidates.py','tools/projections/screen.py','kernels/operators/op30_activation_quantization.py','kernels/operators/op04_swiglu.py','tools/operators/abi.py']
    report['frozen_dependencies']=[]
    for name in dependencies:
        dest=outdir/'measurement-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        report['frozen_dependencies'].append(identity(dest))
    kernels={}
    started=time.perf_counter()
    for dtype in ('float16','float32'):
        for small in (True,False):
            k=hp_correction(5120,17408,32,output_dtype=dtype,small=small)
            kernels[dtype,small]=k
            export(k,outdir/'compiled'/f'hp32-{dtype}-{"small" if small else "large"}',{'kind':'hp','N':5120,'K':17408,'O':32,'output':dtype,'base':'float32','small':small},report)
    report['initial_compile_s']=time.perf_counter()-started;save()
    for layer in (0,32):
        started=time.perf_counter();p,s,z,w,ids,cost=screen.load_weights('down',layer)
        pre,dec=captures(layer,report)
        historical=ROOT/f'artifacts/activation-round2/hp32-layer{layer}.pt'
        hp=torch.load(historical,map_location='cpu',weights_only=True)
        idx=hp['indices'].cuda();validate_indices(idx,17408);side=w[:,idx.long()].contiguous()
        assert torch.equal(side.cpu(),hp['weight']) and torch.equal(hp['mask'].nonzero().flatten().to(torch.int32),idx.cpu().sort().values)
        mask=hp['mask'].cuda()
        layer_row={'layer':layer,'weight_tensors':ids,'offline_preparation':cost,'historical_hp_file':identity(historical),
                   'indices':idx.cpu().tolist(),'side_sha256':tensor_sha(side),'mask_sha256':tensor_sha(mask),
                   'persistent_hp_bytes':side.numel()*2+idx.numel()*4+mask.numel(),'logical_W4_bytes':p.numel()+s.numel()*2+z.numel()}
        report['layers'].append(layer_row);save()
        for m in ROWS:
            if m<=8:cpua=dec[:m].contiguous();origin='consecutive real M1 decode rows, not concurrent model batch'
            elif m<=512:cpua=pre[:m].contiguous();origin='actual512 prefill rows, sliced for tail test' if m==511 else 'actual512 prefill'
            elif m==513:cpua=torch.cat([pre,dec[:1]]);origin='actual512prefill+firstdecode, tail fixture'
            else:cpua=pre.repeat(m//512,1);origin='repeat actual512 rows for shape/performance; not fresh long-context activation'
            a=cpua.cuda();base=torch.randn((m,5120),device='cuda',dtype=torch.float32)*.01
            for dtype in ('float16','float32'):
                kernel=kernels[dtype,m<511];out=torch.empty_like(base,dtype=getattr(torch,dtype))
                run=lambda:hp_run(kernel,a,idx,side,base,out)
                start=time.perf_counter();run();torch.cuda.synchronize()
                first=(time.perf_counter()-start)*1000
                ref=reference(a,idx,side,base,out.dtype);e=checked(out,ref)
                timing,graph=benchmark(run,repetitions=5 if m<=513 else 2,calls_per_replay=16)
                row={'layer':layer,'M':m,'origin':origin,'activation_sha256':tensor_sha(cpua),'output_dtype':dtype,
                     'base_scope':'independent FP32 primitive fixture; full masked GEMM evaluated separately',
                     'kernel_error':e,'timing':timing,'first_launch_sync_ms':first,'persistent_hp_bytes':layer_row['persistent_hp_bytes'],
                     'workspace_bytes':base.numel()*4+out.numel()*out.element_size()}
                if m in (3,513):row['graph_mutations']=mutations(graph,kernel,a,idx,side,base,out)
                else:
                    expected=out.clone();out.fill_(float('nan'));graph.replay();torch.cuda.synchronize();assert torch.equal(out,expected)
                    row['graph_poison_verified']=True
                report['primitive'].append(row);save();print(f'primitive layer{layer} M{m} {dtype}: {timing["median_ms"]:.5f}ms L2 {e["relative_l2"]:.6g}',flush=True)
                del out,graph,ref
            del a,base
        # Full four-node chain, with fresh expansion per call and F32 base.
        bs=(w.float().abs().amax(-1)/127).clamp_min(2**-24).half();expanded=torch.empty_like(w,dtype=torch.int8)
        expand=candidates.expand_weight_q8_inline_lut(5120,17408)
        aq=activation_quantization(17408,masked=True)
        gemm=masked_base_gemm(5120,17408);gemm.adapter.kernels=dict(gemm.adapter.kernels)
        for tag,k in [('expand',expand),('masked-a8',aq),('masked-base-f32',gemm)]:
            if layer==0:export(k,outdir/'compiled'/tag,{'kind':tag,'N':5120,'K':17408},report)
        expand(p,s,z,bs,expanded)
        bw=expanded.float()*bs.float()[:,None];wf=w.float()
        for m in (512,2048):
            a=pre.repeat(m//512,1).cuda();qa=torch.empty_like(a,dtype=torch.int8);asc=torch.empty((m,1),device='cuda',dtype=torch.float16)
            base=torch.empty((m,5120),device='cuda',dtype=torch.float32);out=torch.empty_like(base,dtype=torch.float16)
            correct=kernels['float16',False]
            def expansion():expand.adapter.func(p,s,z,bs,expanded,stream=stream())
            def quant():quant_launch(aq,a,mask,qa,asc,stream=stream())
            def multiply():gemm.adapter.func(qa,expanded,asc,bs,base,stream=stream())
            def merge():hp_run(correct,a,idx,side,base,out)
            def chain():expansion();quant();multiply();merge()
            chain();torch.cuda.synchronize()
            ah=qa.float()*asc.float();hpterm=a[:,idx.long()].float()@side.float().T
            qref=ah@bw.T+hpterm;ref=a.float()@wf.T
            ce=checked(out,qref)
            row={'layer':layer,'M':m,'origin':'actual512prefill' if m==512 else 'repeat actual512 rows for 2K shape; not fresh2Kmodel trace',
                 'kernel_error':ce,'quantization_loss':error(qref,ref),'total_error':error(out,ref),
                 'A8_only_loss':error(ah@wf.T+hpterm,ref),
                 'W8_only_loss':error((a.float()*(mask==0)[None])@bw.T+hpterm,ref),
                 'kernel_count':4,'persistent_hp_bytes':layer_row['persistent_hp_bytes'],'persistent_W8_scale_bytes':bs.numel()*2,
                 'workspace_bytes':expanded.numel()+qa.numel()+asc.numel()*2+base.numel()*4+out.numel()*2}
            timing,graph=benchmark(chain,repetitions=5,calls_per_replay=4);row['full_path_timing']=timing
            for tag,fn in [('expand',expansion),('masked_a8',quant),('base_gemm',multiply),('correction',merge)]:row[tag+'_timing']=benchmark(fn,repetitions=5,calls_per_replay=4)[0]
            # Graph change A and mask -> base+HP math recomputed from actual codes.
            original=a.clone();oldmask=mask.clone();a.mul_(-.75);mask.fill_(1);mask[idx.long()]=1
            out.fill_(float('nan'));base.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            row['graph_A_Mask_changed']=checked(out,reference(a,idx,side,base,out.dtype))
            assert bool((qa==0).all()) and bool((base==0).all())
            a.copy_(original);mask.copy_(oldmask);out.fill_(float('nan'));graph.replay();torch.cuda.synchronize();row['graph_restored']=checked(out,qref)
            row['budget_ms']=report['budget'][f'down_{m}_ms'];row['budget_ratio']=timing['median_ms']/row['budget_ms']
            report['chains'].append(row);save();print(f'chain layer{layer} M{m}: {timing["median_ms"]:.5f}ms L2 {ce["relative_l2"]:.6g}',flush=True)
            del a,qa,asc,base,out,ah,hpterm,qref,ref,original,graph
        del p,s,z,w,wf,bw,expanded,side,idx,mask,bs
        gc.collect()
    edge_tests(outdir,report)
    report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();report['complete']=True;save()


def edge_tests(outdir,report):
    # Non-MMA aligned O and output N; NaN at column zero must never be read.
    for o in (0,1,15,17,33):
        k=hp_correction(133,67,o,output_dtype='float32',small=True)
        export(k,outdir/'compiled'/f'edge-O{o}',{'kind':'tail','N':133,'K':67,'O':o,'output':'float32'},report)
        idx=torch.arange(1,o+1,device='cuda',dtype=torch.int32);validate_indices(idx,67)
        a=torch.randn((5,67),device='cuda',dtype=torch.float16);a[:,0]=float('nan')
        side=torch.randn((133,o),device='cuda',dtype=torch.float16);base=torch.randn((5,133),device='cuda');out=torch.empty_like(base)
        for case in ('random','zero','extremes'):
            if case=='zero':side.zero_()
            if case=='extremes':a[:,1:]=torch.where(torch.arange(66,device='cuda')%2==0,64.,-64.).half();side.fill_(.125)
            hp_run(k,a,idx,side,base,out);torch.cuda.synchronize()
            row={'O':o,'N':133,'M':5,'case':case,'unselected_column0_NaN':True,'error':checked(out,reference(a,idx,side,base,out.dtype))}
            timing,graph=benchmark(lambda:hp_run(k,a,idx,side,base,out),repetitions=5,calls_per_replay=16);row['timing']=timing
            out.fill_(float('nan'));graph.replay();torch.cuda.synchronize();checked(out,reference(a,idx,side,base,out.dtype))
            report['edge_cases'].append(row)
        rejected=[]
        for bad in (torch.tensor([1,1],dtype=torch.int32),torch.tensor([-1],dtype=torch.int32),torch.tensor([67],dtype=torch.int32),torch.tensor([1],dtype=torch.int64)):
            try:validate_indices(bad,67)
            except ValueError:rejected.append(True)
            else:raise AssertionError('bad index accepted')
        try:launch(k,a,idx,side,base,base,stream=stream(),base_is_masked=True)
        except ValueError:rejected.append(True)
        else:raise AssertionError('alias accepted')
        try:launch(k,a,idx,side,base,out,stream=stream(),base_is_masked=False)
        except ValueError:rejected.append(True)
        else:raise AssertionError('unmasked base accepted')
        report.setdefault('rejection_tests',[]).append({'O':o,'invalid_indices_alias_unmasked_rejected':all(rejected)})
        write_json(outdir/'results.json',report)


if __name__=='__main__':main()
