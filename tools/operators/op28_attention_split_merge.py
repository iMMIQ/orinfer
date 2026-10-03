"""Offline same-math, graph, actual ABI and op22 composition checks for op28."""
import argparse
import math
import re
import shutil
import time
from pathlib import Path
import torch
from common import (ROOT,configure,environment,error,export_kernel,benchmark,
                    identity,write_json)
from abi import parse_host
from kernels.operators.op28_attention_split_merge import (attention_split_merge,
                                                         validate_host_statistics)


def reference(m,l,o,gate,mode='native_fp16',dtype=torch.float16,precision=torch.float32):
    m,l,o=m.to(precision),l.to(precision),o.to(precision)
    valid=l>0
    maxima=torch.where(valid,m,torch.full_like(m,-math.inf)).amax(-1)
    any_valid=valid.any(-1)
    safe=torch.where(any_valid,maxima,torch.zeros_like(maxima))
    # torch.where evaluates both arguments, so make the exponent safe first.
    exponent=torch.where(valid,m-safe[...,None],torch.zeros_like(m))
    weights=torch.exp(exponent)*valid.to(precision)
    denominator=(weights*l).sum(-1)
    safe_denominator=torch.where(any_valid,denominator,torch.ones_like(denominator))
    attention=(weights[...,None]*o).sum(2)/safe_denominator[...,None]
    sigmoid=torch.sigmoid(gate.to(precision))
    if mode=='native_fp16':
        return (attention.half()*sigmoid.half()).half()
    return (attention*sigmoid).to(dtype)


def stats(batch,splits,kind='normal'):
    m=torch.randn((batch,24,splits),device='cuda')*4
    l=torch.rand_like(m)*20+1
    # Local weighted V, not the normalized local attention.
    o=torch.randn((batch,24,splits,256),device='cuda')*l[...,None]
    gate=torch.randn((batch,24,256),device='cuda',dtype=torch.float16)*3
    if kind=='highmax':m+=10000
    if kind=='strong':
        m[:]=torch.arange(splits,device='cuda')*200-1000
        gate[:,:,::2]=32;gate[:,:,1::2]=-32
    if kind in ('empty','all_empty'):
        mask=torch.arange(splits,device='cuda')%2==0
        if kind=='all_empty':mask[:]=True
        m[:,:,mask]=-math.inf;l[:,:,mask]=0;o[:,:,mask]=0
    if kind=='head_isolation':
        for h in range(24):m[:,h,:]+=h*100;o[:,h,:,:]*=(h+1)
    validate_host_statistics(m.cpu().tolist(),l.cpu().tolist(),o.cpu().tolist())
    return m,l,o,gate


def validate_policy():
    m=[[[-math.inf] for _ in range(24)]]
    l=[[[0.] for _ in range(24)]]
    o=[[[[0.]*256] for _ in range(24)]]
    assert validate_host_statistics(m,l,o)
    rejected=[]
    for field,value in [('o',1.),('o',math.nan),('o',math.inf),('l',-1.),
                        ('l',math.nan),('l',math.inf),('m',0.),('m',math.nan)]:
        old={'m':m[0][0][0],'l':l[0][0][0],'o':o[0][0][0][0]}[field]
        if field=='o':o[0][0][0][0]=value
        elif field=='l':l[0][0][0]=value
        else:m[0][0][0]=value
        try:validate_host_statistics(m,l,o)
        except ValueError as exc:rejected.append({'field':field,'value':str(value),'reason':str(exc)})
        else:raise AssertionError('illegal statistics accepted')
        if field=='o':o[0][0][0][0]=old
        elif field=='l':l[0][0][0]=old
        else:m[0][0][0]=old
    return rejected


def export(kernel,out,name,s,mode,dtype):
    dest=out/'aot'/name
    exported=export_kernel(kernel,dest)
    source=(dest/'kernel.cu').read_text()
    abi={'operator':'op28_attention_split_merge','variant':name,'sm':87,
         'actual_launches':parse_host((dest/'host.txt').read_text()),
         'cuda_declarations':[decl for decl in re.findall(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)',source)
                              if decl[0] in exported['symbols']],
         'tensor_api_order':['M','L','O','RawGate','Y'],
         'layouts':{'M/L':f'[B,24,{s}]FP32','O':f'[B,24,{s},256]FP32 unnormalized',
                    'RawGate':'[B,24,256]FP16','Y':f'[B,24,256]{dtype}'},
         'B':'dynamic int32','S':s,'contiguous':True,'gate_mode':mode,
         'workspace_bytes':0,'cooperative_launch':False,
         'stream':'explicit; current capture stream resolved for every call',
         'state_policy':'trusted producer or validate_host_statistics after each mutation; exact empty (-inf,0,zero O)',
         'toolchain':environment(),'files':exported['files'],
         'implementation_source':identity(ROOT/'kernels/operators/op28_attention_split_merge.py')}
    write_json(dest/'abi.json',abi)
    return abi


def check(actual,expected,mode='native_fp16'):
    e=error(actual,expected)
    assert e['finite'] and e['relative_l2']<(.002 if actual.dtype==torch.float16 else 2e-5),e
    return e


def run_case(kernel,b,s,kind,repetitions,mode='native_fp16',dtype=torch.float16,graph_test=False):
    started=time.perf_counter();m,l,o,g=stats(b,s,kind);y=torch.empty(g.shape,device='cuda',dtype=dtype)
    prepare=time.perf_counter()-started
    def run():kernel(m,l,o,g,y,stream=torch.cuda.current_stream().cuda_stream)
    started=time.perf_counter();run();torch.cuda.synchronize();first=time.perf_counter()-started
    expected=reference(m,l,o,g,mode,dtype)
    observed=check(y,expected,mode)
    fp64=check(y,reference(m,l,o,g,mode,dtype,torch.float64),mode) if b==1 else None
    if kind=='all_empty':assert torch.count_nonzero(y)==0
    timing,graph=benchmark(run,repetitions=repetitions,calls_per_replay=16)
    saves=[x.clone() for x in (m,l,o,g)]
    permutation=torch.randperm(s,device='cuda')
    m.copy_(saves[0][:,:,permutation]);l.copy_(saves[1][:,:,permutation]);o.copy_(saves[2][:,:,permutation,:])
    y.fill_(math.nan);run();torch.cuda.synchronize()
    perm_error=check(y,expected,mode)
    for target,saved in zip((m,l,o,g),saves):target.copy_(saved)
    replay=[]
    if graph_test:
        for index,label in enumerate(('m','l','o','raw_gate')):
            target=(m,l,o,g)[index]
            if index==0:target[:,:,0].add_(3.25)
            elif index==1:target.mul_(1.5)
            elif index==2:target.mul_(-.75)
            else:target.add_(2)
            y.fill_(math.nan);graph.replay();torch.cuda.synchronize()
            changed=check(y,reference(m,l,o,g,mode,dtype),mode)
            assert not torch.equal(y,expected),f'{label} mutation was ineffective'
            target.copy_(saves[index]);y.fill_(math.nan);graph.replay();torch.cuda.synchronize()
            restored=check(y,expected,mode)
            replay.append({'changed':label,'poisoned_Y':True,'changed_error':changed,'restored_error':restored})
        old=y.clone();o[:,7,:,:].mul_(2);y.fill_(math.nan);graph.replay();torch.cuda.synchronize()
        mask=torch.arange(24,device='cuda')!=7
        assert torch.equal(y[:,mask],old[:,mask])
        check(y,reference(m,l,o,g,mode,dtype),mode);o.copy_(saves[2])
    return {'B':b,'S':s,'kind':kind,'mode':mode,'dtype':str(dtype),
            'prepare_s':prepare,'first_launch_s':first,'error':observed,'fp64_error':fp64,
            'merge_permutation_error':perm_error,'graph_input_mutations':replay,
            'head_isolation_exact':bool(graph_test),'hot':timing,
            'statistics_input_bytes':b*24*s*258*4,'gate_bytes':g.numel()*2,
            'output_bytes':y.numel()*y.element_size(),'workspace_bytes':0,
            'target_ms':.008 if b==1 else None,'target_met':timing['median_ms']<=.008 if b==1 else None}


def compose_op22(out,kernel,repetitions):
    from kernels.operators.op22_attention_decode import paged_attention_partials,validate_host_metadata
    from tools.operators.op22_attention_decode import reference as attention_reference
    bs,mp,np,s=128,68,136,4
    started=time.perf_counter();partial=paged_attention_partials(mp,np,s)
    compile_s=time.perf_counter()-started
    export_kernel(partial,out/'op22_dependency_aot')
    k=torch.randn((np,bs,4,256),device='cuda',dtype=torch.float16)*.5
    v=torch.randn_like(k)
    cases=[]
    for b,context in ((1,512),(1,2048),(1,8192),(2,8448)):
        q=torch.randn((b,24,256),device='cuda',dtype=torch.float16)
        gate=torch.randn_like(q)*3;y=torch.empty_like(q)
        pages=torch.randperm(np,device='cuda').reshape(2,mp)[:b].int().contiguous()
        if b>1:pages[:,0]=pages[0,0]
        lengths=torch.full((b,),context,device='cuda',dtype=torch.int32)
        if b>1:lengths[1]-=1
        positions=lengths-1
        validate_host_metadata(pages.cpu().tolist(),lengths.cpu().tolist(),positions.cpu().tolist(),np,bs)
        m=torch.empty((b,24,s),device='cuda');l=torch.empty_like(m)
        o=torch.empty((b,24,s,256),device='cuda')
        def run():
            stream=torch.cuda.current_stream().cuda_stream
            partial(q,k,v,pages,lengths,positions,m,l,o,stream=stream)
            kernel(m,l,o,gate,y,stream=stream)
        run();torch.cuda.synchronize()
        validate_host_statistics(m.cpu().tolist(),l.cpu().tolist(),o.cpu().tolist())
        # Independent full-attention FP32 stable reference, not reference merge on TL stats.
        ref=attention_reference(q,k,v,pages,lengths,positions,gate,1)[0]
        observed=check(y,ref)
        merged=check(y,reference(m,l,o,gate))
        hot,_=benchmark(run,repetitions=repetitions,calls_per_replay=1)
        cases.append({'B':b,'context':context,'lengths':lengths.cpu().tolist(),'S':s,
                      'full_attention_error':observed,'TL_partials_merge_error':merged,
                      'partial_and_merge_hot':hot,'stats_bytes':b*24*s*258*4})
        print(f'op22+op28 B{b} context{context} L2={observed["relative_l2"]:.3g} ms={hot["median_ms"]:.6f}',flush=True)
    return {'compile_s':compile_s,'cases':cases,
            'dependency_source':identity(ROOT/'kernels/operators/op22_attention_decode.py'),
            'partial_abi':parse_host((out/'op22_dependency_aot/host.txt').read_text())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--repetitions',type=int,default=15);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--skip-composition',action='store_true');args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);configure()
    native=Path('/home/nvidia/model/orin-kv8-mtp-20261001/vllm020/model_executor/models/qwen3_next.py')
    shutil.copyfile(native,out/'qwen3_next.py');assert 'gate = torch.sigmoid(gate)' in native.read_text()
    results={'environment':environment(),'native_reference':identity(out/'qwen3_next.py'),
             'gate_reference_lines':[304,307,308],'cpu_policy_rejections':validate_policy(),
             'exports':[],'cases':[],'status':'in_progress','workspace_bytes':0}
    kernels={}
    for s in ((4,) if args.smoke else (1,2,3,4,5,7,8,16)):
        started=time.perf_counter();kernels[s]=attention_split_merge(s)
        results['exports'].append({'name':f'native_s{s}','compile_prepare_s':time.perf_counter()-started,
                                  'abi':export(kernels[s],out,f'native_s{s}',s,'native_fp16','FP16')})
        write_json(out/'results.json',results)
    cases=[(1,4,'normal',True)] if args.smoke else [
        (b,s,'normal',b==3 and s==4) for s in kernels for b in (1,2,3,4,5,7,8)]
    if not args.smoke:cases += [(1,s,kind,False) for s in kernels for kind in
                               ('highmax','strong','empty','all_empty','head_isolation')]
    for b,s,kind,graph_test in cases:
        case=run_case(kernels[s],b,s,kind,args.repetitions,graph_test=graph_test)
        results['cases'].append(case);write_json(out/'results.json',results)
        print(f'native B{b} S{s} {kind} {case["hot"]["median_ms"]:.6f}ms L2={case["error"]["relative_l2"]:.3g}',flush=True)
    if not args.smoke:
        for dtype in ('float16','float32'):
            started=time.perf_counter();fused=attention_split_merge(4,'fp32_fused',dtype)
            results['exports'].append({'name':f'fused_s4_{dtype}','compile_prepare_s':time.perf_counter()-started,
                'abi':export(fused,out,f'fused_s4_{dtype}',4,'fp32_fused',dtype)})
            for b in (1,3,8):results['cases'].append(run_case(fused,b,4,'normal',args.repetitions,
                'fp32_fused',getattr(torch,dtype),graph_test=b==3))
            write_json(out/'results.json',results)
    if not args.skip_composition:
        results['op22_composition']=compose_op22(out,kernels[4],args.repetitions)
    results['memory']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                       'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    results['implementation_identity']=[identity(ROOT/'kernels/operators/op28_attention_split_merge.py'),identity(Path(__file__))]
    results['status']=('passed standalone tests; composition explicitly skipped' if args.skip_composition else
                       'passed standalone and composition tests; runtime/model integration pending')
    write_json(out/'results.json',results)
    print('op28 complete; wrapper releases GPU lock',flush=True)


if __name__=='__main__':main()
