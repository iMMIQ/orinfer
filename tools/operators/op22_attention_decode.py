"""Isolated op22 correctness, paged boundaries, graph and timing runner."""
import argparse
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
import torch
from common import (ROOT,configure,environment,error,export_kernel,benchmark,
                    identity,write_json)
from abi import parse_host,evaluate
from kernels.operators.op22_attention_decode import (paged_attention_decode,
    paged_attention_partials,validate_host_metadata,paged_attention_decode_gqa,
    paged_attention_partials_gqa)

MP,NP,BS=68,544,128


def reference(q,k,v,pages,lengths,positions,gate,nsplits=1):
    b=q.shape[0]
    m=torch.full((b,24,nsplits),-float('inf'),device=q.device)
    l=torch.zeros_like(m)
    o=torch.zeros((b,24,nsplits,256),device=q.device)
    for r in range(b):
        valid=max(0,min(int(lengths[r]),int(positions[r])+1))
        width=(valid+nsplits-1)//nsplits
        for s in range(nsplits):
            start=s*width;end=min(start+width,valid)
            if start>=end:continue
            tokens=torch.arange(start,end,device=q.device)
            physical=pages[r,tokens//BS].long()
            kr=k[physical,tokens%BS].float().permute(1,0,2)
            vr=v[physical,tokens%BS].float().permute(1,0,2)
            scores=torch.matmul(q[r].float().reshape(4,6,256),kr.transpose(1,2))*.0625
            maximum=scores.amax(-1)
            p=torch.exp(scores-maximum[:,:,None])
            m[r,:,s]=maximum.reshape(24)
            l[r,:,s]=p.sum(-1).reshape(24)
            o[r,:,s]=torch.matmul(p,vr).reshape(24,256)
    return merge(m,l,o,gate),m,l,o


def merge(m,l,o,gate):
    maximum=m.amax(-1)
    any_valid=l.sum(-1)>0
    # Avoid inf-inf for empty groups, and suppress empty partitions.
    safe=torch.where(any_valid,maximum,torch.zeros_like(maximum))
    weights=torch.where(l>0,torch.exp(m-safe[:,:,None]),torch.zeros_like(m))
    denominator=(weights*l).sum(-1)
    attention=(o*weights[:,:,:,None]).sum(2)/denominator.clamp_min(1e-30)[:,:,None]
    return (attention.half()*torch.sigmoid(gate).half()).half()


def metadata(batch,context,ragged=False):
    pages=torch.randperm(NP,device='cuda').reshape(8,MP)[:batch].int().contiguous()
    if batch>1:
        # Immutable full prefix page is shared; different tails remain private.
        pages[:,0]=pages[0,0]
    lengths=torch.full((batch,),context,device='cuda',dtype=torch.int32)
    if ragged:
        for r in range(batch):lengths[r]=max(0,context-r*17)
    positions=lengths-1
    for r in range(batch):
        pages[r,(int(lengths[r])+BS-1)//BS:]=-999
    validate_host_metadata(pages.cpu().tolist(),lengths.cpu().tolist(),positions.cpu().tolist(),NP,BS)
    return pages,lengths,positions


def export(kernel,output,name,split,implementation='simt_fp32'):
    dest=output/'aot'/name
    artifacts=export_kernel(kernel,dest)
    source=(dest/'kernel.cu').read_text()
    data={'operator':'op22_attention_decode','variant':name,'sm':87,'implementation':implementation,
        'actual_launches':parse_host((dest/'host.txt').read_text()),
        'actual_cuda_declarations':[x for x in re.findall(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)',source) if x[0] in artifacts['symbols']],
        'tensor_api_order':['Q','K','V','Pages','SeqLen','QueryPos']+(['M','L','O'] if split else ['RawGate','Y']),
        'shape_parameters':{'B':'dynamic int32','max_pages':MP,'num_pages':NP,'block_size':BS,'splits':split or 1,
                            'tile_tokens':64 if implementation=='gqa_tensorcore' else 32},
        'layouts':{'Q/RawGate/Y':'[B,24,256]FP16 contiguous','K/V':'[544,128,4,256]FP16 contiguous',
            'Pages':'[B,68]int32','SeqLen/QueryPos':'[B]int32','M/L':'[B,24,S]FP32','O':'[B,24,S,256]FP32 unnormalized'},
        'workspace_bytes_per_B':24*split*258*4 if split else 0,
        'metadata_policy':'CPU validates every mutation; defensive device guards prevent illegal page reads; empty valid set outputs0',
        'rounding':('FP32 online sums and fused gate, only final FP16 output' if name=='fp32_fused' else
                    'FP32 online sums; direct half(attention)*half(sigmoid(rawGate)) then half'),
        'partition':'ceil(min(seqLen,queryPos+1)/S); empty=-inf,0,0',
        'PV_probability_rounding':'FP16 operand, FP32 accumulation' if implementation=='gqa_tensorcore' else 'FP32',
        'KV_reuse':'six Q heads per staged KV tile, padded16 rows only6 output owners' if implementation=='gqa_tensorcore' else 'per Q head, no explicit GQA reuse',
        'stream':'each invocation obtains capture-current stream; caller-owned disjoint buffers',
        'cooperative_launch':False,'toolchain':environment(),'files':artifacts['files']}
    write_json(dest/'abi.json',data)
    return data


def check_partials(actual,expected):
    am,al,ao=actual;em,el,eo=expected
    assert torch.equal(torch.isneginf(am),torch.isneginf(em))
    finite=torch.isfinite(em)
    me=error(am[finite],em[finite]) if finite.any() else {'finite':True,'relative_l2':0,'max_abs':0}
    le=error(al,el);oe=error(ao,eo)
    assert me['max_abs']<.02 and le['finite'] and oe['finite']
    # Local max rescaling means unnormalized O comparisons are meaningful.
    assert le['relative_l2']<.002 and oe['relative_l2']<.002
    empty=el==0
    assert torch.equal(al[empty],el[empty]) and (ao[empty]==0).all()
    return {'m':me,'l':le,'o_unnormalized':oe,'empty_partitions_exact':True}


def run_case(batch,context,k,v,direct,splits,repetitions,kind='normal',graph_test=False):
    prep=time.perf_counter()
    pages,lengths,positions=metadata(batch,context,kind=='ragged')
    q=torch.randn((batch,24,256),device='cuda',dtype=torch.float16)
    gate=torch.randn_like(q)*3
    if kind=='causal':positions.copy_(lengths//2-1)
    elif kind=='strong':q.mul_(64);gate[:,:,::2]=32;gate[:,:,1::2]=-32
    elif kind=='empty':lengths.zero_();positions.fill_(-1);pages.fill_(-999)
    elif kind=='no_valid':positions.fill_(-1)
    tail_saves=[]
    if kind=='poison_tail':
        for r in range(batch):
            tail=int(lengths[r])%BS
            if tail:
                page=int(pages[r,int(lengths[r])//BS])
                tail_saves.append((page,tail,k[page,tail:].clone(),v[page,tail:].clone()))
                k[page,tail:]=float('nan');v[page,tail:]=float('nan')
    y=torch.empty_like(q)
    storage={s:(torch.empty((batch,24,s),device='cuda'),torch.empty((batch,24,s),device='cuda'),
                torch.empty((batch,24,s,256),device='cuda')) for s in splits}
    prepare_s=time.perf_counter()-prep
    def run_direct():direct(q,k,v,pages,lengths,positions,gate,y,stream=torch.cuda.current_stream().cuda_stream)
    start=time.perf_counter();run_direct();torch.cuda.synchronize();first_s=time.perf_counter()-start
    expected=reference(q,k,v,pages,lengths,positions,gate)
    observed=error(y,expected[0]);assert observed['finite'] and observed['relative_l2']<.002
    hot,graph=benchmark(run_direct,repetitions=repetitions,calls_per_replay=1)
    details=[]
    for s,kernel in splits.items():
        m,l,o=storage[s]
        def run_partial():kernel(q,k,v,pages,lengths,positions,m,l,o,stream=torch.cuda.current_stream().cuda_stream)
        started=time.perf_counter();run_partial();torch.cuda.synchronize();partial_first=time.perf_counter()-started
        ref=reference(q,k,v,pages,lengths,positions,gate,s)
        errors=check_partials((m,l,o),ref[1:])
        merged=error(merge(m,l,o,gate),expected[0]);assert merged['finite'] and merged['relative_l2']<.002
        timing,partial_graph=benchmark(run_partial,repetitions=repetitions)
        replay=None
        if graph_test:
            saves=[x.clone() for x in (q,k,v,gate,pages,lengths,positions)]
            q.mul_(-.5);k.mul_(.75);v.mul_(-.25);gate.add_(1.25)
            pages.copy_((pages+7)%NP);lengths.sub_(1);positions.copy_(lengths//2-1)
            validate_host_metadata(pages.cpu().tolist(),lengths.cpu().tolist(),positions.cpu().tolist(),NP,BS)
            y.fill_(float('nan'));m.fill_(float('nan'));l.fill_(float('nan'));o.fill_(float('nan'))
            graph.replay();partial_graph.replay();torch.cuda.synchronize()
            new=reference(q,k,v,pages,lengths,positions,gate,s)
            assert error(y,new[0])['relative_l2']<.002
            check_partials((m,l,o),new[1:])
            for target,saved in zip((q,k,v,gate,pages,lengths,positions),saves):target.copy_(saved)
            y.fill_(float('nan'));m.fill_(float('nan'));l.fill_(float('nan'));o.fill_(float('nan'))
            graph.replay();partial_graph.replay();torch.cuda.synchronize()
            assert error(y,expected[0])['relative_l2']<.002
            check_partials((m,l,o),ref[1:])
            replay={'changed':'Q,K,V,rawGate,Pages,seqLengths,queryPositions','poisoned':'Y,M,L,O','restored':True}
        details.append({'splits':s,'first_launch_s':partial_first,'errors':errors,'merged_output_error':merged,
                        'partial_only_hot':timing,'workspace_bytes':batch*24*s*258*4,'graph':replay})
    for page,tail,ksave,vsave in tail_saves:
        k[page,tail:]=ksave;v[page,tail:]=vsave
    return {'B':batch,'context':context,'kind':kind,'prepare_allocation_metadata_s':prepare_s,
        'first_direct_launch_s':first_s,'direct_error':observed,'direct_hot':hot,'partials':details,
        'shared_prefix_page':batch>1,'budget_ms':.35 if batch==1 and context==8448 else None,
        'budget_met':hot['median_ms']<=.35 if batch==1 and context==8448 else None,
        'KV_capacity_bytes':k.numel()*k.element_size()*2,'output_bytes':y.numel()*y.element_size()}


def cpu_rejections():
    cases=[([[0]],[-1],[-1]),([[0]],[129],[0]),([[0]],[1],[-2]),([[0]],[1],[1]),
           ([[-1]],[1],[0]),([[NP]],[1],[0]),([[0]],[True],[0]),([[0]],[],[])]
    result=[]
    for args in cases:
        try:validate_host_metadata(*args,NP,BS)
        except ValueError as exc:result.append(str(exc))
        else:raise AssertionError('illegal metadata accepted')
    assert validate_host_metadata([[-1]],[0],[-1],NP,BS)
    return result


def defensive_device_guards(k,v,direct,splits):
    q=torch.randn((1,24,256),device='cuda',dtype=torch.float16)
    gate=torch.zeros_like(q);y=torch.empty_like(q)
    pages=torch.zeros((1,MP),device='cuda',dtype=torch.int32)
    lengths=torch.ones(1,device='cuda',dtype=torch.int32)
    positions=torch.zeros_like(lengths)
    m=torch.empty((1,24,8),device='cuda');l=torch.empty_like(m)
    o=torch.empty((1,24,8,256),device='cuda')
    observed=[]
    # Intentionally bypass the mandatory CPU scheduler policy only in this test.
    for name,page,length,pos in [('negative_page',-1,1,0),('large_page',NP,1,0),
                                  ('negative_length',0,-1,0),('negative_query',0,1,-2)]:
        pages.fill_(page);lengths.fill_(length);positions.fill_(pos)
        y.fill_(float('nan'));m.fill_(float('nan'));l.fill_(float('nan'));o.fill_(float('nan'))
        stream=torch.cuda.current_stream().cuda_stream
        direct(q,k,v,pages,lengths,positions,gate,y,stream=stream)
        splits[8](q,k,v,pages,lengths,positions,m,l,o,stream=stream)
        torch.cuda.synchronize()
        assert (y==0).all() and torch.isneginf(m).all() and (l==0).all() and (o==0).all()
        observed.append({'kind':name,'guarded_zero_outputs':True,'policy':'CPU must reject; bypass diagnostic only'})
    return observed


def fp32_candidate(k,v,kernel,repetitions):
    pages,lengths,positions=metadata(1,8448)
    q=torch.randn((1,24,256),device='cuda',dtype=torch.float16)
    gate=torch.randn_like(q)*3;y=torch.empty_like(q)
    native,m,l,o=reference(q,k,v,pages,lengths,positions,gate)
    ref=((o[:,:,0,:]/l[:,:,0,None])*torch.sigmoid(gate.float())).half()
    def run():kernel(q,k,v,pages,lengths,positions,gate,y,stream=torch.cuda.current_stream().cuda_stream)
    started=time.perf_counter();run();torch.cuda.synchronize();first=time.perf_counter()-started
    same_math=error(y,ref);assert same_math['finite'] and same_math['relative_l2']<.002
    timing,_=benchmark(run,repetitions=repetitions)
    return {'B':1,'context':8448,'gate_mode':'fp32_fused','same_math_error':same_math,
        'vs_native_fp16_rounding':error(y,native),'first_launch_s':first,'hot':timing,
        'scope':'separately identified rounding candidate, not native default'}


def generate_rust_probe(out,exports):
    """Generate a dependency-free Driver probe from the actual launch ABI.

    Compiled on the host, then invoked by --rust-probe under run.sh's GPU lock.
    Uniform exact-half fixtures isolate ABI/stream/graph correctness.
    """
    direct=next(e['abi'] for e in exports if e['name']=='direct')['actual_launches'][0]
    partial=next(e['abi'] for e in exports if e['name']=='partials4')['actual_launches'][0]
    def arguments(abi):
        result=[]
        for arg in abi['ordered_arguments']:
            value=arg['value']
            if value.endswith('.data_ptr()'):
                result.append(f'&mut {value[:-11]} as *mut u64 as *mut c_void')
            elif value=='batch':result.append('&mut batch as *mut i32 as *mut c_void')
            else:raise ValueError(f'unexpected actual ABI {arg}')
        return ','.join(result)
    def launch(abi):
        dims=abi['launch_expressions']
        return ','.join(str(evaluate(dims[x],{'batch':3})) for x in
                       ('gridDimX','gridDimY','gridDimZ','blockDimX','blockDimY','blockDimZ','sharedMemBytes'))
    source=r'''// Generated from op22 actual host ABI. Standalone validation, not runtime.
use std::{ffi::{c_char,c_void,CString},ptr};
type H=*mut c_void;
#[link(name="cuda")]
unsafe extern "C" {
fn cuInit(f:u32)->i32;fn cuDeviceGet(d:*mut i32,n:i32)->i32;
fn cuDevicePrimaryCtxRetain(c:*mut H,d:i32)->i32;fn cuDevicePrimaryCtxRelease_v2(d:i32)->i32;
fn cuCtxSetCurrent(c:H)->i32;fn cuModuleLoad(m:*mut H,p:*const c_char)->i32;
fn cuModuleGetFunction(f:*mut H,m:H,n:*const c_char)->i32;fn cuModuleUnload(m:H)->i32;
fn cuFuncSetAttribute(f:H,a:i32,v:i32)->i32;
fn cuMemAlloc_v2(p:*mut u64,n:usize)->i32;fn cuMemFree_v2(p:u64)->i32;
fn cuMemsetD16_v2(p:u64,v:u16,n:usize)->i32;fn cuMemsetD32_v2(p:u64,v:u32,n:usize)->i32;
fn cuMemcpyHtoD_v2(p:u64,s:*const c_void,n:usize)->i32;
fn cuMemcpyDtoH_v2(d:*mut c_void,p:u64,n:usize)->i32;
fn cuStreamCreate(s:*mut H,f:u32)->i32;fn cuStreamDestroy_v2(s:H)->i32;fn cuStreamSynchronize(s:H)->i32;
fn cuLaunchKernel(f:H,gx:u32,gy:u32,gz:u32,bx:u32,by:u32,bz:u32,sm:u32,s:H,a:*mut *mut c_void,e:*mut *mut c_void)->i32;
fn cuStreamBeginCapture_v2(s:H,m:i32)->i32;fn cuStreamEndCapture(s:H,g:*mut H)->i32;
fn cuGraphInstantiateWithFlags(e:*mut H,g:H,f:u64)->i32;fn cuGraphLaunch(e:H,s:H)->i32;
fn cuGraphExecDestroy(e:H)->i32;fn cuGraphDestroy(g:H)->i32;
}
fn check(c:i32,where_:&str){assert_eq!(c,0,"CUDA {where_}: {c}");}
unsafe fn alloc(n:usize)->u64{let mut p=0;check(unsafe{cuMemAlloc_v2(&mut p,n)},"alloc");p}
unsafe fn upload<T>(p:u64,v:&[T]){check(unsafe{cuMemcpyHtoD_v2(p,v.as_ptr().cast(),std::mem::size_of_val(v))},"upload");}
unsafe fn read<T:Default+Clone>(p:u64,n:usize)->Vec<T>{let mut v=vec![T::default();n];check(unsafe{cuMemcpyDtoH_v2(v.as_mut_ptr().cast(),p,n*std::mem::size_of::<T>())},"read");v}
fn main(){unsafe{
let argv:Vec<String>=std::env::args().collect();assert_eq!(argv.len(),3);
check(cuInit(0),"init");let mut d=0;check(cuDeviceGet(&mut d,0),"device");
let mut ctx=ptr::null_mut();check(cuDevicePrimaryCtxRetain(&mut ctx,d),"retain");check(cuCtxSetCurrent(ctx),"context");
let mut modules=[ptr::null_mut();2];let mut functions=[ptr::null_mut();2];
for i in 0..2{let path=CString::new(argv[i+1].clone()).unwrap();check(cuModuleLoad(&mut modules[i],path.as_ptr()),"module");
let symbol=CString::new("@SYMBOL@").unwrap();check(cuModuleGetFunction(&mut functions[i],modules[i],symbol.as_ptr()),"symbol");}
check(cuFuncSetAttribute(functions[0],8,@SHARED_DIRECT@),"shared direct");
check(cuFuncSetAttribute(functions[1],8,@SHARED_PARTIAL@),"shared partial");
let mut stream=ptr::null_mut();check(cuStreamCreate(&mut stream,1),"nondefault stream");
let mut batch=3i32;let n=3*24*256;let kv=544*128*4*256;let stats=3*24*4;
let mut Q=alloc(n*2);let mut K=alloc(kv*2);let mut V=alloc(kv*2);let mut RawGate=alloc(n*2);let mut Y=alloc(n*2);
let mut Pages=alloc(3*68*4);let mut SeqLen=alloc(3*4);let mut QueryPos=alloc(3*4);
let mut M=alloc(stats*4);let mut L=alloc(stats*4);let mut O=alloc(stats*256*4);
let mut da=[@DIRECT_ARGS@];let mut pa=[@PARTIAL_ARGS@];
let launch=|da:&mut [*mut c_void],pa:&mut [*mut c_void]|{
check(cuLaunchKernel(functions[0],@DIRECT_LAUNCH@,stream,da.as_mut_ptr(),ptr::null_mut()),"launch direct");
check(cuLaunchKernel(functions[1],@PARTIAL_LAUNCH@,stream,pa.as_mut_ptr(),ptr::null_mut()),"launch partial");};
let mut graph:H=ptr::null_mut();let mut exec:H=ptr::null_mut();
for changed in [false,true,false]{
check(cuMemsetD16_v2(Q,if changed{0x3400}else{0},n),"Q");
check(cuMemsetD16_v2(K,if changed{0x3c00}else{0},kv),"K");
check(cuMemsetD16_v2(V,if changed{0x4000}else{0x3800},kv),"V");
check(cuMemsetD16_v2(RawGate,if changed{0x5000}else{0},n),"gate");
upload(Pages,&vec![if changed{1i32}else{0};3*68]);upload(SeqLen,&vec![if changed{2i32}else{3};3]);
upload(QueryPos,&vec![if changed{1i32}else{2};3]);
check(cuMemsetD16_v2(Y,0x7e00,n),"poison Y");
for (p,count) in [(M,stats),(L,stats),(O,stats*256)]{check(cuMemsetD32_v2(p,f32::NAN.to_bits(),count),"poison stats");}
if exec.is_null(){launch(&mut da,&mut pa);check(cuStreamSynchronize(stream),"initial sync");
check(cuStreamBeginCapture_v2(stream,0),"begin capture");launch(&mut da,&mut pa);
check(cuStreamEndCapture(stream,&mut graph),"end capture");check(cuGraphInstantiateWithFlags(&mut exec,graph,0),"instantiate");}
check(cuGraphLaunch(exec,stream),"graph replay");check(cuStreamSynchronize(stream),"replay sync");
assert!(read::<u16>(Y,n).iter().all(|x|*x==if changed{0x4000}else{0x3400}));
let ms=read::<f32>(M,stats);let ls=read::<f32>(L,stats);let os=read::<f32>(O,stats*256);
for i in 0..stats{let active=i%4<if changed{2}else{3};
assert_eq!(ms[i],if active{if changed{4.0}else{0.0}}else{f32::NEG_INFINITY});
assert_eq!(ls[i],if active{1.0}else{0.0});
for j in 0..256{assert_eq!(os[i*256+j],if active{if changed{2.0}else{0.5}}else{0.0});}}}
check(cuGraphExecDestroy(exec),"exec destroy");check(cuGraphDestroy(graph),"graph destroy");
for p in [Q,K,V,RawGate,Y,Pages,SeqLen,QueryPos,M,L,O]{check(cuMemFree_v2(p),"free");}
check(cuStreamDestroy_v2(stream),"stream destroy");for m in modules{check(cuModuleUnload(m),"unload");}
check(cuDevicePrimaryCtxRelease_v2(d),"release");
println!("{{\"rust_actual_driver_abi\":true,\"B\":3,\"direct_and_partials4\":true,\"explicit_nondefault_stream\":true,\"exact_half_fixture\":true}}");
}}
'''
    replacements={'@SYMBOL@':direct['symbol'],'@SHARED_DIRECT@':direct['launch_expressions']['sharedMemBytes'],
        '@SHARED_PARTIAL@':partial['launch_expressions']['sharedMemBytes'],'@DIRECT_ARGS@':arguments(direct),
        '@PARTIAL_ARGS@':arguments(partial),'@DIRECT_LAUNCH@':launch(direct),'@PARTIAL_LAUNCH@':launch(partial)}
    for key,value in replacements.items():source=source.replace(key,value)
    (out/'op22_driver_probe.rs').write_text(source)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--repetitions',type=int,default=5);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--implementation',choices=['simt_fp32','gqa_tensorcore'],default='simt_fp32')
    parser.add_argument('--rust-probe',help='host-compiled executable; requires --rust-aot-root')
    parser.add_argument('--rust-aot-root')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.rust_probe:
        root=Path(args.rust_aot_root)
        completed=subprocess.run([args.rust_probe,str(root/'aot/direct/kernel.cubin'),str(root/'aot/partials4/kernel.cubin')],capture_output=True,text=True,check=True)
        write_json(out/'rust-result.json',{'stdout':completed.stdout,'stderr':completed.stderr,
            'probe':identity(args.rust_probe),'source':identity(root/'op22_driver_probe.rs')})
        print(completed.stdout,flush=True);return
    configure()
    native=Path('/home/nvidia/model/orin-kv8-mtp-20261001/vllm020/model_executor/models/qwen3_next.py')
    config_path=Path('/home/nvidia/model/vllm-comparison-20260930/awq-http/config.json')
    config=json.loads(config_path.read_text())['text_config']
    assert (config['num_attention_heads'],config['num_key_value_heads'],config['head_dim'])==(24,4,256)
    assert config.get('attn_output_gate',True) is True
    shutil.copyfile(native,out/'qwen3_next.py')
    assert 'gate = torch.sigmoid(gate)' in native.read_text()
    results={'environment':environment(),'implementation':args.implementation,'native_reference_source':identity(out/'qwen3_next.py'),
        'model_config':identity(config_path),'model_shape':{'Q_heads':24,'KV_heads':4,'D':256,'GQA':6,'gate':'sigmoid'},
        'native_gate_lines':[307,308,309],'cpu_rejections':cpu_rejections(),'exports':[],'cases':[],
        'status':'in_progress','scope':'synthetic same-math attention, no KV quantization or model quality claim'}
    direct_builder=paged_attention_decode if args.implementation=='simt_fp32' else paged_attention_decode_gqa
    partial_builder=paged_attention_partials if args.implementation=='simt_fp32' else paged_attention_partials_gqa
    start=time.perf_counter();direct=direct_builder(MP,NP)
    results['exports'].append({'name':'direct','compile_prepare_s':time.perf_counter()-start,
        'abi':export(direct,out,'direct',0,args.implementation)})
    splits={}
    for s in (1,2,4,8):
        start=time.perf_counter();splits[s]=partial_builder(MP,NP,s)
        results['exports'].append({'name':f'partials{s}','compile_prepare_s':time.perf_counter()-start,
            'abi':export(splits[s],out,f'partials{s}',s,args.implementation)})
        write_json(out/'results.json',results)
    k=torch.randn((NP,BS,4,256),device='cuda',dtype=torch.float16)*.5
    v=torch.randn_like(k)
    cases=[(1,129,'normal',False)] if args.smoke else [
        (b,c,'normal',False) for b in (1,2,3,4,5,7,8) for c in (512,2048,8192,8448)]
    if not args.smoke:cases += [(3,c,'ragged',c==513) for c in (127,128,129,511,513)] + [
        (2,513,'causal',False),(1,513,'strong',False),(3,0,'empty',False),(3,129,'no_valid',False),
        (2,3,'normal',False),(3,129,'poison_tail',False)]
    for b,c,kind,graph_test in cases:
        case=run_case(b,c,k,v,direct,splits,args.repetitions,kind,graph_test)
        results['cases'].append(case);write_json(out/'results.json',results)
        print(f'B{b} ctx{c} {kind} direct={case["direct_hot"]["median_ms"]:.6f}ms L2={case["direct_error"]["relative_l2"]:.3g} splits='+','.join(f'{p["splits"]}:{p["partial_only_hot"]["median_ms"]:.5f}' for p in case['partials']),flush=True)
    if not args.smoke and args.implementation=='simt_fp32':
        started=time.perf_counter();candidate=paged_attention_decode(MP,NP,gate_mode='fp32_fused')
        results['exports'].append({'name':'fp32_fused','compile_prepare_s':time.perf_counter()-started,
            'abi':export(candidate,out,'fp32_fused',0)})
        results['fp32_fusion_candidate']=fp32_candidate(k,v,candidate,args.repetitions)
    results['memory']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    results['defensive_device_guards']=defensive_device_guards(k,v,direct,splits)
    results['implementation_identity']=[identity(ROOT/'kernels/operators/op22_attention_decode.py'),identity(Path(__file__))]
    generate_rust_probe(out,results['exports'])
    results['status']='passed standalone tests; Rust/model/prefix allocator integration pending'
    write_json(out/'results.json',results)
    print('op22 complete; wrapper releases GPU lock',flush=True)


if __name__=='__main__':main()
