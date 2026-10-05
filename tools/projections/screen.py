#!/usr/bin/env python3
"""Run only under the wrapper's flock. Real checkpoint, captured activations.

CUDA paths: TileLang W4A16, per-call W4->W8 + A8 + TileLang GEMM,
TileLang complete FP16 head (source BF16-to-FP16 cast loss reported separately), cuBLAS reference.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from tools.reference import CHECKPOINT, checkpoint_sha256
import statistics
import subprocess
import sys
import time
import traceback
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'kernels/projections'))
import candidates as kernels
import tilelang
import tilelang.language as T

MODEL = str(CHECKPOINT / 'model.safetensors')
CASES = {
 'gate_up': ['mlp.gate_proj', 'mlp.up_proj'],
 'down': ['mlp.down_proj'],
 'gdn_qkvz': ['linear_attn.in_proj_qkv', 'linear_attn.in_proj_z'],
 'gdn_out': ['linear_attn.out_proj'],
}
MATCH = {'gate_up':'gate_up_proj', 'down':'down_proj', 'gdn_qkvz':'in_proj_qkvz', 'gdn_out':'out_proj', 'head':'lm_head'}


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def error(ref, out):
    d = out.float() - ref.float()
    return dict(relative_l2=(d.norm()/ref.float().norm().clamp_min(1e-20)).item(),
                max_abs=d.abs().max().item(), mean_abs=d.abs().mean().item(),
                finite=bool(torch.isfinite(out).all()))


def graph_ms(run, output, a=None):
    for _ in range(3): run()
    torch.cuda.synchronize()
    expected = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5): run()
    output.fill_(float('nan') if output.is_floating_point() else 0); graph.replay(); torch.cuda.synchronize()
    assert torch.equal(output, expected), 'Graph replay output changed'
    if a is not None:
        old = a.clone(); a.zero_(); output.fill_(float('nan'))
        graph.replay(); torch.cuda.synchronize()
        assert bool((output == 0).all()), 'Graph did not observe changed input'
        a.copy_(old); graph.replay(); torch.cuda.synchronize()
        assert torch.equal(output, expected), 'Graph restore failed'
    ts=[]
    for _ in range(3):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(4): graph.replay()
        end.record(); end.synchronize(); ts.append(begin.elapsed_time(end)/20)
    return statistics.median(ts)


def load_weights(case, layer):
    prefix=f'model.language_model.layers.{layer}.'
    fields={name:[] for name in ('weight_packed','weight_scale','weight_zero_point')}
    identity=[]
    started=time.monotonic()
    costs=dict(source_read_hash_cpu_s=0.,unpack_repack_cpu_s=0.)
    with safe_open(MODEL, framework='pt', device='cpu') as f:
        for part in CASES[case]:
            read_started=time.monotonic()
            part_t={field:f.get_tensor(prefix+part+'.'+field) for field in fields}
            n,k8=part_t['weight_packed'].shape
            for field,t in part_t.items():
                identity.append(dict(name=prefix+part+'.'+field, shape=list(t.shape), dtype=str(t.dtype), sha256=sha(t)))
            costs['source_read_hash_cpu_s']+=time.monotonic()-read_started
            pack_started=time.monotonic()
            raw=part_t['weight_packed']; zeros=part_t['weight_zero_point']; scale=part_t['weight_scale'].half()
            shifts=torch.arange(8,dtype=torch.int32)*4
            codes=((raw[:,:,None]>>shifts)&15).to(torch.uint8).reshape(n,k8*8)
            z=((zeros[:,None,:]>>shifts[None,:,None])&15).to(torch.int8).reshape(n,k8*8//128)
            # Lossless adjacent nibble representation. BF16 scales -> FP16 is exact
            # only if verified (all selected values are checked below).
            assert torch.equal(scale.bfloat16(),part_t['weight_scale']), 'Scale conversion is lossy'
            fields['weight_packed'].append((codes[:,::2]|(codes[:,1::2]<<4)).contiguous())
            fields['weight_zero_point'].append(z); fields['weight_scale'].append(scale)
            costs['unpack_repack_cpu_s']+=time.monotonic()-pack_started
    copy_started=time.monotonic()
    p,s,z=[torch.cat(fields[field]).cuda() for field in fields]
    torch.cuda.synchronize(); costs['concat_H2D_s']=time.monotonic()-copy_started
    reference_started=time.monotonic()
    n,k2=p.shape; k=k2*2
    b=torch.empty((n,k),device='cuda',dtype=torch.float16)
    for start in range(0,n,1024):
        code=torch.stack((p[start:start+1024]&15,p[start:start+1024]>>4),dim=-1).reshape(-1,k)
        b[start:start+1024]=((code.reshape(-1,k//128,128).float()-z[start:start+1024,:,None].float())*s[start:start+1024,:,None].float()).reshape(-1,k).half()
    torch.cuda.synchronize()
    costs['benchmark_only_reference_dequant_s']=time.monotonic()-reference_started
    costs['total_s']=time.monotonic()-started
    return p,s,z,b,identity,costs


def activations(folder, case, layer):
    rows=[]
    for path in sorted(folder.glob('*.json')):
        meta=json.loads(path.read_text())
        kind=meta['kind']
        if (case=='head' and kind=='lm_head') or (case!='head' and f'layers.{layer}.' in kind and kind.endswith(MATCH[case])):
            t=torch.load(folder/meta['file'],map_location='cpu',weights_only=True).half()
            if meta['mode']=='prefill':
                rows.append(('prefill',t,meta))
            else: rows.append(('decode',t,meta))
    pref=next(((t,meta) for mode,t,meta in rows if mode=='prefill' and (len(t)==512 or case=='head')),None)
    dec=[(t,meta) for mode,t,meta in rows if mode=='decode']
    dec.sort(key=lambda row:row[1]['computed_tokens_before'])
    if not pref or len(dec)<8: raise RuntimeError(f'Missing real activations for {case}: prefill={bool(pref)} decode={len(dec)}')
    joined=torch.cat([t for t,_ in dec])
    result=[]
    for m in (1,2,3,4,5,7,8):
        result.append((m,joined[:m].contiguous(),dict(origin='concatenated consecutive actual M1 decode steps; not simultaneous batch trace',sources=[meta for _,meta in dec[:m]])))
    if case!='head': result.append((512,pref[0],dict(origin='actual 512-row prefill',sources=[pref[1]])))
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--activations',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--cases',default='gate_up,down,gdn_qkvz,gdn_out,head')
    ap.add_argument('--layer',type=int,default=0); args=ap.parse_args()
    torch.set_num_threads(2); torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.cuda.get_device_properties(0)
    records=[]
    thermal={}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try: thermal[(zone/'type').read_text().strip()]=int((zone/'temp').read_text())
        except (OSError,TypeError,ValueError): pass
    try: power_mode=subprocess.run(['nvpmodel','-q'],capture_output=True,text=True,timeout=5).stdout
    except (OSError,subprocess.TimeoutExpired): power_mode='unavailable in container'
    report=dict(machine_before=dict(host_meminfo=Path('/proc/meminfo').read_text(),thermal_millicelsius=thermal,power_mode=power_mode,cuda_free_total_bytes=torch.cuda.mem_get_info()),source_checkpoint=MODEL,full_file_sha256=checkpoint_sha256(MODEL),weight_reference='community compressed-tensors AWQ; not official BF16 model',
                torch=torch.__version__,tilelang=tilelang.__version__,device=str(device),seed=20261002,
                timing='median CUDA event time of 3 x 20 calls using replayed CUDA graph; graph output poisoning/input-change checks; compile excluded',
                cases=records,failures=[],scope='first-pass isolated full projections; not model TPS or full model validation')
    def write(): args.output.write_text(json.dumps(report,indent=2,default=str))
    def export_kernel(name, kernel):
        folder=args.output.parent/'compiled'; folder.mkdir(exist_ok=True)
        source=kernel.get_kernel_source(); (folder/(name+'.cu')).write_text(source)
        host=Path(kernel.adapter.lib_generator.pypath).read_text()
        (folder/(name+'-host.py.txt')).write_text(host)
        report.setdefault('compiled_kernels',[]).append(dict(name=name,cuda_source_sha256=hashlib.sha256(source.encode()).hexdigest(),host_abi_sha256=hashlib.sha256(host.encode()).hexdigest(),cuda_source=str(folder/(name+'.cu')),host_abi=str(folder/(name+'-host.py.txt'))))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for case in args.cases.split(','):
        try:
            inputs=activations(args.activations,case,args.layer)
            if case=='head':
                started=time.monotonic()
                with safe_open(MODEL,framework='pt',device='cpu') as f: original=f.get_tensor('lm_head.weight')
                identity=[dict(name='lm_head.weight',shape=list(original.shape),dtype=str(original.dtype),sha256=sha(original))]
                assert bool(torch.isfinite(original).all()) and float(original.abs().max()) < 65504
                b=original.cuda().half()
                cast_stats=dict(changed_values=0,underflow_to_zero=0,max_abs=0.,squared_error=0.,squared_reference=0.)
                for begin in range(0,len(b),1024):
                    source=original[begin:begin+1024].cuda().float(); converted=b[begin:begin+1024].float(); diff=converted-source
                    cast_stats['changed_values']+=int((converted!=source).sum())
                    cast_stats['underflow_to_zero']+=int(((converted==0)&(source!=0)).sum())
                    cast_stats['max_abs']=max(cast_stats['max_abs'],float(diff.abs().max()))
                    cast_stats['squared_error']+=float((diff*diff).sum()); cast_stats['squared_reference']+=float((source*source).sum())
                cast_stats['relative_l2']=(cast_stats['squared_error']/cast_stats['squared_reference'])**.5
                del original,source,converted,diff
                # FP16 follows native vLLM dtype; source-to-FP16 cast loss is explicit. Original
                # checkpoint tensor is separately identified; this is no W4 head.
                p=s=z=None; load_s=dict(head_read_hash_cast_s=time.monotonic()-started)
            else: p,s,z,b,identity,load_s=load_weights(case,args.layer)
            n,k=b.shape
            row=dict(case=case,layer=args.layer,N=n,K=k,weight_tensors=identity,offline_weight_preparation=load_s,
                     resident_weight_bytes=(b.numel()*2 if case=='head' else p.numel()+s.numel()*2+z.numel()),shapes=[])
            if case=='head': row['source_BF16_to_FP16_cast_error']=cast_stats
            records.append(row); write()
            scale_started=time.monotonic()
            scale_begin,scale_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            scale_begin.record()
            bs=(b.float().abs().amax(-1)/127).clamp_min(2**-24).half() if case!='head' else None
            scale_end.record(); scale_end.synchronize()
            row['offline_W8_row_scale_gpu_ms']=scale_begin.elapsed_time(scale_end)
            row['offline_W8_row_scale_host_s']=time.monotonic()-scale_started
            row['W8_scale_persistent_bytes']=0 if bs is None else bs.numel()*2
            dynamic=T.dynamic('M')
            small_w4=small_w8=small_aq=small_fp=small_gemv=None
            build_started=time.monotonic()
            if case=='head': small_fp=kernels.fp16_gemm(dynamic,n,k,16,64,64,2,128)
            else:
                small_gemv=kernels.w4_gemv(dynamic,n,k)
                small_w4=kernels.prefill(dynamic,n,k,'nk','ng','w4a16',16,64,128,2,128,0,True)
                small_w8=kernels.int8_gemm(dynamic,n,k,16,64,128,2,128)
                small_aq=kernels.activation_q8(dynamic,k)
                expand=kernels.expand_weight_q8_precomputed(n,k)
                make_lut=expand_lut=None
                try:
                    make_lut=kernels.weight_q8_lut(n,k)
                    expand_lut=kernels.expand_weight_q8_lut(n,k)
                except Exception as exc:
                    report['failures'].append(dict(case=case,candidate='temporary LUT expansion',error=repr(exc),traceback=traceback.format_exc()))
                    make_lut=expand_lut=None
            row['small_dynamic_compile_s']=time.monotonic()-build_started
            if case=='head': export_kernel(case+'-dynamic-fp16',small_fp)
            else:
                for tag,compiled in [('w4gemv',small_gemv),('w4mma',small_w4),('w8mma',small_w8),('a8',small_aq),('expand',expand),('lut',make_lut),('expandlut',expand_lut)]:
                    if compiled is not None: export_kernel(case+'-dynamic-'+tag,compiled)
            for m,cpu_a,origin in inputs:
                a=cpu_a.cuda().contiguous(); output=torch.empty((m,n),device='cuda',dtype=torch.float16)
                ref=a.float()@b.float().T
                shape=dict(M=m,activation=origin,tensor_sha256=sha(cpu_a),results=[])
                row['shapes'].append(shape); write()
                def control(): torch.mm(a,b.T,out=output)
                control(); shape['cublas_fp16_control']=dict(ms=graph_ms(control,output,a),error=error(ref,output))
                if case=='head':
                    def run(): small_fp(a,b,output)
                    run(); shape['results'].append(dict(route='TileLang full FP16 head from community BF16 tensor',ms=graph_ms(run,output,a),kernel_error=error(ref,output),workspace_bytes=output.numel()*2))
                    write(); continue
                if m < 512:
                    try:
                        def gemv(): small_gemv(a,p,s,z,output)
                        gemv(); e=error(ref,output); assert e['relative_l2']<.002,e
                        shape['results'].append(dict(route='TileLang W4A16 streaming GEMV',ms=graph_ms(gemv,output,a),kernel_error=e,workspace_bytes=output.numel()*2))
                    except Exception as exc: shape['results'].append(dict(route='TileLang W4A16 streaming GEMV',failure=repr(exc)))
                try:
                    started=time.monotonic()
                    w4=small_w4 if m<512 else kernels.prefill(m,n,k,'nk','ng','w4a16',64,64,128,2,128,0,True)
                    compile_s=time.monotonic()-started
                    if m==512: export_kernel(case+'-512-w4mma',w4)
                    def run4(): w4(a,p,s,z,torch.empty(0,device='cuda'),output)
                    # Avoid allocation during capture; W4A16 ABI includes unused AS.
                    dummy=torch.ones(m,device='cuda',dtype=torch.float16)
                    def run4(): w4(a,p,s,z,dummy,output)
                    run4(); e=error(ref,output); assert e['relative_l2']<.002,e
                    shape['results'].append(dict(route='TileLang W4A16',ms=graph_ms(run4,output,a),kernel_error=e,compile_s=compile_s,workspace_bytes=output.numel()*2))
                except Exception as exc: shape['results'].append(dict(route='TileLang W4A16',failure=repr(exc)))
                try:
                    started=time.monotonic()
                    w8=small_w8 if m<512 else kernels.int8_gemm(m,n,k,128,128,128,2,256)
                    aq=small_aq if m<512 else kernels.activation_q8(m,k)
                    compile_s=time.monotonic()-started
                    if m==512:
                        export_kernel(case+'-512-w8mma',w8); export_kernel(case+'-512-a8',aq)
                    allocation_started=time.monotonic()
                    expanded=torch.empty((n,k),device='cuda',dtype=torch.int8)
                    qa=torch.empty((m,k),device='cuda',dtype=torch.int8); asc=torch.empty(m,device='cuda',dtype=torch.float16)
                    allocation_s=time.monotonic()-allocation_started
                    def expansion(): expand(p,s,z,bs,expanded)
                    def quant(): aq(a,qa,asc)
                    def gemm(): w8(qa,expanded,asc,bs,output)
                    def chain(): expansion(); quant(); gemm()
                    chain()
                    ref8=(qa.float()*asc.float()[:,None])@(expanded.float()*bs.float()[:,None]).T
                    e=error(ref8,output); assert e['relative_l2']<.002,e
                    shape['results'].append(dict(route='TileLang W4 resident temporary W8A8',full_path_ms=graph_ms(chain,output,a),expansion_ms=graph_ms(expansion,expanded),activation_quant_ms=graph_ms(quant,qa),gemm_ms=graph_ms(gemm,output),kernel_error=e,extra_W8_A8_error=error(ref,ref8),A8_only_error=error(ref,(qa.float()*asc.float()[:,None])@b.float().T),W8_only_error=error(ref,a.float()@(expanded.float()*bs.float()[:,None]).T),weight_W8_error=error(b,expanded.float()*bs.float()[:,None]),total_error=error(ref,output),compile_s=compile_s,temporary_allocation_host_s=allocation_s,weight_row_scale_bytes=bs.numel()*2,workspace_bytes=expanded.numel()+qa.numel()+asc.numel()*2+output.numel()*2))
                    if expand_lut is not None:
                        lookup=torch.empty((n,k//128,16),device='cuda',dtype=torch.int8)
                        before=expanded.clone()
                        def lut(): make_lut(s,z,bs,lookup)
                        def lut_expansion(): expand_lut(p,lookup,expanded)
                        def fastchain(): lut(); lut_expansion(); quant(); gemm()
                        fastchain(); assert torch.equal(expanded,before), 'LUT changed W8 quantization codes'
                        shape['results'].append(dict(route='TileLang W4 resident temporary W8A8 LUT expansion',full_path_ms=graph_ms(fastchain,output,a),LUT_generation_ms=graph_ms(lut,lookup),expansion_ms=graph_ms(lut_expansion,expanded),activation_quant_ms=graph_ms(quant,qa),gemm_ms=graph_ms(gemm,output),kernel_error=error(ref8,output),extra_W8_A8_error=error(ref,ref8),total_error=error(ref,output),LUT_codes_exact=True,workspace_bytes=expanded.numel()+lookup.numel()+qa.numel()+asc.numel()*2+output.numel()*2,weight_row_scale_bytes=bs.numel()*2))
                        del before,lookup
                    del expanded,qa,asc,ref8
                except Exception as exc: shape['results'].append(dict(route='TileLang W4 resident temporary W8A8',failure=repr(exc),traceback=traceback.format_exc()[-2000:]))
                print(json.dumps(dict(case=case,M=m,results=shape['results'])),flush=True); write()
                del a,output,ref; gc.collect(); torch.cuda.empty_cache()
            del p,s,z,b,bs; gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            report['failures'].append(dict(case=case,error=repr(exc),traceback=traceback.format_exc())); write(); print(traceback.format_exc(),flush=True)
    report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(); write()

if __name__=='__main__': main()
