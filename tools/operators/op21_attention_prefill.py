"""Offline deterministic full-shape op21 correctness, graph, timing and ABI."""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path
import torch
from common import (ROOT, configure, environment, error, export_kernel,
                    benchmark, identity, tensor_sha, write_json)
from abi import parse_host
from kernels.operators.op21_attention_prefill import attention_prefill, validate_metadata

SOURCE = Path('/home/nvidia/model/orin-kv8-mtp-20261001/vllm020/model_executor/models/qwen3_next.py')


def reference(q,k,v,gate,pos,lengths,layout='token_major',kv_layout='head_major',
              output_layout='token_major',gate_mode='native_fp16',row_block=64):
    """Stable FP32 reference; bounded [B,24,row_block,Tkv] scores only.

    Production uses unnormalized FP16 P tensorcore tiles, whose approximation
    is reported against this FP32 reference rather than hidden in its baseline.
    """
    if layout == 'token_major': q=q.permute(0,2,1,3); gate=gate.permute(0,2,1,3)
    if kv_layout == 'token_major': k=k.permute(0,2,1,3); v=v.permute(0,2,1,3)
    b,h,t,d=q.shape
    index=torch.arange(24,device=q.device)//6
    kf=k.float()[:,index]; vf=v.float()[:,index]
    result=torch.empty_like(q)
    keypos=torch.arange(k.shape[2],device=q.device)[None,None,None,:]
    for start in range(0,t,row_block):
        end=min(t,start+row_block)
        scores=torch.matmul(q[:,:,start:end].float(),kf.transpose(-1,-2))*0.0625
        mask=(keypos <= pos[:,None,start:end,None]) & (keypos < lengths[:,None,None,None])
        scores.masked_fill_(~mask,-float('inf'))
        maximum=scores.amax(-1,keepdim=True)
        maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
        probs=torch.exp(scores-maximum)
        probs=probs/probs.sum(-1,keepdim=True).clamp_min(1e-30)
        a=torch.matmul(probs,vf)
        g=torch.sigmoid(gate[:,:,start:end].float())
        if gate_mode == 'native_fp16': result[:,:,start:end]=(a.half()*g.half())
        else: result[:,:,start:end]=(a*g).half()
    return result.permute(0,2,1,3).contiguous() if output_layout == 'token_major' else result


def metadata_cpu_checks():
    validate_metadata([[-1,0,511]],[512],512)
    validate_metadata([[-1,-1]],[0],512)
    rejected=0
    for args in [([[0]],[513],512),([[0]],[-1],512),([[-2]],[1],512),([[0,1],[0]],[1,1],512)]:
        try: validate_metadata(*args)
        except ValueError: rejected+=1
    assert rejected==4
    return {'valid_empty_zero':True,'invalid_cases_rejected':rejected}


def inputs(b,tq,tkv,kind,layout,kv_layout):
    q=torch.randn((b,tq,24,256),device='cuda',dtype=torch.float16)
    g=torch.randn_like(q)
    k=torch.randn((b,4,tkv,256),device='cuda',dtype=torch.float16)
    v=torch.randn_like(k)
    pos=(torch.arange(tq,device='cuda',dtype=torch.int32)+max(0,tkv-tq))[None].repeat(b,1)
    lengths=torch.full((b,),tkv,device='cuda',dtype=torch.int32)
    if kind in ('ragged','prefix_shared'):
        for j in range(b):
            lengths[j]=max(0,tkv-j*5-1)
            pos[j]=torch.arange(tq,device='cuda',dtype=torch.int32)+j%3
            if kind=='prefix_shared': pos[j].add_(max(0,tkv-tq))
        if b>1: k[1]=k[0]; v[1]=v[0]  # shared immutable prefix, private causal tails
        pos[0,0]=-1
    elif kind=='strong': q.mul_(12); k.mul_(12); g.mul_(10)
    elif kind=='zero': q.zero_(); k.zero_(); g.zero_()
    elif kind=='empty': lengths.zero_(); pos.fill_(-1)
    elif kind=='mixed_empty': lengths[0]=0; pos[0].fill_(-1)
    elif kind=='head_isolation':
        q.zero_(); k.zero_()
        for h in range(4): v[:,h].fill_(h+1)
    elif kind=='nonmonotonic':
        pos=pos.flip(1).contiguous()
        pos[:,0]=-1
    if layout=='head_major': q=q.permute(0,2,1,3).contiguous(); g=g.permute(0,2,1,3).contiguous()
    if kv_layout=='token_major': k=k.permute(0,2,1,3).contiguous(); v=v.permute(0,2,1,3).contiguous()
    return [q,k,v,g,pos,lengths]


def run_case(output,b,tq,tkv,kind='random',layout='token_major',kv_layout='head_major',
             out_layout='token_major',mode='native_fp16',bm=32,bn=32,do_bench=False,
             dynamic_batch=False):
    label=f'b{b}_q{tq}_kv{tkv}_{kind}_{layout}_{kv_layout}_{out_layout}_{mode}_m{bm}n{bn}'
    if dynamic_batch: label += '_dynamic_batch'
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    args=inputs(b,tq,tkv,kind,layout,kv_layout)
    validate_metadata(args[4].cpu().tolist(),args[5].cpu().tolist(),tkv)
    y=torch.empty((b,tq,24,256) if out_layout=='token_major' else (b,24,tq,256),device='cuda',dtype=torch.float16)
    torch.cuda.synchronize(); prepare=time.perf_counter()-start
    start=time.perf_counter()
    kernel=attention_prefill(None if dynamic_batch else b,tq,tkv,layout,out_layout,kv_layout,mode,bm,bn)
    torch.cuda.synchronize(); compile_s=time.perf_counter()-start
    def run(): kernel(*args,y,stream=torch.cuda.current_stream().cuda_stream)
    start=time.perf_counter(); run(); torch.cuda.synchronize(); first_s=time.perf_counter()-start
    ref=reference(*args,layout,kv_layout,out_layout,mode)
    numerical=error(y,ref)
    assert numerical['finite'] and numerical['relative_l2'] <= .002,(label,numerical)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): run()
    frozen=[a.clone() for a in args]
    # Every data tensor and both metadata tensors change after capture.
    args[0].mul_(.75); args[1].mul_(.875); args[2].add_(.25); args[3].add_(.5)
    args[4].sub_(3).clamp_(min=-1); args[5].sub_(2).clamp_(min=0)
    changed=reference(*args,layout,kv_layout,out_layout,mode)
    y.fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
    replay_error=error(y,changed)
    assert replay_error['finite'] and replay_error['relative_l2'] <= .002,(label,'changed graph',replay_error)
    for a,saved in zip(args,frozen): a.copy_(saved)
    y.fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
    restored=error(y,ref)
    assert restored['finite'] and restored['relative_l2'] <= .002
    timing=None
    if do_bench:
        timing,_=benchmark(run,repetitions=5,warmup=2,calls_per_replay=1)
    dest=output/label
    exported=export_kernel(kernel,dest)
    abi=parse_host((dest/'host.txt').read_text())
    report={'case':label,'shape':{'B':b,'Hq':24,'Hkv':4,'D':256,'Tq':tq,'Tkv':tkv},
        'input_layout':layout,'kv_layout':kv_layout,'output_layout':out_layout,
        'gate_mode':mode,'numerical':numerical,'graph_changed':replay_error,
        'batch_specialization':'dynamic' if dynamic_batch else 'static',
        'graph_restored':restored,'timing':timing,'prepare_s':prepare,
        'compile_s':compile_s,'first_call_s':first_s,'workspace_bytes':0,
        'tensor_resident_bytes':sum(a.numel()*a.element_size() for a in args)+y.numel()*y.element_size(),
        'validation_peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'source_inputs':{'Q':tensor_sha(args[0]),'K':tensor_sha(args[1]),'V':tensor_sha(args[2]),'RawGate':tensor_sha(args[3])},
        'export':exported,'actual_abi':abi,'target':'sm_87',
        'rounding':'FP16 Q/K/V; QK FP32; online exp/sum/rescale FP32; unnormalized P FP16; PV FP32; native attention FP16 + sigmoid FP16 + product FP16',
        'cache_contract':'contiguous KV only; absolute keys 0..Tkv-1, kh=qh//6; empty -> zero',
        'error_threshold_note':'0.002 is operator implementation diagnostic, not model/KV quality'}
    write_json(dest/'manifest.json',report)
    print(json.dumps({'case':label,'error':numerical,'timing':timing}),flush=True)
    del kernel,args,y,ref,changed,frozen,graph
    gc.collect(); torch.cuda.empty_cache()
    return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--tune',action='store_true')
    parser.add_argument('--followup',action='store_true')
    opt=parser.parse_args(); output=Path(opt.output); output.mkdir(parents=True,exist_ok=True)
    configure(); started=time.perf_counter()
    refdir=output/'reference-source'; refdir.mkdir(exist_ok=True)
    shutil.copyfile(SOURCE,refdir/'qwen3_next.py')
    source=SOURCE.read_text()
    assert 'gate = torch.sigmoid(gate)' in source and 'attn_output = attn_output * gate' in source
    summary={'environment':environment(),'host_metadata':metadata_cpu_checks(),
        'reference':identity(refdir/'qwen3_next.py'),'source_freeze':[identity(ROOT/'kernels/operators/op21_attention_prefill.py'),identity(ROOT/'tools/operators/op21_attention_prefill.py')],
        'scope':'isolated full-size attention operator, no model or paged runtime integration','cases':[]}
    def case(*args,**kwargs):
        summary['cases'].append(run_case(output,*args,**kwargs)); write_json(output/'results.json',summary)
    case(1,31,65)
    if opt.smoke: return
    if opt.followup:
        for b in (1,2,3,4,5,7,8): case(b,33,67,'prefix_shared',dynamic_batch=True)
        case(2,33,67,'nonmonotonic')
        return
    if opt.tune:
        for bm,bn in [(16,64),(32,32),(64,32),(64,64)]: case(1,512,512,bm=bm,bn=bn,do_bench=True)
        return
    for t in (511,512,513,2048,8192): case(1,t,t,do_bench=True)
    for b in (1,2,3,4,5,7,8): case(b,33,67,'ragged')
    case(1,64,512,do_bench=True); case(1,64,8448,do_bench=True)
    for kind in ('strong','zero','empty','head_isolation'): case(1,33,67,kind)
    case(3,33,67,'mixed_empty')
    case(2,33,67,'ragged',layout='head_major',out_layout='head_major')
    case(2,33,67,'ragged',kv_layout='token_major')
    case(1,33,67,mode='fp32_fused')
    case(2,33,67,'nonmonotonic')
    for b in (1,2,3,4,5,7,8): case(b,33,67,'prefix_shared',dynamic_batch=True)
    cubin_hashes={c['export']['files'][2]['sha256'] for c in summary['cases']
                  if c['batch_specialization']=='dynamic'}
    assert len(cubin_hashes)==1, 'dynamic cases must share one identical cubin'
    summary['dynamic_batch_shared_cubin_sha256']=next(iter(cubin_hashes))
    summary['wall_s']=time.perf_counter()-started; write_json(output/'results.json',summary)


if __name__=='__main__': main()
