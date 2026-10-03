#!/usr/bin/env python3
"""Synthetic real-shape GDN recurrence validation and safe immutable-state timing."""
import argparse
import ast
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from common import (ROOT, benchmark, configure, environment, error, export_kernel,
                    identity, tensor_sha, write_json)
from gdn_reference import recurrent
from kernels.operators.op10_gdn_recurrent import gdn_recurrent, launch

NATIVE = Path('/home/nvidia/model/orin-kv8-mtp-20261001/vllm020/model_executor/layers/fla/ops/fused_recurrent.py')
NATIVE_OP = NATIVE.parent / 'op.py'
SCALE = 1 / math.sqrt(128)
BATCHES = (1, 2, 3, 4, 5, 7, 8)


def inputs(batch, dtype='float16', steps=None):
    shape = (batch, 16, 128) if steps is None else (steps, batch, 16, 128)
    q = torch.randn(shape, device='cuda', dtype=torch.float32)
    k = torch.randn_like(q)
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    if dtype == 'float32':
        q = q * SCALE
    q, k = q.to(getattr(torch, dtype)), k.to(getattr(torch, dtype))
    vshape = (batch,48,128) if steps is None else (steps,batch,48,128)
    v = (torch.randn(vshape,device='cuda') * .5).half()
    gateshape = vshape[:-1]
    g = -(torch.rand(gateshape,device='cuda') * .02 + .001)
    beta = torch.rand(gateshape,device='cuda') * .8 + .1
    state = torch.randn((batch,48,128,128),device='cuda') * .05
    return q,k,v,g,beta,state


def reference(data, scale, rounded=False):
    q,k,v,g,beta,state=data
    beta = beta.half().float() if rounded else beta
    o,s = recurrent(q.unsqueeze(2),k.unsqueeze(2),v.unsqueeze(2),g.unsqueeze(2),beta.unsqueeze(2),state,q_scale=scale)
    return o.squeeze(2),s


def check(output, state, expected, state_limit=.001):
    result = {'output':error(output,expected[0]),'state':error(state,expected[1])}
    assert result['output']['finite'] and result['state']['finite'], result
    assert result['output']['relative_l2'] <= .002, result
    assert result['state']['relative_l2'] <= state_limit, result
    return result


def runner(kernel,data,output,stateout):
    return lambda: launch(kernel,*data,stateout,output,stream=torch.cuda.current_stream().cuda_stream)


def native_kernel(out):
    shutil.copyfile(NATIVE,out/'native-fused-recurrent.py')
    shutil.copyfile(NATIVE_OP,out/'native-op.py')
    scope={'torch':torch,'triton':triton,'tl':tl,'__name__':'op10_locked_native'}
    # Locked op.py default FLA_USE_FAST_OPS=0 maps exp directly to tl.exp.
    scope['exp']=tl.exp
    tree=ast.parse(NATIVE.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fused_recurrent_gated_delta_rule_fwd_kernel')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(NATIVE),'exec'),scope)
    return scope['fused_recurrent_gated_delta_rule_fwd_kernel']


def run_native(kernel,data,scale):
    q,k,v,g,beta,state=data
    batch=q.shape[0]
    ns=state.transpose(-2,-1).contiguous()
    so=torch.empty_like(ns)
    o=torch.empty_like(v,dtype=q.dtype)
    kernel[(1,4,batch*48)](q,k,v,g,beta,o,ns,so,None,None,None,scale,batch,1,
        B=batch,H=16,HV=48,K=128,V=128,BK=128,BV=32,
        stride_init_state_token=48*128*128,stride_final_state_token=48*128*128,
        stride_indices_seq=1,stride_indices_tok=1,INPLACE_FINAL_STATE=False,
        IS_BETA_HEADWISE=False,USE_QK_L2NORM_IN_KERNEL=False,IS_KDA=False,
        num_warps=1,num_stages=3)
    torch.cuda.synchronize()
    return o,so.transpose(-2,-1).contiguous()


def export(kernel,dest,report,dtype,scale,tile,threads,rounded,output_dtype='float16'):
    exported=export_kernel(kernel,dest)
    source=(dest/'kernel.cu').read_text()
    decl=re.search(r'extern "C" __global__ void (\w+)\(([^;]+)\);',source)
    assert decl
    host=(dest/'host.txt').read_text()
    smem=re.search(r'config.sharedMemBytes = (\d+)',host)
    shared_bytes=int(smem.group(1)) if smem else None
    resource=subprocess.run(['cuobjdump','--dump-resource-usage',str(dest/'kernel.cubin')],capture_output=True,text=True)
    (dest/'resources.txt').write_text(resource.stdout+resource.stderr)
    abi={'operator':'op10_gdn_recurrent','sm':87,'entry_symbol':decl.group(1),
         'ordered_arguments':[{'index':i,'declaration':a.strip(),'driver_type':'device_ptr:u64' if '*' in a else 'int32'} for i,a in enumerate(decl.group(2).split(','))],
         'logical_buffers':{'Q':['B',16,128,dtype],'K':['B',16,128,dtype],
          'V':['B',48,128,'float16'],'G':['B',48,'float32'],'Beta':['B',48,'float32'],
          'StateIn':['B',48,128,128,'float32'],'StateOut':['B',48,128,128,'float32'],
          'Out':['B',48,128,output_dtype]},
         'layout':'contiguous row-major; state[K,V]; kh=vh//3',
         'launch':{'grid':[128//tile,'B*48',1],'block':[threads,1,1],'cooperative':False,
                   'dynamic_shared_memory_bytes':shared_bytes,'shared_memory_source':'actual host.txt and resources.txt'},
         'q_scale':scale,'beta_round_fp16':rounded,'output_dtype':output_dtype,
         'aliasing':'all buffers disjoint; immutable StateIn; stable addresses; explicit caller stream',
         'workspace_bytes':0,'state_input_bytes_per_request':48*128*128*4,
         'state_output_bytes_per_request':48*128*128*4,'artifacts':exported,
         'toolchain':report['environment'],'host_wrapper':host,'resources':resource.stdout,
         'actual_argument_order_source':'generated CUDA declaration and host wrapper'}
    write_json(dest/'abi.json',abi)
    return abi


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--long-lengths',default='512,2048,8192')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    configure()
    lock=ROOT/'artifacts/experimental-vllm/reference-lock.json'
    shutil.copyfile(lock,out/'reference-lock.json')
    report={'environment':environment(),'source':identity(ROOT/'kernels/operators/op10_gdn_recurrent.py'),
        'input_scope':'Synthetic normalized Q/K,V,g,beta and full 48-head FP32 states; no actual model state trace',
        'reference_lock':identity(lock),'native_sources':[identity(NATIVE),identity(NATIVE_OP)],'tuning':[],'cases':[],
        'native_comparisons':[],'state_tests':[],'long_chains':[],
        'mathematics':'decay=exp(g); Sd=decay*S; delta=beta*(v-k^T*Sd); Sn=Sd+k*delta^T; out=(q*q_scale)^T*Sn',
        'rounding':'FP32 state/arithmetic; Q/K input FP16 unscaled or FP32 Q already scaled; V FP16; g/beta FP32; explicit beta-round-FP16 variant',
        'budget_ms':.080,'workspace_bytes':0,'state_bytes_per_request':3145728,
        'state_read_write_bytes_per_step_per_request':6291456}
    cache={}
    def build(dtype='float16',tile=32,threads=128,rounded=False,odtype='float16'):
        key=(dtype,tile,threads,rounded,odtype)
        if key not in cache:
            started=time.perf_counter()
            cache[key]=gdn_recurrent(q_scale=SCALE if dtype=='float16' else 1.,qk_dtype=dtype,
                  value_tile=tile,threads=threads,beta_round_fp16=rounded,output_dtype=odtype)
            return cache[key],time.perf_counter()-started
        return cache[key],0
    for tile,threads in ((16,128),(32,128),(64,128),(128,256)):
        kernel,prepare=build(tile=tile,threads=threads)
        trial={'value_tile':tile,'threads':threads,'prepare_s':prepare,'timing':{}}
        for b in (1,8):
            data=inputs(b);o=torch.empty_like(data[2]);s=torch.empty_like(data[-1])
            run=runner(kernel,data,o,s)
            started=time.perf_counter();run();torch.cuda.synchronize()
            trial.setdefault('first_use_ms',{})[str(b)]=1000*(time.perf_counter()-started)
            trial.setdefault('error',{})[str(b)]=check(o,s,reference(data,SCALE))
            trial['timing'][str(b)],graph=benchmark(run,repetitions=30,calls_per_replay=16)
            del graph
        report['tuning'].append(trial);write_json(out/'progress.json',report)
        print(json.dumps({'tuning':trial}),flush=True)
    best=min(report['tuning'],key=lambda x:x['timing']['1']['median_ms'])
    tile,threads=best['value_tile'],best['threads'];report['selected']={'value_tile':tile,'threads':threads}
    native=native_kernel(out)
    for dtype,rounded in (('float16',False),('float16',True),('float32',False),('float32',True)):
        kernel,prepare=build(dtype,tile,threads,rounded)
        scale=SCALE if dtype=='float16' else 1.
        export(kernel,out/f'{dtype}-beta{16 if rounded else 32}',report,dtype,scale,tile,threads,rounded)
        for b in BATCHES:
            data=inputs(b,dtype);state_identity=tensor_sha(data[-1])
            o=torch.empty_like(data[2]);s=torch.empty_like(data[-1]);run=runner(kernel,data,o,s)
            run();torch.cuda.synchronize();expected=reference(data,scale,rounded)
            case={'B':b,'qk_dtype':dtype,'beta_round_fp16':rounded,'prepare_s':prepare,'error':check(o,s,expected),
               'state_in_sha256':state_identity,'state_read_write_bytes':b*6291456,
               'io_bytes':b*(2*16*128*(2 if dtype=='float16' else 4)+2*48*128*2+2*48*4)}
            case['timing'],graph=benchmark(run,repetitions=30,calls_per_replay=16)
            oldq=data[0].clone();olds=data[-1].clone();oldg=data[3].clone();oldbeta=data[4].clone()
            data[0].mul_(-.75);data[-1].add_(.125);data[3].sub_(.01);data[4].mul_(.5)
            o.fill_(float('nan'));s.fill_(float('nan'))
            graph.replay();torch.cuda.synchronize();case['graph_changed']=check(o,s,reference(data,scale,rounded))
            data[0].copy_(oldq);data[-1].copy_(olds);data[3].copy_(oldg);data[4].copy_(oldbeta)
            o.fill_(float('nan'));s.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            case['graph_restored']=check(o,s,expected);assert tensor_sha(data[-1])==state_identity
            report['cases'].append(case);del graph
            if b in (1,3):
                ndata=list(data);ndata[4]=data[4].half() if rounded else data[4]
                no,ns=run_native(native,ndata,scale)
                report['native_comparisons'].append({'B':b,'qk_dtype':dtype,'beta_round_fp16':rounded,
                  'native_output_dtype':str(no.dtype),'error_to_native':check(o,s,(no,ns)),
                  'native_to_FP32_math':check(no,ns,expected)})
        write_json(out/'progress.json',report)
    kernel,_=build('float16',tile,threads)
    # Rotate eight immutable per-layer states to exceed L2 capacity.
    cachedata=inputs(1);statepairs=[(cachedata[-1].clone(),torch.empty_like(cachedata[-1])) for _ in range(8)]
    cacheout=torch.empty_like(cachedata[2]);counter=[0]
    def rotate_run():
        sin,sout=statepairs[counter[0]%8];counter[0]+=1
        launch(kernel,*cachedata[:-1],sin,sout,cacheout,stream=torch.cuda.current_stream().cuda_stream)
    timing,graph=benchmark(rotate_run,repetitions=30,calls_per_replay=16);del graph
    report['cache_working_set']={'B':1,'distinct_state_pairs':8,'state_allocation_bytes':8*6291456,
        'timing':timing,'notes':'48MiB rotating immutable inputs/outputs; exceeds SM87 4MiB L2; no state reset copies inside timing'}
    del cachedata,statepairs,cacheout
    # Identity/extreme gate tests, full shape.
    for mode in ('zero_state','zero_beta','g_zero','high_decay'):
        data=list(inputs(3))
        if mode=='zero_state':data[-1].zero_()
        if mode=='zero_beta':data[4].zero_()
        if mode=='g_zero':data[3].zero_()
        if mode=='high_decay':data[3].fill_(-80)
        o=torch.empty_like(data[2]);s=torch.empty_like(data[-1]);runner(kernel,data,o,s)();torch.cuda.synchronize()
        report['state_tests'].append({'mode':mode,'B':3,'error':check(o,s,reference(data,SCALE))})
    # Two requests: isolation, split batch, checkpoint branch/restore.
    data=inputs(2);o=torch.empty_like(data[2]);s=torch.empty_like(data[-1]);runner(kernel,data,o,s)();torch.cuda.synchronize()
    expected=reference(data,SCALE);combined=(o.clone(),s.clone())
    for b in range(2):
        one=[x[b:b+1] for x in data];oo=torch.empty_like(one[2]);ss=torch.empty_like(one[-1])
        runner(kernel,one,oo,ss)();torch.cuda.synchronize()
        assert torch.equal(oo,combined[0][b:b+1]) and torch.equal(ss,combined[1][b:b+1])
    nextdata=list(inputs(2));nextdata[-1]=combined[1].clone();snapshot=nextdata[-1].clone()
    runner(kernel,nextdata,o,s)();torch.cuda.synchronize();branch=(o.clone(),s.clone())
    nextdata[0].neg_();runner(kernel,nextdata,o,s)();torch.cuda.synchronize()
    nextdata[0].neg_();nextdata[-1].copy_(snapshot);runner(kernel,nextdata,o,s)();torch.cuda.synchronize()
    assert torch.equal(o,branch[0]) and torch.equal(s,branch[1])
    report['state_tests'].append({'mode':'two_requests_isolation_branch_restore','B':2,'bitwise_split_batch':True,
                                'bitwise_branch_restore':True,'error':check(o,s,reference(nextdata,SCALE))})
    # Full-size synthetic long chains. Inputs are pre-generated; each step advances
    # by swapping two distinct buffers; these loops are validation, never benchmark.
    for length in [int(x) for x in args.long_lengths.split(',') if int(x)>0]:
        for dtype in ('float16','float32'):
            data=inputs(1,dtype,steps=length);q,k,v,g,beta,initial=data
            ck,_=build(dtype,tile,threads);scale=SCALE if dtype=='float16' else 1.
            a=initial.clone();b=torch.empty_like(a);o=torch.empty((1,48,128),device='cuda',dtype=torch.float16)
            outputs=torch.empty((length,1,48,128),device='cuda',dtype=torch.float16)
            checkpoints={length//2,length-1};saved={}
            started=time.perf_counter()
            for t in range(length):
                launch(ck,q[t],k[t],v[t],g[t],beta[t],a,b,o,stream=torch.cuda.current_stream().cuda_stream)
                outputs[t].copy_(o);a,b=b,a
                if t in checkpoints:saved[t]=a.clone()
            torch.cuda.synchronize();kernel_validation_s=time.perf_counter()-started
            started=time.perf_counter()
            ro,rs=recurrent(q.permute(1,2,0,3),k.permute(1,2,0,3),v.permute(1,2,0,3),
                            g.permute(1,2,0),beta.permute(1,2,0),initial,q_scale=scale)
            torch.cuda.synchronize()
            chain={'length':length,'B':1,'heads':48,'qk_dtype':dtype,
                   'error':check(outputs.permute(1,2,0,3),a,(ro,rs),state_limit=.005),
                   'kernel_validation_s':kernel_validation_s,'reference_validation_s':time.perf_counter()-started,
                   'input_tensor_sha256':[tensor_sha(x) for x in (q,k,v,g,beta,initial)]}
            # Recompute a suffix from saved midpoint: same state checkpoint and
            # token history must recover exactly (scheduler restore analogue).
            mid=length//2;a=saved[mid].clone();b=torch.empty_like(a)
            for t in range(mid+1,length):
                launch(ck,q[t],k[t],v[t],g[t],beta[t],a,b,o,stream=torch.cuda.current_stream().cuda_stream);a,b=b,a
            torch.cuda.synchronize();assert torch.equal(a,saved[length-1])
            chain['bitwise_midpoint_restore']=True;report['long_chains'].append(chain)
            write_json(out/'progress.json',report);print(json.dumps({'long_chain':chain}),flush=True)
            del data,q,k,v,g,beta,initial,a,b,o,outputs,ro,rs,saved
    # FP32 output interface for later op17 fusion; full-head owner remains optional.
    fk,prepare=build('float32',tile,threads,False,'float32');data=inputs(1,'float32')
    o=torch.empty_like(data[2],dtype=torch.float32);s=torch.empty_like(data[-1]);runner(fk,data,o,s)();torch.cuda.synchronize()
    report['fp32_output_case']=check(o,s,reference(data,1.))
    export(fk,out/'float32-output',report,'float32',1.,tile,threads,False,'float32')
    report['peak_validation_torch_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['status']='passed';report['budget_met_M1']=best['timing']['1']['median_ms']<=.080
    report['measurement_limitations']='CUDA graph repeats immutable state input with independent state output; 16 actual nodes/replay. Small batch fits L2 partially; long-chain validation is synthetic; no model state traces or model throughput claims.'
    write_json(out/'results.json',report)
    print(json.dumps({'status':'passed','selected':report['selected'],'cases':len(report['cases']),
      'long_chains':len(report['long_chains']),'M1_ms':best['timing']['1']['median_ms'],'budget_met':report['budget_met_M1']},indent=2),flush=True)


if __name__=='__main__':main()
