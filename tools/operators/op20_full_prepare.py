"""Reproducible op20 isolated numerical/state/graph/ABI validation."""
import argparse
import ast
import gc
import json
import re
import shutil
import time
import types
from pathlib import Path
import torch
from safetensors import safe_open
from common import (ROOT, configure, environment, error, export_kernel, benchmark,
                    identity, tensor_sha, write_json)
from abi import parse_host
from kernels.operators.op20_full_prepare import (full_prepare, reset_metadata,
    validate_metadata, validate_host_metadata, bandwidth_copy)

MODEL = Path('/home/nvidia/model/vllm-comparison-20260930/awq-http')
SOURCE = Path('/home/nvidia/model/orin-kv8-mtp-20261001/vllm020')
B, MP, NP, BS, MAXPOS = 8, 68, 548, 128, 8576


def methods(path, klass, names, namespace):
    tree = ast.parse(path.read_text())
    container = tree.body if klass is None else next(n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == klass).body
    nodes = [n for n in container if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), 'exec'), namespace)


def binding(output):
    started = time.perf_counter()
    files = ['model_executor/models/qwen3_next.py', 'model_executor/models/qwen3_5.py',
        'model_executor/layers/layernorm.py', 'ir/ops/layernorm.py',
        'model_executor/layers/rotary_embedding/base.py',
        'model_executor/layers/rotary_embedding/common.py',
        'model_executor/layers/rotary_embedding/mrope_interleaved.py',
        'model_executor/layers/rotary_embedding/__init__.py']
    frozen = output/'reference-source'
    for relative in files:
        dest = frozen/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/relative, dest)
    config = json.loads((MODEL/'config.json').read_text())['text_config']
    assert (config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']) == (24,4,256)
    assert config['rope_parameters']['rope_theta'] == 10000000
    assert config['rope_parameters']['mrope_section'] == [11,11,10]
    assert config['partial_rotary_factor'] == .25 and config['rms_norm_eps'] == 1e-6
    lock = json.loads((ROOT/'artifacts/reference/reference-lock.json').read_text())
    checkpoint = next(v for v in lock['files'] if v['name'] == 'model.safetensors')
    assert checkpoint['bytes'] == (MODEL/'model.safetensors').stat().st_size
    weights, identities = {}, []
    with safe_open(str(MODEL/'model.safetensors'), framework='pt', device='cpu') as f:
        for layer in range(3,64,4):
            pair = []
            for which in ('q','k'):
                name = f'model.language_model.layers.{layer}.self_attn.{which}_norm.weight'
                raw = f.get_tensor(name); half = raw.half().contiguous()
                assert raw.shape == (256,) and torch.equal(raw, half.to(raw.dtype))
                identities.append({'name':name, 'source_dtype':str(raw.dtype),
                    'source_tensor_sha256':tensor_sha(raw), 'fp16_sha256':tensor_sha(half),
                    'conversion':'exact BF16 to FP16', 'shape':[256]})
                pair.append(half.cuda())
            weights[layer] = pair
    ns = {'torch':torch, 'Tensor':torch.Tensor}
    methods(frozen/'ir/ops/layernorm.py', None, ['rms_norm'], ns)
    ns['ir'] = types.SimpleNamespace(ops=types.SimpleNamespace(rms_norm=ns['rms_norm']))
    methods(frozen/'model_executor/layers/layernorm.py', 'GemmaRMSNorm', ['forward_native'], ns)
    native_norm = ns['forward_native']
    ropepath = frozen/'model_executor/layers/rotary_embedding'
    methods(ropepath/'common.py', 'ApplyRotaryEmb', ['forward_static'], ns)
    static = ns['forward_static']
    methods(ropepath/'base.py', 'RotaryEmbeddingBase', ['_compute_inv_freq','_compute_cos_sin_cache'], ns)
    instance = types.SimpleNamespace(base=1e7, rotary_dim=64, max_position_embeddings=MAXPOS)
    instance._compute_inv_freq = types.MethodType(ns['_compute_inv_freq'], instance)
    # Explicit CPU FP32 native cache preparation, then exact FP16 upload.
    cpu_cache = ns['_compute_cos_sin_cache'](instance).half()
    with torch.device('cuda'):
        cache = ns['_compute_cos_sin_cache'](instance).half()
    cache_cpu_delta = error(cache, cpu_cache.cuda())
    methods(ropepath/'mrope_interleaved.py', 'MRotaryEmbeddingInterleaved',
            ['get_mrope_interleaved_id_list','_rebuild_pos_emb','forward'], ns)
    inst = types.SimpleNamespace(cos_sin_cache=cache, head_size=256, rotary_dim=64,
        mrope_dim=ns['get_mrope_interleaved_id_list'](11,11,10,True)*2,
        apply_rotary_emb=types.SimpleNamespace(forward_native=lambda x,c,s:static(x,c,s,True,False)))
    inst._rebuild_pos_emb = types.MethodType(ns['_rebuild_pos_emb'], inst)
    native_rope = ns['forward']
    def native(x, pos, wq, wk):
        qg = x[:,:12288].reshape(-1,24,512)
        q = native_norm(types.SimpleNamespace(weight=wq, variance_epsilon=1e-6), qg[:,:,:256])
        k = native_norm(types.SimpleNamespace(weight=wk, variance_epsilon=1e-6), x[:,12288:13312].reshape(-1,4,256))
        qr, kr = native_rope(inst, pos.long().unsqueeze(0).expand(3,-1), q, k)
        return qr, qg[:,:,256:].contiguous(), kr, x[:,13312:].reshape(-1,4,256), q, k
    info = {'checkpoint':dict(checkpoint, path=str(MODEL/'model.safetensors'),
            identity_policy='reuse locked whole-file hash, size checked'),
        'config':identity(MODEL/'config.json'), 'reference_lock':identity(ROOT/'artifacts/reference/reference-lock.json'),
        'sources':[identity(frozen/f) for f in files], 'all_16_full_attention_weight_pairs':identities,
        'projection_input_origin':'synthetic seeded FP16 X[M,14336]; capture-*.pt are 5120 projection INPUTS, not usable outputs',
        'native':'AST executes frozen GemmaRMSNorm.forward_native and MRotaryEmbeddingInterleaved.forward/_rebuild_pos_emb plus ApplyRotaryEmb.forward_static',
        'cache':{'shape':[MAXPOS,64], 'dtype':'float16', 'tensor_sha256':tensor_sha(cache),
            'preparation':'native GPU FP32 theta1e7 cache then FP16; runtime binding explicitly frozen',
            'cpu_native_cache_sha256':tensor_sha(cpu_cache),'cpu_vs_gpu':cache_cpu_delta},
        'preparation_s':time.perf_counter()-started}
    write_json(output/'binding.json', info)
    return weights, cache, native, info


def math_reference(x, pos, wq, wk, cache, rounds=True, norm_rounds=None, rope_rounds=None):
    norm_rounds=rounds if norm_rounds is None else norm_rounds
    rope_rounds=rounds if rope_rounds is None else rope_rounds
    qg = x[:,:12288].reshape(-1,24,512)
    def norm(z,w):
        z=z.float(); n=(z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6))*(1+w.float())
        return n.half() if norm_rounds else n
    q=norm(qg[:,:,:256],wq); k=norm(x[:,12288:13312].reshape(-1,4,256),wk)
    def rope(z):
        c,s=cache[pos.long()].chunk(2,-1); c=c[:,None].float(); s=s[:,None].float()
        z1,z2=z[:,:,:32].float(),z[:,:,32:64].float()
        if rope_rounds:
            a=(z1*c).half(); b=(z2*s).half(); d=(z2*c).half(); e=(z1*s).half()
            rot=torch.cat(((a.float()-b.float()).half(),(d.float()+e.float()).half()),-1)
        else:
            rot=torch.cat((z1*c-z2*s,z2*c+z1*s),-1)
        return torch.cat((rot,z[:,:,64:]),-1).half()
    return rope(q),qg[:,:,256:].contiguous(),rope(k),x[:,13312:].reshape(-1,4,256),q,k


def metadata(m, boundary=False):
    # Permute physical pages; request lengths ragged; row order is shuffled.
    table=torch.randperm(B*MP,dtype=torch.int64).reshape(B,MP).int()
    req=(torch.arange(m)*torch.arange(1,m+1)%B).int()
    counters=[0]*B; positions=[]
    for r in req.tolist():
        positions.append(127+counters[r]); counters[r]+=1
    if boundary:
        req=torch.arange(m).remainder(B).int()
        positions=[(0,63,127,128,8191,8447,64,129)[i%8] for i in range(m)]
    perm=torch.randperm(m)
    req=req[perm]; pos=torch.tensor(positions,dtype=torch.int32)[perm]
    slots=validate_host_metadata(req.tolist(),pos.tolist(),table.tolist(),NP,BS,MAXPOS)
    return req.cuda(),pos.cuda(),table.cuda(),slots


def bitexact(a,b):
    return torch.equal(a.view(torch.int16),b.view(torch.int16))


def exports(kernel, output, name):
    dest=output/'aot'/name
    artifact=export_kernel(kernel,dest)
    source=(dest/'kernel.cu').read_text(); host=(dest/'host.txt').read_text()
    abi={'operator':'op20_full_prepare','variant':name,'actual_launches':parse_host(host),
        'actual_cuda_declarations':[entry for entry in re.findall(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)',source) if entry[0] in artifact['symbols']],
        'tensor_api_order':{'fused':['X','WQ','WK','Cache','Req','Pos','Pages','Status','Q','Gate','K','V'],
            'copy':['X','Y'],'reset':['Owner','Status'],'validate':['Req','Pos','Pages','Owner','Status']}[name],
        'layout':{'X':'[M,14336]FP16 head-interleaved Q256,gate256 then K1024,V1024',
            'Q/Gate':'[M,24,256]FP16','K/V':'independent [548,128,4,256]FP16',
            'Pages':'[8,68]int32','Req/Pos':'[M]int32','Cache':'[8576,64]FP16 cos32,sin32',
            'WQ/WK':'[256]FP16','Status':'[1]int32','Owner':'[70144]int32'},
        'shape_parameters':{'batch':B,'max_pages':MP,'num_pages':NP,'block_size':BS,'max_position':MAXPOS,'M':'dynamic int32'},
        'cooperative_launch':False,'sm':87,'workspace_bytes':0 if name=='copy' else NP*BS*4+4,
        'persistent_norm_bytes':1024 if name=='fused' else 0,'persistent_rope_bytes':MAXPOS*64*2 if name=='fused' else 0,
        'stream':'explicit capture-current stream per invocation; reset/validate/fused/commit strictly same stream',
        'alias':'all tensor buffers disjoint; stable graph addresses; allocator exclusive writable pages',
        'state':'no seqLens update; status!=0 inhibits every output/KV write; Rust handles status and commits position',
        'toolchain':environment(),'files':artifact['files']}
    write_json(dest/'abi.json',abi)
    return abi


def run_case(m, kernels, weights, cache, native, repetitions, boundary=False):
    reset, validate, fused=kernels
    req,pos,pages,slots=metadata(m,boundary)
    x=torch.randn((m,14336),device='cuda',dtype=torch.float16)
    # Deliberately different gate and Q scales makes layout bugs conspicuous.
    x[:,:12288].reshape(m,24,512)[:,:,256:].mul_(3)
    wq,wk=weights
    q=torch.empty((m,24,256),device='cuda',dtype=torch.float16); gate=torch.empty_like(q)
    k=torch.full((NP,BS,4,256),-37.25,device='cuda',dtype=torch.float16); v=torch.full_like(k,19.5)
    owner=torch.empty(NP*BS,device='cuda',dtype=torch.int32); status=torch.empty(1,device='cuda',dtype=torch.int32)
    def core():
        fused(x,wq,wk,cache,req,pos,pages,status,q,gate,k,v,stream=torch.cuda.current_stream().cuda_stream)
    def checked():
        stream=torch.cuda.current_stream().cuda_stream
        reset(owner,status,stream=stream); validate(req,pos,pages,owner,status,stream=stream); core()
    start=time.perf_counter(); checked(); torch.cuda.synchronize(); first=time.perf_counter()-start
    assert status.item()==0
    expected=native(x,pos,wq,wk)
    math=math_reference(x,pos,wq,wk,cache)
    flatk=k.reshape(-1,4,256); flatv=v.reshape(-1,4,256)
    idx=torch.tensor(slots,device='cuda',dtype=torch.long)
    def check(ref):
        qe=error(q,ref[0]); ke=error(flatk[idx],ref[2])
        assert qe['finite'] and ke['finite'] and qe['relative_l2']<.001 and ke['relative_l2']<.001
        assert bitexact(gate,ref[1]) and bitexact(flatv[idx],ref[3])
        return {'Q':qe,'K':ke,'gate_raw_bitexact':True,'V_bitexact':True}
    observed=check(expected)
    assert all(bitexact(a,b) for a,b in zip(expected,math)), 'independent staged math vs AST native mismatch'
    untouched=torch.ones(NP*BS,device='cuda',dtype=torch.bool); untouched[idx]=False
    assert (flatk[untouched]==-37.25).all() and (flatv[untouched]==19.5).all()
    native_norm_q=error(q[:,:,64:],expected[4][:,:,64:]); native_norm_k=error(flatk[idx,:,64:],expected[5][:,:,64:])
    unrounded=math_reference(x,pos,wq,wk,cache,False)
    rounding_effect={'all_fp32':{'Q':error(unrounded[0],expected[0]),'K':error(unrounded[2],expected[2])}}
    del unrounded
    for label,nround,rround in [('unrounded_norm',False,True),('unrounded_rope',True,False)]:
        alternate=math_reference(x,pos,wq,wk,cache,norm_rounds=nround,rope_rounds=rround)
        rounding_effect[label]={'Q':error(alternate[0],expected[0]),'K':error(alternate[2],expected[2])}
        del alternate
    del math
    hot,graph=benchmark(core,repetitions=repetitions,calls_per_replay=16 if m<9 else 1)
    checked_hot,_=benchmark(checked,repetitions=repetitions,calls_per_replay=16 if m<9 else 1)
    # Capture includes all three production stages. Same addresses, changed data.
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): checked()
    original=q.clone(); xsave=x.clone(); psave=pos.clone(); tablesave=pages.clone()
    x.mul_(-.375); pos.add_(1)
    # shift ALL physical page ids: recompute unique valid destinations without alias.
    pages.copy_((pages+1)%(B*MP))
    newslots=validate_host_metadata(req.cpu().tolist(),pos.cpu().tolist(),pages.cpu().tolist(),NP,BS,MAXPOS)
    k.fill_(-37.25); v.fill_(19.5); q.fill_(float('nan')); gate.fill_(float('nan'))
    graph.replay(); torch.cuda.synchronize(); assert status.item()==0
    changed=native(x,pos,wq,wk); newidx=torch.tensor(newslots,device='cuda')
    assert error(q,changed[0])['relative_l2']<.001 and error(flatk[newidx],changed[2])['relative_l2']<.001
    assert bitexact(gate,changed[1]) and bitexact(flatv[newidx],changed[3]) and not bitexact(q,original)
    newuntouched=torch.ones(NP*BS,device='cuda',dtype=torch.bool); newuntouched[newidx]=False
    assert (flatk[newuntouched]==-37.25).all() and (flatv[newuntouched]==19.5).all()
    # Device-only metadata mutation invalidates previously CPU-checked metadata.
    invalids=[]
    x.copy_(xsave);pos.copy_(psave);pages.copy_(tablesave)
    for kind in ('negative_req','large_req','negative_position','large_position','invalid_page','duplicate_writer','alias_writer'):
        reqsave=req.clone(); pos.copy_(psave); pages.copy_(tablesave)
        if kind=='negative_req':req[0]=-1
        elif kind=='large_req':req[0]=B
        elif kind=='negative_position':pos[0]=-1
        elif kind=='large_position':pos[0]=MAXPOS
        elif kind=='invalid_page':pages[req[0],pos[0]//BS]=NP
        elif kind=='alias_writer' and m>1:
            req[1]=(req[0]+1)%B;pos[1]=pos[0]
            pages[req[1],pos[1]//BS]=pages[req[0],pos[0]//BS]
        elif m>1:req[1]=req[0];pos[1]=pos[0]
        else:continue
        k.fill_(-37.25);v.fill_(19.5);q.fill_(7.25);gate.fill_(-8.5)
        graph.replay();torch.cuda.synchronize()
        code=status.item();assert code!=0
        assert (k==-37.25).all() and (v==19.5).all() and (q==7.25).all() and (gate==-8.5).all()
        invalids.append({'kind':kind,'status':code,'all_writes_inhibited':True});req.copy_(reqsave)
    pos.copy_(psave);pages.copy_(tablesave);q.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
    assert status.item()==0 and bitexact(q,original)
    budget=.01 if m<9 else .1*m/512
    # X bytes + Q/gate bytes + K/V bytes; not entire KV-capacity traversal.
    traffic=m*(14336+2*6144+2*1024)*2
    return {'M':m,'weights_layer':None,'boundary_positions':boundary,
        'metadata':'ragged mixed request ids, shuffled row order, shuffled physical pages',
        'reference_errors':observed,'norm_unrotated_errors':{'Q':native_norm_q,'K':native_norm_k},
        'staged_math_vs_native_bitexact':True,'fp32_fusion_vs_native_rounding_effect':rounding_effect,
        'untouched_slots_bitexact':True,'graph':{'X_position_page_mutation':True,'poison_outputs_and_KV':True,
        'changed_destination_untouched_sentinel':True,'restored_Q_bitexact':True},
        'device_invalid_rejected_without_writes':invalids,'first_launch_s':first,'hot_fused':hot,'hot_checked':checked_hot,
        'budget_ms':budget,'fused_budget_met':hot['median_ms']<=budget,'checked_budget_met':checked_hot['median_ms']<=budget,
        'minimum_io_bytes':traffic,'effective_minimum_io_GB_s':traffic/hot['median_ms']/1e6,
        'budget_requires_GB_s':traffic/budget/1e6,'workspace_bytes':NP*BS*4+4,
        'KV_capacity_bytes':2*NP*BS*4*256*2,'query_gate_bytes':m*24*256*2*2}


def continuity_and_weight_sweep(kernels,weights,cache,native):
    reset,validate,fused=kernels
    m=513;req,pos,pages,slots=metadata(m)
    x=torch.randn((m,14336),device='cuda',dtype=torch.float16)
    q=torch.empty((m,24,256),device='cuda',dtype=torch.float16);gate=torch.empty_like(q)
    k=torch.full((NP,BS,4,256),-37.25,device='cuda',dtype=torch.float16);v=torch.full_like(k,19.5)
    owner=torch.empty(NP*BS,device='cuda',dtype=torch.int32);status=torch.empty(1,device='cuda',dtype=torch.int32)
    wq,wk=weights[3]
    def apply(begin,end,qbuf,gbuf):
        stream=torch.cuda.current_stream().cuda_stream
        reset(owner,status,stream=stream)
        validate(req[begin:end],pos[begin:end],pages,owner,status,stream=stream)
        fused(x[begin:end],wq,wk,cache,req[begin:end],pos[begin:end],pages,status,qbuf,gbuf,k,v,stream=stream)
        assert status.item()==0
    apply(0,m,q,gate);qbase=q.clone();gbase=gate.clone();kbase=k.clone();vbase=v.clone()
    tested=[]
    for sizes in ([1]*m,[127,1,128,1,256],[511,2]):
        k.fill_(-37.25);v.fill_(19.5);q.fill_(float('nan'));gate.fill_(float('nan'));start=0
        for size in sizes:
            apply(start,start+size,q[start:start+size],gate[start:start+size]);start+=size
        assert start==m and bitexact(q,qbase) and bitexact(gate,gbase) and bitexact(k,kbase) and bitexact(v,vbase)
        tested.append({'chunk_sizes':sizes if len(sizes)<10 else '513 sequential M1 calls','Q_gate_KV_bitexact':True})
    idx=torch.tensor(slots[:5],device='cuda')
    sweep=[]
    for layer,pair in weights.items():
        wq,wk=pair;apply(0,5,q[:5],gate[:5]);expected=native(x[:5],pos[:5],wq,wk)
        qe=error(q[:5],expected[0]);ke=error(k.reshape(-1,4,256)[idx],expected[2])
        assert qe['relative_l2']<.001 and ke['relative_l2']<.001
        sweep.append({'layer':layer,'Q':qe,'K':ke})
    return {'M':m,'chunk_and_sequential_results':tested,'all_16_weight_pairs':sweep,
        'scope':'isolated preparation with absolute position metadata, full-versus-chunked KV updates; Rust prefix/COW integration not implemented'}


def cpu_rejections():
    table=[[0,1],[2,3]]
    cases=[([-1],[0],table),([2],[0],table),([0],[-1],table),([0],[256],table),
        ([0],[0],[[4,1],[2,3]]),([0,0],[3,3],table),([0,1],[0,0],[[0,1],[0,3]]),
        ([True],[0],table),([0],[0.5],table)]
    rejected=[]
    for r,p,t in cases:
        try:validate_host_metadata(r,p,t,4)
        except ValueError as exc:rejected.append(str(exc))
        else:raise AssertionError('CPU invalid metadata accepted')
    return rejected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--rows',default='1,2,3,4,5,7,8,511,512,513,2048,8192')
    parser.add_argument('--repetitions',type=int,default=10)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    configure(); weights,cache,native,bind=binding(output)
    results={'environment':environment(),'binding':bind,'cpu_rejections':cpu_rejections(),
        'cases':[],'exports':[],'status':'in_progress'}
    kernels=[]
    for name,build in [('reset',lambda:reset_metadata(NP)),('validate',lambda:validate_metadata(B,MP,NP,max_position=MAXPOS)),
                       ('fused',lambda:full_prepare(B,MP,NP,max_position=MAXPOS))]:
        start=time.perf_counter();kernel=build();elapsed=time.perf_counter()-start;kernels.append(kernel)
        results['exports'].append({'name':name,'prepare_compile_s':elapsed,'abi':exports(kernel,output,name)})
        write_json(output/'results.json',results)
    for m in [int(v) for v in args.rows.split(',')]:
        layer=3 if m%2 else 31
        result=run_case(m,kernels,weights[layer],cache,native,args.repetitions,boundary=m<=8)
        result['weights_layer']=layer;results['cases'].append(result);write_json(output/'results.json',results)
        print(f'M{m} layer{layer} fused={result["hot_fused"]["median_ms"]:.6f}ms checked={result["hot_checked"]["median_ms"]:.6f}ms Ql2={result["reference_errors"]["Q"]["relative_l2"]:.3g}',flush=True)
        gc.collect()
    results['continuity_and_weight_sweep']=continuity_and_weight_sweep(kernels,weights,cache,native)
    started=time.perf_counter();copy=bandwidth_copy();copy_prepare=time.perf_counter()-started
    results['exports'].append({'name':'copy','prepare_compile_s':copy_prepare,'abi':exports(copy,output,'copy')})
    results['bandwidth']=[]
    for m in (512,2048,8192):
        x=torch.randn((m,14336),device='cuda',dtype=torch.float16);y=torch.empty_like(x)
        run=lambda:copy(x,y,stream=torch.cuda.current_stream().cuda_stream)
        timing,_=benchmark(run,repetitions=args.repetitions)
        assert bitexact(x,y)
        traffic=m*14336*4
        results['bandwidth'].append({'M':m,'copy_timing':timing,'read_write_bytes':traffic,
            'measured_GB_s':traffic/timing['median_ms']/1e6,'op20_budget_requires_GB_s':293.60128,
            'op20_minimum_io_floor_ms_at_copy_bandwidth':timing['median_ms'],
            'scope':'independent TileLang streaming X->Y copy; equal minimum bytes to op20 but no norm/rope/random KV writes'})
        del x,y
    results['memory']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    results['implementation_identity']=[identity(ROOT/'kernels/operators/op20_full_prepare.py'),identity(Path(__file__))]
    results['status']='passed standalone synthetic projection data, true weight binding, metadata/graph tests; not model integration'
    write_json(output/'results.json',results)
    print('op20 completed; wrapper cleanup releases GPU lock',flush=True)

if __name__=='__main__':main()
