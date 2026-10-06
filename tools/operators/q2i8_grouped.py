"""GPU routed Q2I8 FFN, including top-k/dispatch/quantize/combine.

The 512-slot probe repeats ten BF16-derived expert fixtures. It checks real
matrix geometry and bank traffic, not full-checkpoint/model quality or TPS.
"""
import argparse
import gc

import numpy as np
import torch

from pathlib import Path
from tools.quantization.q2i8 import load, quantize
from tools.operators.q2i8 import upload
from tools.operators.common import configure, benchmark, error, export_kernel, identity, environment, write_json
from kernels.model.moe import router_topk, expert_histogram, expert_offsets, expert_tiles, expert_dispatch, moe_combine
from kernels.model.q2i8 import q2i8_grouped
from kernels.model.q2a8 import q2a8_grouped
from kernels.model.integer_vq import integer_vq_grouped, rotate_activation
from tools.quantization.vq import load as load_vq
from kernels.operators.op30_activation_quantization import activation_quantization, swiglu_activation_quantization, launch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weights-dir',type=Path)
    p.add_argument('--q2-fixture-dir',type=Path)
    p.add_argument('--vq-dir',type=Path,help='Experimental integer VQ screen output')
    p.add_argument('--vq-variant',choices=('vq4-plain','vq4-rot128','e8p-plain','e8p-rot128','e8p-rot128-patch128'))
    p.add_argument('--profile',action='store_true',help='Isolated hot graph component timings; not additive/full-chain TPS')
    a = p.parse_args()
    configure()
    report = {'environment':environment(),'cases':[],'sources':[],'complete':False,
              'scope':__doc__,'global_w8_bytes':0,'inner_k_float_scales':0,
              'global_w8_note':'Execution uses shared tile expansion only. Validation-only donor W8 oracles are outside the timed graph.'}
    rng = np.random.default_rng(20261002)
    g = [quantize(rng.normal(0,.02,(128,128)).astype(np.float32),group_size=64) for _ in range(37)]
    d = [quantize(rng.normal(0,.02,(128,64)).astype(np.float32),group_size=64) for _ in range(37)]
    if bool(a.vq_dir) != bool(a.vq_variant):raise ValueError('VQ directory and variant required together')
    if a.vq_dir:
        report['scope'] = 'GPU routed integer VQ FFN including original-input routing, dispatch, A8, optional rotations, projections and combine. Ten actual BF16-derived experts repeated into 512 slots, synthetic activations/routing; no full-checkpoint quality or model TPS.'
        report['vq_variant'] = a.vq_variant
    banks = [] if a.vq_dir else [('synthetic37',g,d,37,10,(1,3,17,512),None)]
    if a.weights_dir:
        g,d = [],[]
        for family,bank in [('gate_up',g),('down',d)]:
            for e in range(10):
                path = a.weights_dir/f'{family}-expert{e}.safetensors'
                bank.append(load(path));report['sources'].append(identity(path))
        old = None
        if a.q2_fixture_dir:
            paths = [a.q2_fixture_dir/f'q2-{family}-packed.npy' for family in ('gate_up','down')]
            report['sources'].extend(identity(path) for path in paths)
            old = tuple(np.load(path,allow_pickle=False) for path in paths)
        banks.append(('ten_real_experts_repeated_into_512_slots',g,d,512,10,(1,8,512),old))
    if a.vq_dir:
        g,d = [],[]
        for family,bank in [('gate_up',g),('down',d)]:
            for e in (0,7,63,127,191,255,319,383,447,511):
                path = a.vq_dir/f'{a.vq_variant}-{family}-expert{e}.safetensors'
                bank.append(load_vq(path));report['sources'].append(identity(path))
        old = None
        if a.q2_fixture_dir:
            paths = [a.q2_fixture_dir/f'q2-{family}-packed.npy' for family in ('gate_up','down')]
            report['sources'].extend(identity(path) for path in paths)
            old = tuple(np.load(path,allow_pickle=False) for path in paths)
        banks.append((a.vq_variant+'-ten_real_experts_repeated_into_512_slots',g,d,512,10,(1,8,512),old))
    for label,gate_banks,down_banks,b,top_k,sizes,old in banks:
        donor = len(gate_banks)
        n,h = gate_banks[0].validate();f = n//2
        vector = hasattr(gate_banks[0],'kind')
        group = 128 if vector else gate_banks[0].group_size
        assert all(w.validate() == (n,h) for w in gate_banks)
        assert all(w.validate() == (h,f) for w in down_banks)
        if vector:
            assert all(w.kind == gate_banks[0].kind for w in gate_banks+down_banks)
            assert all((w.patches is not None) == (gate_banks[0].patches is not None) for w in gate_banks+down_banks)
        else:
            assert all(w.group_size == group for w in gate_banks+down_banks)
        rotated = vector and len(gate_banks[0].signs) > 0
        if rotated:
            assert all(np.array_equal(w.signs,gate_banks[0].signs) for w in gate_banks)
            assert all(np.array_equal(w.signs,down_banks[0].signs) for w in down_banks)
            gate_signs = torch.from_numpy(gate_banks[0].signs).cuda()
            down_signs = torch.from_numpy(down_banks[0].signs).cuda()
        gp,gb,gs = upload([gate_banks[e%donor] for e in range(b)])
        dp,db,ds = upload([down_banks[e%donor] for e in range(b)])
        patched = vector and gate_banks[0].patches is not None
        if vector:
            gp_patch = torch.from_numpy(np.stack([gate_banks[e%donor].patches.T for e in range(b)]).copy()).cuda() if patched else torch.zeros(1,device='cuda',dtype=torch.uint16)
            dp_patch = torch.from_numpy(np.stack([down_banks[e%donor].patches.T for e in range(b)]).copy()).cuda() if patched else torch.zeros(1,device='cuda',dtype=torch.uint16)
        execution_weights = [gp,gb,gs,dp,db,ds]
        if rotated:execution_weights.extend((gate_signs,down_signs))
        if patched:execution_weights.extend((gp_patch,dp_patch))
        execution_weight_bytes = sum(t.numel()*t.element_size() for t in execution_weights)
        del execution_weights
        gi = torch.from_numpy(np.stack([w.integer_weights() for w in gate_banks])).cuda()
        di = torch.from_numpy(np.stack([w.integer_weights() for w in down_banks])).cuda()
        donor_gs = torch.from_numpy(np.stack([w.scales for w in gate_banks])).cuda()
        donor_ds = torch.from_numpy(np.stack([w.scales for w in down_banks])).cuda()
        if old is not None:
            def legacy(x,k):
                x = x.reshape(donor,-1,k//64,18)[np.arange(b)%donor]
                return torch.from_numpy(np.ascontiguousarray(x.transpose(0,2,1,3))).cuda()
            old_g,old_d = legacy(old[0],h),legacy(old[1],f)
        for m in sizes:
            assignments = m*top_k
            capacity = (assignments+15)//16+min(b,assignments)
            x = (torch.randn((m,h),device='cuda')*.2).half()
            logits = torch.randn((m,b),device='cuda')
            ids = torch.empty((m,top_k),device='cuda',dtype=torch.int32)
            prob = torch.empty((m,top_k),device='cuda')
            counts = torch.empty(b,device='cuda',dtype=torch.int32);relative = torch.empty_like(ids)
            offsets = torch.empty_like(counts);tile_offsets = torch.empty_like(counts)
            tile_count = torch.empty(1,device='cuda',dtype=torch.int32)
            tile_expert = torch.empty(capacity,device='cuda',dtype=torch.int32);tile_row = torch.empty_like(tile_expert)
            aq = torch.empty_like(x,dtype=torch.int8);sa = torch.empty((m,1),device='cuda',dtype=torch.float16)
            dispatch = torch.empty((assignments,h),device='cuda',dtype=torch.int8)
            das = torch.empty((assignments,1),device='cuda',dtype=torch.float16);slot_map = torch.empty_like(ids)
            gu = torch.empty((assignments,n),device='cuda',dtype=torch.float16)
            fq = torch.empty((assignments,f),device='cuda',dtype=torch.int8)
            fs = torch.empty(assignments,device='cuda',dtype=torch.float16)
            y = torch.empty((assignments,h),device='cuda',dtype=torch.float16)
            shared = torch.zeros_like(x);shared_gate = torch.zeros(m,device='cuda',dtype=torch.float16);out = torch.empty_like(x)
            mask = torch.zeros(h,device='cuda',dtype=torch.uint8);fm = torch.zeros(f,device='cuda',dtype=torch.uint8)
            route = router_topk(m,b,top_k);hist = expert_histogram(m,b,top_k)
            prefix = expert_offsets(b);tiles = expert_tiles(m,b,top_k)
            scatter = expert_dispatch(m,h,b,top_k,scale_group=h)
            quant = activation_quantization(h);swiglu = swiglu_activation_quantization(f)
            if vector:
                gate = integer_vq_grouped(assignments,b,capacity,n,h,kind=gate_banks[0].kind,patched=patched)
                down = integer_vq_grouped(assignments,b,capacity,h,f,kind=down_banks[0].kind,patched=patched)
            else:
                gate = q2i8_grouped(assignments,b,capacity,n,h,group_size=group)
                down = q2i8_grouped(assignments,b,capacity,h,f,group_size=group)
            if rotated:
                rotate_gate = rotate_activation(m,h)
                rotate_down = rotate_activation(assignments,f,swiglu=True)
                rx = torch.empty_like(x)
                rf = torch.empty((assignments,f),device='cuda',dtype=torch.float16)
                down_quant = activation_quantization(f)
            combine = moe_combine(m,h,assignments,top_k)
            def run():
                stream = torch.cuda.current_stream().cuda_stream
                route(logits,ids,prob);hist(ids,counts,relative)
                prefix(counts,offsets,tile_offsets,tile_count);tiles(counts,tile_offsets,tile_expert,tile_row)
                if rotated:rotate_gate(x,gate_signs,rx)
                launch(quant,rx if rotated else x,mask,aq,sa.view(-1),stream=stream)
                scatter(aq,sa,ids,relative,offsets,dispatch,das,slot_map)
                if vector:gate(dispatch,gp,gb,gp_patch,gs,das.view(-1),counts,offsets,tile_expert,tile_row,tile_count,gu)
                else:gate(dispatch,gp,gb,gs,das.view(-1),counts,offsets,tile_expert,tile_row,tile_count,gu)
                if rotated:
                    rotate_down(gu,down_signs,rf)
                    launch(down_quant,rf,fm,fq,fs,stream=stream)
                else:
                    launch(swiglu,gu,fm,fq,fs,stream=stream)
                if vector:down(fq,dp,db,dp_patch,ds,fs,counts,offsets,tile_expert,tile_row,tile_count,y)
                else:down(fq,dp,db,ds,fs,counts,offsets,tile_expert,tile_row,tile_count,y)
                combine(y,slot_map,prob,shared,shared_gate,out)
            def validate():
                expected_ids = torch.argsort(logits,descending=True,stable=True)[:,:top_k]
                assert torch.equal(ids.long(),expected_ids)
                expected_counts = torch.bincount(ids.flatten().long(),minlength=b).int()
                assert torch.equal(counts,expected_counts)
                assert torch.equal(offsets,expected_counts.cumsum(0).int()-expected_counts)
                assert int(tile_count) == int(((expected_counts+15)//16).sum()) <= capacity
                assert torch.equal(torch.sort(slot_map.flatten()).values,torch.arange(assignments,device='cuda',dtype=torch.int32))
                assert torch.equal(dispatch[slot_map.long()],aq[:,None,:].expand(m,top_k,h))
                assert torch.equal(das[slot_map.long()],sa[:,None,:].expand(m,top_k,1))
                compact_expert = torch.repeat_interleave(torch.arange(b,device='cuda'),counts.long())%donor
                ref_gu = torch.empty_like(gu);ref_y = torch.empty_like(y)
                for e in range(donor):
                    rows = torch.where(compact_expert == e)[0]
                    if not len(rows):continue
                    dot = (dispatch[rows].double()@gi[e].double().T).float()
                    ref_gu[rows] = ((dot*donor_gs[e].float()[None,:])*das[rows].float()).half()
                    dot = (fq[rows].double()@di[e].double().T).float()
                    ref_y[rows] = ((dot*donor_ds[e].float()[None,:])*fs[rows].float()[:,None]).half()
                assert torch.equal(gu,ref_gu) and torch.equal(y,ref_y)
                ref_out = (ref_y[slot_map.long()].float()*prob[...,None]).sum(1).half()
                metric = error(out,ref_out)
                assert metric['finite'] and metric['relative_l2'] < .002,metric
                return metric
            timing,graph = benchmark(run,repetitions=4)
            metric = validate();saved = out.clone()
            out.fill_(float('nan'));gu.fill_(float('nan'));y.fill_(float('nan'))
            graph.replay();torch.cuda.synchronize();assert torch.equal(out,saved)
            saved_logits = logits.clone()
            logits.fill_(-20);logits[:,:top_k] = torch.arange(top_k,device='cuda').float()
            graph.replay();torch.cuda.synchronize();changed_metric = validate()
            assert not torch.equal(out,saved)
            skewed,_ = benchmark(run,repetitions=4)
            logits.copy_(saved_logits);graph.replay();torch.cuda.synchronize();assert torch.equal(out,saved)
            original_x = x.clone();x.zero_();graph.replay();torch.cuda.synchronize();assert bool((out == 0).all())
            x.copy_(original_x);graph.replay();torch.cuda.synchronize();assert torch.equal(out,saved)
            case = {'bank':label,'rows':m,'experts':b,'top_k':top_k,'timing':timing,'skewed_timing':skewed,
                    'integer_oracle_equal':True,'mixture_error':metric,'changed_routing_error':changed_metric,
                    'graph_checks':['poison','changed_routing','zero_activation','restore'],
                    'weight_bytes':execution_weight_bytes,
                    'fixture_payload_bytes':sum(gate_banks[e%donor].nbytes+down_banks[e%donor].nbytes for e in range(b)),
                    'validation_only_w8_bytes':gi.numel()*gi.element_size()+di.numel()*di.element_size()}
            if vector:case['rotation_included_in_timing'] = rotated
            if a.profile:
                operations = {}
                if vector:
                    operations['gate_up'] = lambda:gate(dispatch,gp,gb,gp_patch,gs,das.view(-1),counts,offsets,tile_expert,tile_row,tile_count,gu)
                    operations['down'] = lambda:down(fq,dp,db,dp_patch,ds,fs,counts,offsets,tile_expert,tile_row,tile_count,y)
                else:
                    operations['gate_up'] = lambda:gate(dispatch,gp,gb,gs,das.view(-1),counts,offsets,tile_expert,tile_row,tile_count,gu)
                    operations['down'] = lambda:down(fq,dp,db,ds,fs,counts,offsets,tile_expert,tile_row,tile_count,y)
                if rotated:
                    operations['input_rotation'] = lambda:rotate_gate(x,gate_signs,rx)
                    operations['swiglu_rotation'] = lambda:rotate_down(gu,down_signs,rf)
                case['hot_components'] = {name:benchmark(operation,repetitions=4)[0] for name,operation in operations.items()}
                case['hot_components_note'] = 'Isolated graph replay; different locality than full chain; do not sum or infer full-model TPS'
            if old is not None:
                old_sa = torch.empty((m,h//64),device='cuda',dtype=torch.float16)
                old_das = torch.empty((assignments,h//64),device='cuda',dtype=torch.float16)
                old_fs = torch.empty((assignments,f//64),device='cuda',dtype=torch.float16)
                old_quant = activation_quantization(h,64,threads=128)
                old_swiglu = swiglu_activation_quantization(f,64,threads=128)
                old_scatter = expert_dispatch(m,h,b,top_k)
                old_gate = q2a8_grouped(assignments,b,capacity,n,h,layout='group_major')
                old_down = q2a8_grouped(assignments,b,capacity,h,f,layout='group_major')
                def baseline():
                    stream = torch.cuda.current_stream().cuda_stream
                    route(logits,ids,prob);hist(ids,counts,relative)
                    prefix(counts,offsets,tile_offsets,tile_count);tiles(counts,tile_offsets,tile_expert,tile_row)
                    launch(old_quant,x,mask,aq,old_sa,stream=stream)
                    old_scatter(aq,old_sa,ids,relative,offsets,dispatch,old_das,slot_map)
                    old_gate(dispatch,old_g,old_g,old_das,counts,offsets,tile_expert,tile_row,tile_count,gu)
                    launch(old_swiglu,gu,fm,fq,old_fs,stream=stream)
                    old_down(fq,old_d,old_d,old_fs,counts,offsets,tile_expert,tile_row,tile_count,y)
                    combine(y,slot_map,prob,shared,shared_gate,out)
                old_timing,old_graph = benchmark(baseline,repetitions=4)
                assert torch.equal(old_das[slot_map.long()],old_sa[:,None,:].expand(m,top_k,h//64))
                case['q2_0_group64_baseline_timing'] = old_timing
                case['speedup'] = old_timing['median_ms']/timing['median_ms']
                case['baseline_note'] = 'Same repeated donor banks/routing; different weight representation and activation group sizes; no quality acceptance'
                del old_graph,old_gate,old_down
            folder = a.output/f'{label}-M{m}'
            for name,kernel in [('dispatch',scatter),('gate_up',gate),('down',down)]:
                export_kernel(kernel,folder/name)
            report['cases'].append(case);write_json(a.output/'results.json',report)
            print(label,m,timing['median_ms'],case.get('speedup'),flush=True)
            del graph,gate,down
            gc.collect();torch.cuda.empty_cache()
        del gp,gb,gs,dp,db,ds,gi,di
        if old is not None:del old_g,old_d
        gc.collect();torch.cuda.empty_cache()
    report['complete'] = True;write_json(a.output/'results.json',report)


if __name__ == '__main__':
    main()
