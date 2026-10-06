"""Q2I8 native-INT8 projection and real-weight expert FFN checks.

Optional BF16 fixtures contain ten layer0 experts. Activations are synthetic;
this diagnoses the representation/kernel and is not full-model quality/TPS.
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from tools.quantization.q2i8 import Weights, quantize, save, load
from kernels.model.q2i8 import q2i8
from kernels.model.q2a8 import q2a8
from kernels.operators.op30_activation_quantization import activation_quantization, swiglu_activation_quantization, launch
from tools.operators.common import configure, benchmark, error, environment, export_kernel, identity, write_json


def upload(bank):
    storage = [w.gpu_layout() for w in bank]
    return tuple(torch.from_numpy(np.stack([s[i] for s in storage])).cuda() for i in range(3))


def reference(a, integer, ws, sa):
    # FP64 dot is an independent exact integer oracle for these bounded shapes.
    dot = torch.bmm(a.double(),integer.double().transpose(1,2)).float()
    return ((dot*ws.float()[:,None,:])*sa.float()[...,None]).half()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixture-dir',type=Path)
    a = p.parse_args()
    configure()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    report = {'environment':environment(),'cases':[],'ffn':[],'sources':[],'complete':False,
              'scope':'Native INT8 kernels and pre-dispatched FFN; synthetic activations, real BF16 layer0 weights if supplied. No full-model quality/TPS.',
              'global_w8_bytes':0,'inner_k_float_scales':0}
    def record():
        write_json(a.output/'results.json',report)
    rng = np.random.default_rng(20261002)
    for group in (64,128):
        e,n,k = 2,65,256
        banks = []
        for expert in range(e):
            indices = rng.integers(0,256,(n,k//4),dtype=np.uint8)
            books = rng.integers(-128,128,(n,k//group,4),dtype=np.int16).astype(np.int8)
            books[0] = np.array([-128,-3,0,127],np.int8)
            scales = np.full(n,.015625,np.float16)
            banks.append(Weights(indices,books,scales,group))
        pp,book,ws = upload(banks)
        integer = torch.from_numpy(np.stack([w.integer_weights() for w in banks])).cuda()
        for m in (1,3,17):
            label = f'synthetic-g{group}-m{m}-n{n}-k{k}'
            x = torch.randint(-128,128,(e,m,k),device='cuda',dtype=torch.int8)
            sa = torch.full((e,m),.0078125,device='cuda',dtype=torch.float16)
            output = torch.full((e*m+1,n),91.,device='cuda',dtype=torch.float16)
            kernel = q2i8(e,m,n,k,group_size=group)
            def run():kernel(x.view(e*m,k),pp,book,ws,sa.view(-1),output[:e*m])
            timing,graph = benchmark(run,repetitions=5)
            expected = reference(x,integer,ws,sa).view(e*m,n)
            metric = error(output[:e*m],expected)
            assert torch.equal(output[:e*m],expected),metric
            assert bool((output[-1] == 91).all())
            original_x = x.clone()
            x.zero_();output[:e*m].fill_(91);graph.replay();torch.cuda.synchronize()
            assert bool((output[:e*m] == 0).all())
            x.copy_(original_x);original_book = book.clone()
            book.zero_();output[:e*m].fill_(91);graph.replay();torch.cuda.synchronize()
            assert bool((output[:e*m] == 0).all())
            book.copy_(original_book);graph.replay();torch.cuda.synchronize()
            assert torch.equal(output[:e*m],expected)
            report['cases'].append({'name':label,'error':metric,'exact_integer_oracle':True,
                'changed_activation_and_palette_replay':True,'tail_guard':True,'timing':timing,
                'export':export_kernel(kernel,a.output/label)})
            record()
        del pp,book,ws,integer,kernel,graph
        gc.collect();torch.cuda.empty_cache()
    if a.fixture_dir:
        matrices = []
        for family in ('gate_up','down'):
            path = a.fixture_dir/f'{family}-bf16.npy'
            report['sources'].append(identity(path))
            raw = np.load(path,mmap_mode='r',allow_pickle=False)
            fitted = []
            for expert in range(len(raw)):
                w = quantize(raw[expert])
                path = a.output/f'{family}-expert{expert}.safetensors'
                save(path,w,provenance={'source':report['sources'][-1],'expert_fixture_index':expert,'calibration':'weight-only'})
                fitted.append(load(path))
            matrices.append(fitted)
        gate_banks,down_banks = matrices
        e = len(gate_banks)
        n,k = gate_banks[0].validate();f = n//2
        assert all(w.validate() == (n,k) for w in gate_banks)
        assert all(w.validate() == (k,f) for w in down_banks)
        gp,gb,gs = upload(gate_banks);dp,db,ds = upload(down_banks)
        gi = torch.from_numpy(np.stack([w.integer_weights() for w in gate_banks])).cuda()
        di = torch.from_numpy(np.stack([w.integer_weights() for w in down_banks])).cuda()
        old_g = torch.from_numpy(np.load(a.fixture_dir/'q2-gate_up-packed.npy').reshape(e,n,k//64*18)).cuda()
        old_d = torch.from_numpy(np.load(a.fixture_dir/'q2-down-packed.npy').reshape(e,k,f//64*18)).cuda()
        for m in (1,16,64):
            label = f'real-ffn-e{e}-m{m}'
            x = (torch.randn((e*m,k),device='cuda')*.2).half()
            aq = torch.empty_like(x,dtype=torch.int8);sa = torch.empty(e*m,device='cuda',dtype=torch.float16)
            gu = torch.empty((e*m,n),device='cuda',dtype=torch.float16)
            fq = torch.empty((e*m,f),device='cuda',dtype=torch.int8);fs = torch.empty(e*m,device='cuda',dtype=torch.float16)
            out = torch.empty((e*m,k),device='cuda',dtype=torch.float16)
            mask = torch.zeros(k,device='cuda',dtype=torch.uint8);fm = torch.zeros(f,device='cuda',dtype=torch.uint8)
            quant = activation_quantization(k)
            swiglu = swiglu_activation_quantization(f)
            gate = q2i8(e,m,n,k)
            down = q2i8(e,m,k,f)
            def run():
                stream = torch.cuda.current_stream().cuda_stream
                launch(quant,x,mask,aq,sa,stream=stream)
                gate(aq,gp,gb,gs,sa,gu)
                launch(swiglu,gu,fm,fq,fs,stream=stream)
                down(fq,dp,db,ds,fs,out)
            timing,graph = benchmark(run,repetitions=5)
            expected_gu = reference(aq.view(e,m,k),gi,gs,sa.view(e,m)).view(e*m,n)
            expected_out = reference(fq.view(e,m,f),di,ds,fs.view(e,m)).view(e*m,k)
            metric = error(out,expected_out)
            assert torch.equal(gu,expected_gu) and torch.equal(out,expected_out),metric
            # Float materialization -> row A8 reference, independent of fused path.
            g = expected_gu[:,:f].float()
            exp = (-g.abs()).exp()
            sigmoid = torch.where(g >= 0,1/(1+exp),exp/(1+exp))
            value = ((g*sigmoid)*expected_gu[:,f:].float()).half().float()
            maximum = value.abs().amax(-1)
            ref_fs = torch.where(maximum > 0,(maximum/127).clamp_min(2**-24),1.).half()
            ref_fq = (value/ref_fs.float()[:,None]).round().clamp(-127,127).to(torch.int8)
            assert torch.equal(fs,ref_fs) and torch.equal(fq,ref_fq)
            saved = x.clone();x.zero_();out.fill_(91);graph.replay();torch.cuda.synchronize()
            assert bool((out == 0).all())
            x.copy_(saved);graph.replay();torch.cuda.synchronize()
            assert torch.equal(out,expected_out)
            # Existing direct Q2_0 chain includes its group64 A8 and float rescale.
            old_sa = torch.empty((e*m,k//64),device='cuda',dtype=torch.float16)
            old_fs = torch.empty((e*m,f//64),device='cuda',dtype=torch.float16)
            old_quant = activation_quantization(k,64,threads=128)
            old_swiglu = swiglu_activation_quantization(f,64,threads=128)
            old_gate = q2a8(e,m,n,k,implementation='shared')
            old_down = q2a8(e,m,k,f,implementation='shared')
            def baseline():
                stream = torch.cuda.current_stream().cuda_stream
                launch(old_quant,x,mask,aq,old_sa,stream=stream)
                old_gate(aq.view(e,m,k),old_g,old_sa.view(e,m,k//64),gu.view(e,m,n))
                launch(old_swiglu,gu,fm,fq,old_fs,stream=stream)
                old_down(fq.view(e,m,f),old_d,old_fs.view(e,m,f//64),out.view(e,m,k))
            old_timing,old_graph = benchmark(baseline,repetitions=5)
            report['ffn'].append({'name':label,'error':metric,'exact_integer_oracle':True,
                'swiglu_row_a8_oracle':True,'changed_input_replay':True,'timing':timing,
                'q2_0_group64_baseline_timing':old_timing,'speedup':old_timing['median_ms']/timing['median_ms'],
                'note':'Different quantized representations and A8 group sizes. Performance comparison only; no quality acceptance.',
                'weights_bytes':sum(w.nbytes for w in gate_banks+down_banks),
                'gate_export':export_kernel(gate,a.output/(label+'-gate')),
                'down_export':export_kernel(down,a.output/(label+'-down'))})
            record()
            del graph,old_graph,gate,down,old_gate,old_down
            gc.collect();torch.cuda.empty_cache()
    report['complete'] = True
    record()


if __name__ == '__main__':
    main()
