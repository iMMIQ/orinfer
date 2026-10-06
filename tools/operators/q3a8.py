"""Check native Q3_K projection/embedding against supplied GGML C reference.

--reference-source is a directory containing ggml-common.h and ggml-quants.c.
The reference function/struct are extracted unchanged and compiled only inside
the ignored output directory. No reference library enters online execution.
"""
import argparse
import ctypes
import gc
import re
import subprocess
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from kernels.model.q3a8 import q3a8, q3_embedding
from kernels.operators.op30_activation_quantization import activation_quantization, launch
from tools.operators.q2a8 import reference_quant
from tools.operators.common import configure, benchmark, error, export_kernel, environment, identity, write_json


def reference(directory, output):
    common = (directory/'ggml-common.h').read_text()
    quants = (directory/'ggml-quants.c').read_text()
    struct = re.search(r'typedef struct \{[^{}]*\} block_q3_K;',common).group()
    start = quants.index('void dequantize_row_q3_K(')
    stop = quants.index('\n}\n',start)+3
    function = quants[start:stop]
    text = ('#include <stdint.h>\n#include <string.h>\n#include <assert.h>\n'
            '#define QK_K 256\n#define GGML_RESTRICT restrict\ntypedef uint16_t ggml_half;\n'
            'static float from_half(ggml_half bits) { _Float16 value; memcpy(&value,&bits,2); return (float)value; }\n'
            '#define GGML_FP16_TO_FP32(x) from_half(x)\n'+struct+'\n'+function)
    source = output/'reference.c'
    source.write_text(text)
    lib = output/'reference.so'
    subprocess.run(['cc','-O2','-shared','-fPIC',str(source),'-o',str(lib)],check=True)
    reference_lib = ctypes.CDLL(str(lib.resolve()))
    decode = reference_lib.dequantize_row_q3_K
    decode.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
    decode.restype = None
    def run(packed):
        packed = np.ascontiguousarray(packed)
        values = np.empty((*packed.shape[:-1],packed.shape[-1]//110*256),dtype=np.float32)
        decode(packed.ctypes.data,values.ctypes.data,values.size)
        return values
    return run,[identity(directory/name) for name in ('ggml-common.h','ggml-quants.c')]


def synthesize(n,k):
    rng = np.random.default_rng(20261002)
    blocks = rng.integers(0,256,(n,k//256,110),dtype=np.uint8)
    scales = rng.uniform(-.001,.001,(n,k//256)).astype('<f2')
    scales[0] = 0
    blocks[...,108:] = scales.view(np.uint8).reshape(n,k//256,2)
    # Explicit corner: q=-4, s=-32 produces +128 before the sign flip.
    blocks[1,:,:108] = 0
    return blocks.reshape(n,k//256*110)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-source',type=Path,required=True)
    parser.add_argument('--checkpoint-dir',type=Path)
    args = parser.parse_args()
    configure()
    decode, sources = reference(args.reference_source,args.output)
    report = {'environment':environment(),'sources':sources,'cases':[],'complete':False,
              'scope':'Native mixed Q3_K operators; synthetic activations, no model quality/TPS.',
              'global_w8_bytes':0}
    banks = [('synthetic',synthesize(128,512),(1,3,17,64))]
    if args.checkpoint_dir:
        path = args.checkpoint_dir/'model-00001-of-00002.safetensors'
        with safe_open(path,framework='np') as reader:
            for name, sizes in [('blk.5.attn_qkv.weight',(1,8,512)),('blk.0.ffn_up_shexp.weight',(1,17,512))]:
                assert reader.metadata()[f'ggml.type.{name}']=='11'
                packed = reader.get_tensor(name)
                banks.append((name,packed,sizes))
    for name, packed, sizes in banks:
        n, pk = packed.shape
        k = pk//110*256
        p = torch.from_numpy(packed).cuda()
        values = decode(packed)
        w = torch.from_numpy(values).cuda()
        for m in sizes:
            x = (torch.randn((m,k),device='cuda')*.2).half()
            x[0,:64] = 0
            if m > 1:
                x[-1] = 0
            aq = torch.empty_like(x,dtype=torch.int8)
            sa = torch.empty((m,k//64),device='cuda',dtype=torch.float16)
            out = torch.empty((m,n),device='cuda',dtype=torch.float16)
            mask = torch.zeros(k,device='cuda',dtype=torch.uint8)
            quant = activation_quantization(k,64,threads=128)
            projection = q3a8(m,n,k,64 if m >= 64 else 16)
            def run():
                launch(quant,x,mask,aq,sa,stream=torch.cuda.current_stream().cuda_stream)
                projection(aq,p,sa,out)
            def validate():
                eq,es,xr = reference_quant(x)
                assert torch.equal(aq,eq) and torch.equal(sa,es)
                expected = (xr @ w.T).half()
                metric = error(out,expected)
                assert metric['finite'] and metric['relative_l2'] < .002,metric
                return metric
            run()
            metrics = validate()
            timing, graph = benchmark(run,repetitions=8)
            saved = out.clone()
            out.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out,saved)
            original = x.clone()
            x.copy_(torch.randn_like(x)*.2)
            graph.replay()
            torch.cuda.synchronize()
            changed = validate()
            x.copy_(original)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out,saved)
            p.zero_() # All super-block scales become zero.
            graph.replay()
            torch.cuda.synchronize()
            assert bool((out == 0).all())
            p.copy_(torch.from_numpy(packed).cuda())
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out,saved)
            folder = args.output/f'{name.replace(".","-")}-M{m}'
            export_kernel(projection,folder)
            assembly = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(folder/'kernel.cubin')],text=True)
            assert 'IMMA.16832.S8.S8' in assembly
            report['cases'].append({'bank':name,'rows':m,'timing':timing,'errors':metrics,
                                    'changed_input_errors':changed,'sass_int8_verified':True,
                                    'graph_checks':['poison','changed_input','changed_weight','restore']})
            write_json(args.output/'results.json',report)
            print(name,m,timing['median_ms'],flush=True)
            del graph
            gc.collect()
        del p,w
        gc.collect()
    packed = synthesize(128,2560)
    p = torch.from_numpy(packed).cuda()
    values = torch.from_numpy(decode(packed)).cuda().half()
    ids = torch.tensor([0,1,127,1,-1,128],device='cuda',dtype=torch.int32)
    out = torch.empty((ids.numel(),2560),device='cuda',dtype=torch.float16)
    embedding = q3_embedding(ids.numel(),2560,128)
    def run_embedding():
        embedding(ids,p,out)
    def validate_embedding():
        valid = (ids>=0)&(ids<128)
        expected = values[ids.clamp(0,127).long()]*valid[:,None]
        assert torch.equal(out,expected)
    run_embedding()
    validate_embedding()
    timing,graph = benchmark(run_embedding,repetitions=8)
    ids.copy_(torch.tensor([127,-2,0,1,128,42],device='cuda',dtype=torch.int32))
    out.fill_(float('nan'))
    graph.replay()
    torch.cuda.synchronize()
    validate_embedding()
    export_kernel(embedding,args.output/'embedding')
    report['cases'].append({'bank':'synthetic_embedding','timing':timing,'exact_reference':True,
                            'graph_checks':['changed_ids','poison']})
    report['complete'] = True
    write_json(args.output/'results.json',report)


if __name__ == '__main__':
    main()
