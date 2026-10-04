"""Real W4 weights: dynamic-row byte permutation, tails and graph input replay."""
import argparse
import json
import mmap
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import numpy as np
    import torch
    from kernels.model.w4_small_m import w4_small_m
    from tools.operators.common import configure, benchmark, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure()
    weights = args.model / 'cache/weights'
    index = json.loads((weights/'model.safetensors.index.json').read_text())['weight_map']

    def tensor(name):
        with (weights/index[name]).open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as raw:
            size = int.from_bytes(raw[:8], 'little')
            info = json.loads(raw[8:8+size])[name]
            dtype = {'I32':np.int32, 'F16':np.float16, 'I8':np.int8}[info['dtype']]
            start, end = info['data_offsets']
            data = np.frombuffer(raw[8+size+start:8+size+end],dtype=dtype).copy().reshape(info['shape'])
        return torch.from_numpy(data).cuda()

    torch.manual_seed(20261002)
    report = dict(status='running',seed=20261002,cases=[],
        scope='Real checkpoint projection weights; optimized dynamic-row layout versus original W4 arithmetic, not a BF16/FP8 model quality assessment.')
    for family in ['GateUp', 'Down']:
        pp, scales, zeros = [tensor('L0_'+family+suffix) for suffix in ['_P','_S','_Z']]
        n, groups = scales.shape
        k, split = groups*128, 8 if family == 'Down' else 1
        dtype = 'float32' if split > 1 else 'float16'
        tile_n = 64 if split > 1 else 128
        base = w4_small_m(None,n,k,split,dtype,TILE_N=tile_n,weight_layout='i8')
        fast = w4_small_m(None,n,k,split,dtype,TILE_N=tile_n,weight_layout='i8',byte_permute=True,vector_words=4)
        for rows in [1,2,3,4,8,17,32,65,128]:
            a = torch.randn((rows,k),device='cuda',dtype=torch.float16)*0.1
            golden = torch.empty((split,rows,n),device='cuda',dtype=getattr(torch,dtype))
            guarded = torch.full((split*rows*n+256,),123.,device='cuda',dtype=getattr(torch,dtype))
            result = guarded[:-256].view(split,rows,n)
            def original(): base.adapter.func(a,pp,scales,zeros,golden,stream=torch.cuda.current_stream().cuda_stream)
            def optimized(): fast.adapter.func(a,pp,scales,zeros,result,stream=torch.cuda.current_stream().cuda_stream)
            before, _ = benchmark(original,repetitions=8)
            after, graph = benchmark(optimized,repetitions=8)
            assert torch.equal(golden,result), (family,rows,'output differs')
            assert bool((guarded[-256:]==123.).all()), 'Tail was overwritten'
            initial, saved = result.clone(), a.clone()
            a.zero_(); result.fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
            assert bool((result==0).all()), 'Graph did not consume changed input'
            a.copy_(saved); graph.replay(); torch.cuda.synchronize()
            assert torch.equal(initial,result), 'Graph restoration differs'
            assert bool((guarded[-256:]==123.).all()), 'Graph overwrote tail'
            report['cases'].append(dict(family=family,rows=rows,exact_equal=True,tail_safe=True,
                graph_zero_restore=True,baseline=before,optimized=after))
            write_json(args.output/'result.json',report)
            print(family,rows,'passed',flush=True)
    report['status']='passed'
    write_json(args.output/'result.json',report)


if __name__ == '__main__':
    main()
