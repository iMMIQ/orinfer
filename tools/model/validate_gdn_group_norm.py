"""Validate fused GDN head-wise A8 against RMSNorm -> FP16 -> group A8."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 30, 62])
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from kernels.model.gdn_gated_norm_a8 import gdn_gated_norm_a8
    from kernels.operators.op17_gdn_gated_norm import gdn_gated_norm
    from kernels.operators.op30_activation_quantization import activation_quantization
    from tools.operators.common import benchmark, configure, export_kernel, write_json

    configure()
    torch.empty(1, device='cuda')
    data = json.loads((args.model / 'cache/model.json').read_text())
    buffers = {b['name']: b for b in data['metadata']['buffers']}
    weights = args.model / 'cache/weights'
    index = json.loads((weights / 'model.safetensors.index.json').read_text())['weight_map']
    baseline = gdn_gated_norm()
    quantize = activation_quantization(6144, 128, threads=128)
    fused = gdn_gated_norm_a8(128, group_activation=True)
    export_kernel(fused, args.output / 'grouped-norm')
    report = dict(status='running', seed=20261002, cases=[],
                  scope='Real norm weights and synthetic recurrent outputs; exact FP16/A8 boundary and Graph tests, not model quality.')
    for layer in args.layers:
        name = f'L{layer}_GatedWeight'
        with safe_open(str(weights / index[name]), framework='pt', device='cpu') as source:
            weight = source.get_tensor(name)
        if hashlib.sha256(weight.numpy().tobytes()).hexdigest() != buffers[name]['data']['sha256']:
            raise ValueError('Norm weight digest mismatch')
        weight = weight.cuda()
        for rows in [1, 2, 3, 8, 17, 32, 65, 128]:
            for mode in (['random', 'zero', 'tiny', 'large'] if rows == 3 else ['random']):
                x = torch.randn((rows, 48, 128), device='cuda', dtype=torch.float16)
                z = torch.randn_like(x)
                if mode == 'zero':
                    x.zero_()
                    z.zero_()
                elif mode == 'tiny':
                    x.mul_(2**-20)
                    z.mul_(2**-20)
                elif mode == 'large':
                    x.mul_(1000)
                    z.mul_(10)
                normalized = torch.empty_like(x)
                expected_q = torch.empty((rows, 6144), device='cuda', dtype=torch.int8)
                expected_s = torch.empty((rows, 48), device='cuda', dtype=torch.float16)
                mask = torch.zeros(6144, device='cuda', dtype=torch.uint8)
                q_guard = torch.full((rows * 6144 + 256,), 42, device='cuda', dtype=torch.int8)
                s_guard = torch.full((rows * 48 + 64,), 4., device='cuda', dtype=torch.float16)
                q = q_guard[:-256].view(rows, 6144)
                scale = s_guard[:-64].view(rows, 48)

                def ordinary():
                    baseline.adapter.func(x, z, weight, normalized, stream=torch.cuda.current_stream().cuda_stream)
                    quantize.adapter.func(normalized, mask, expected_q, expected_s,
                                          stream=torch.cuda.current_stream().cuda_stream)

                def run():
                    fused.adapter.func(x, z, weight, q, scale, stream=torch.cuda.current_stream().cuda_stream)

                ordinary()
                timing, graph = benchmark(run, repetitions=8)
                assert torch.equal(q, expected_q) and torch.equal(scale, expected_s), (layer, rows, mode)
                assert bool((q_guard[-256:] == 42).all() and (s_guard[-64:] == 4.).all())
                saved_x, saved_z = x.clone(), z.clone()
                x.zero_()
                z.zero_()
                q.fill_(42)
                scale.zero_()
                graph.replay()
                torch.cuda.synchronize()
                assert bool((q == 0).all() and (scale == 1).all()), 'Graph ignored changed input'
                x.copy_(saved_x)
                z.copy_(saved_z)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(q, expected_q) and torch.equal(scale, expected_s)
                assert bool((q_guard[-256:] == 42).all() and (s_guard[-64:] == 4.).all())
                report['cases'].append(dict(layer=layer, rows=rows, mode=mode, timing=timing,
                                           exact_boundary=True, graph_zero_restore=True, guard=True))
                write_json(args.output / 'result.json', report)
                print('pass', layer, rows, mode, flush=True)
                del graph, x, z, normalized, expected_q, expected_s, mask, q_guard, s_guard, q, scale, saved_x, saved_z
    report['status'] = 'passed'
    write_json(args.output / 'result.json', report)


if __name__ == '__main__':
    main()
