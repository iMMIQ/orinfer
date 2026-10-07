"""Screen unchanged Flash Next W8 projections on SM87, including graph replay.

This measures individual kernels, not inference throughput. A captured FFN
operand file is optional; absent inputs are seeded implementation probes.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from kernels.model.int8_projection import int8_gemv, int8_projection
from tools.model.flash_checkpoint import Checkpoint
from tools.operators.common import benchmark, configure, environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/models/flash-next-e8p-a8'))
    parser.add_argument('--activations', type=Path)
    args = parser.parse_args()
    configure()
    source = Checkpoint(args.checkpoint, verify_hashes=False)
    fixtures = torch.load(args.activations, map_location='cpu', weights_only=False) if args.activations else {}
    report = dict(complete=False, environment=environment(), scope=__doc__, cases=[])
    names = [f'model.language_model.layers.0.mlp.shared_expert.{p}_proj.weight'
             for p in ('gate', 'up', 'down')] + ['lm_head.weight']
    for name in names:
        parts = list(source.int8_parts(name))
        weight = torch.from_numpy(np.concatenate([p[1] for p in parts])).cuda()
        scale = torch.from_numpy(np.concatenate([p[2] for p in parts])).cuda()
        n, k = weight.shape
        for m in (1, 4, 8):
            fixture = fixtures.get((name, 1 if m == 1 else 8))
            if fixture:
                activation = fixture['a8'][:m].cuda()
                token_scale = fixture['as'][:m].reshape(-1).cuda()
            else:
                activation = torch.randint(-127, 128, (m, k), device='cuda', dtype=torch.int8)
                token_scale = torch.full((m,), .01, device='cuda', dtype=torch.float16)
            dtype = 'float32' if name == 'lm_head.weight' else 'float16'
            output = torch.empty((m, n), device='cuda', dtype=getattr(torch, dtype))
            reference = torch.empty_like(output)
            baseline = int8_projection(m, n, k, dtype)
            operands = (activation, weight, scale, token_scale, output)
            baseline(activation, weight, scale, token_scale, reference)
            variants = [('mma-64-64', baseline, operands)]
            for bn, bk in ((64, 128), (128, 64), (128, 128)):
                variants.append((f'mma-{bn}-{bk}', int8_projection(m, n, k, dtype, 16, bn, bk), operands))
            if m == 1:
                for bn in (4, 8, 16, 32):
                    variants.append((f'dp4a-{bn}', int8_gemv(n, k, dtype, bn),
                                     (activation.view(torch.int32), weight.view(torch.int32), scale, token_scale, output)))
            row = dict(name=name, rows=m, captured_activation=fixture is not None, variants=[])
            for label, kernel, inputs in variants:
                baseline(activation, weight, scale, token_scale, reference)
                kernel(*inputs)
                torch.cuda.synchronize()
                assert torch.equal(output, reference), (name, m, label)
                timing, graph = benchmark(lambda: kernel(*inputs), repetitions=50)
                # Replay must observe changed input, rather than captured constants.
                original = activation.clone()
                activation.neg_()
                baseline(activation, weight, scale, token_scale, reference)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(output, reference), (name, m, label, 'changed-input')
                activation.copy_(original)
                del graph
                row['variants'].append(dict(kernel=label, exact=True, changed_input_exact=True, **timing))
                print(name, m, label, timing['median_ms'], flush=True)
            report['cases'].append(row)
            write_json(args.output/'results.json', report)
        del weight, scale, parts
    report['complete'] = True
    write_json(args.output/'results.json', report)


if __name__ == '__main__':
    main()
