"""Check direct Q2 bank reads, split gate/up and mutable routing under graphs.

Routing and activations are synthetic. Optional fixtures contain real layer-0
expert bytes. This checks projections, not complete-model quality or TPS.
"""
import argparse
import gc
import subprocess
from pathlib import Path

import numpy as np
import torch

from kernels.model.q2a8 import q2a8_indexed
from tools.operators.q2a8 import decode, synthesize
from tools.operators.common import (
    configure, benchmark, error, export_kernel, environment, identity, write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture-dir', type=Path)
    args = parser.parse_args()
    configure()
    report = {'environment': environment(), 'cases': [], 'sources': [],
              'complete': False, 'scope': 'Indexed Q2 projections; synthetic routing '
              'and activations. No full-model quality/TPS.', 'global_w8_bytes': 0}
    banks = [('synthetic_bank512', synthesize(512, 128, 128), [1, 3, 8, 17])]
    if args.fixture_dir:
        source = args.fixture_dir / 'q2-gate_up-packed.npy'
        report['sources'].append(identity(source))
        banks.append(('real_layer0', np.load(source).reshape(10, 1280, 720), [1, 8, 17]))
    for label, packed, rows in banks:
        b, n = packed.shape[:2]
        k = packed.shape[2] // 18 * 64
        reference_bank = decode(packed, k)
        for split in (False, True):
            gate = torch.from_numpy(packed[:, :n//2].copy() if split else packed).cuda()
            up = torch.from_numpy(packed[:, n//2:].copy()).cuda() if split else gate
            e = 6
            ids = torch.tensor([-1, b-1, 0, b-1, b, 3], device='cuda', dtype=torch.int32)
            original_ids = ids.clone()
            for m in rows:
                a = torch.randint(-127, 128, (e, m, k), device='cuda', dtype=torch.int8)
                scale = (torch.rand((e, m, k//64), device='cuda') * .01 + .001).half()
                out = torch.empty((e, m, n), device='cuda', dtype=torch.float16)

                def reference():
                    ar = (a.float().reshape(e, m, k//64, 64) * scale.float()[..., None]).reshape(e, m, k)
                    selected = reference_bank[ids.clamp(0, b-1).long()]
                    value = torch.bmm(ar, selected.transpose(1, 2)).half()
                    value[(ids < 0) | (ids >= b)] = 0
                    return value

                for impl in ('register', 'shared'):
                    kernel = q2a8_indexed(b, e, m, n, k, split_gate_up=split, implementation=impl)

                    def run():
                        kernel(a, gate, up, ids, scale, out)

                    run()
                    torch.cuda.synchronize()
                    expected = reference()
                    metric = error(out, expected)
                    assert metric['finite'] and metric['relative_l2'] < .002, metric
                    assert bool((out[(ids < 0) | (ids >= b)] == 0).all())
                    timing, graph = benchmark(run, repetitions=8)
                    saved = out.clone()
                    out.fill_(float('nan'))
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out, saved)
                    ids.copy_(original_ids.flip(0))
                    graph.replay()
                    torch.cuda.synchronize()
                    changed = error(out, reference())
                    assert changed['finite'] and changed['relative_l2'] < .002, changed
                    assert not torch.equal(out, saved)
                    ids.copy_(original_ids)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out, saved)
                    original_a = a.clone()
                    a.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert bool((out == 0).all())
                    a.copy_(original_a)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out, saved)
                    original_gate = gate.clone()
                    gate.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert bool((out[..., :n//2] == 0).all())
                    if split:
                        assert torch.equal(out[..., n//2:], saved[..., n//2:])
                    else:
                        assert bool((out == 0).all())
                    gate.copy_(original_gate)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out, saved)
                    folder = args.output / f'{label}-M{m}-split{int(split)}-{impl}'
                    export_kernel(kernel, folder)
                    sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-sass',
                                                    str(folder / 'kernel.cubin')], text=True)
                    assert 'IMMA.16832.S8.S8' in sass
                    case = {'bank': label, 'bank_experts': b, 'rows': m, 'split_gate_up': split,
                            'implementation': impl, 'timing': timing, 'arithmetic': metric,
                            'changed_routing': changed, 'sass_int8_verified': True,
                            'graph_checks': ['poison', 'changed_ids', 'changed_activation',
                                             'changed_bank', 'restore']}
                    report['cases'].append(case)
                    write_json(args.output / 'results.json', report)
                    print(label, m, split, impl, timing['median_ms'], flush=True)
                    del graph, saved, original_a, original_gate, expected
                    gc.collect()
                del a, scale, out
            del gate, up
        del reference_bank
        gc.collect()
    report['complete'] = True
    write_json(args.output / 'results.json', report)


if __name__ == '__main__':
    main()
