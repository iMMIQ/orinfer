"""Validate Q2A8 projections and a complete pre-dispatched expert FFN.

Run through tools/operators/run.sh. Optional --fixture-dir supplies real
q2-gate_up-packed.npy and q2-down-packed.npy, each containing ten experts.
Inputs/routing are synthetic; this does not measure model TPS or task quality.
"""
import argparse
import gc
import subprocess
from pathlib import Path

import numpy as np
import torch

from kernels.model.q2a8 import q2a8
from kernels.operators.op30_activation_quantization import (
    activation_quantization, swiglu_activation_quantization, launch,
)
from tools.operators.common import (
    configure, benchmark, error, export_kernel, environment, identity, write_json,
)


def decode(packed, k):
    """Independent CPU interpretation of the block format, including signed d."""
    blocks = packed.reshape(*packed.shape[:2], k // 64, 18)
    scale = blocks[..., :2].copy().view('<f2').reshape(*packed.shape[:2], k // 64).astype(np.float32)
    codes = ((blocks[..., 2:, None].astype(np.uint16) >> (np.arange(4) * 2)) & 3)
    values = (codes.reshape(*packed.shape[:2], k // 64, 64).astype(np.float32) - 1) * scale[..., None]
    return torch.from_numpy(values.reshape(*packed.shape[:2], k)).cuda()


def reference_quant(value):
    values = value.float().reshape(*value.shape[:-1], -1, 64)
    amax = values.abs().amax(-1)
    scales = torch.where(amax > 0, (amax / 127).clamp_min(2**-24), 1.).half()
    codes = (values / scales.float()[..., None]).round().clamp(-127, 127).to(torch.int8)
    reconstructed = (codes.float() * scales.float()[..., None]).reshape_as(value)
    return codes.reshape_as(value), scales, reconstructed


def synthesize(e, n, k):
    rng = np.random.default_rng(20261002)
    blocks = rng.integers(0, 256, (e, n, k // 64, 18), dtype=np.uint8)
    scales = rng.uniform(-.04, .04, (e, n, k // 64)).astype('<f2')
    scales[0, 0, :] = 0
    blocks[..., :2] = scales.view(np.uint8).reshape(e, n, k // 64, 2)
    return blocks.reshape(e, n, k // 64 * 18)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture-dir', type=Path)
    args = parser.parse_args()
    configure()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    report = {'environment': environment(), 'cases': [], 'sources': [], 'complete': False,
              'scope': 'Pre-dispatched expert FFN; synthetic activations and mixtures. '
                       'No router, attention, recurrent state or full-model quality/TPS.',
              'resident_layout': 'Q2_0: 64 weights per 18-byte block; no global W8'}

    def save():
        write_json(args.output / 'results.json', report)

    banks = [('synthetic_signed_scales', synthesize(2, 128, 128), synthesize(2, 128, 64), [1, 3, 8, 17])]
    if args.fixture_dir:
        paths = [args.fixture_dir / f'q2-{family}-packed.npy' for family in ('gate_up', 'down')]
        report['sources'] = [identity(path) for path in paths]
        banks.append(('real_layer0_experts', np.load(paths[0]).reshape(10, 1280, 720),
                      np.load(paths[1]).reshape(10, 2560, 180), [1, 3, 8, 64]))
    for label, gp, dp, rows in banks:
        e, n = gp.shape[:2]
        k = gp.shape[2] // 18 * 64
        f = n // 2
        assert dp.shape == (e, k, f // 64 * 18)
        g, d = torch.from_numpy(gp).cuda(), torch.from_numpy(dp).cuda()
        wg, wd = decode(gp, k), decode(dp, f)
        for m in rows:
            a = torch.randn((e, m, k), device='cuda', dtype=torch.float16)
            # Empty/padded expert row exercises scale=1 and output=0.
            a[0, -1] = 0
            mask_g = torch.zeros(k, device='cuda', dtype=torch.uint8)
            mask_d = torch.zeros(f, device='cuda', dtype=torch.uint8)
            aq = torch.empty_like(a, dtype=torch.int8)
            sa = torch.empty((e, m, k // 64), device='cuda', dtype=torch.float16)
            gu = torch.empty((e, m, n), device='cuda', dtype=torch.float16)
            xq = torch.empty((e, m, f), device='cuda', dtype=torch.int8)
            sx = torch.empty((e, m, f // 64), device='cuda', dtype=torch.float16)
            y = torch.empty((e, m, k), device='cuda', dtype=torch.float16)
            weights = torch.softmax(torch.randn((e, m, 1), device='cuda'), dim=0)
            mixture = torch.empty((m, k), device='cuda')
            quant = activation_quantization(k, 64, threads=128)
            fused = swiglu_activation_quantization(f, 64, threads=128)
            for impl in ('register', 'shared'):
                gemg = q2a8(e, m, n, k, implementation=impl)
                gemd = q2a8(e, m, k, f, implementation=impl)

                def run():
                    stream = torch.cuda.current_stream().cuda_stream
                    launch(quant, a.view(e*m, k), mask_g, aq.view(e*m, k), sa.view(e*m, k//64), stream=stream)
                    gemg(aq, g, sa, gu)
                    launch(fused, gu.view(e*m, n), mask_d, xq.view(e*m, f), sx.view(e*m, f//64), stream=stream)
                    gemd(xq, d, sx, y)
                    torch.sum(y.float() * weights, dim=0, out=mixture)

                run()
                torch.cuda.synchronize()
                expected_aq, expected_sa, ar = reference_quant(a)
                assert torch.equal(aq, expected_aq) and torch.equal(sa, expected_sa)
                expected_gu = torch.bmm(ar, wg.transpose(1, 2)).half()
                # Check SwiGLU materialization independently on the GPU output.
                x = (torch.nn.functional.silu(gu[:, :, :f].float()) * gu[:, :, f:].float()).half()
                expected_xq, expected_sx, xr = reference_quant(x)
                # CUDA/Torch sigmoid arithmetic can differ at FP16 boundaries;
                # require a small reconstructed difference, not identical codes.
                actual_xr = (xq.float().reshape(e, m, f//64, 64) * sx.float()[..., None]).reshape_as(x)
                fused_error = error(actual_xr, xr)
                assert fused_error['relative_l2'] < .002, fused_error
                expected_y = torch.bmm(actual_xr, wd.transpose(1, 2)).half()
                eg, ed = error(gu, expected_gu), error(y, expected_y)
                assert eg['relative_l2'] < .002 and ed['relative_l2'] < .002, (eg, ed)
                assert bool((y[0, -1] == 0).all())
                full_gu = torch.bmm(a.float(), wg.transpose(1, 2))
                full_x = torch.nn.functional.silu(full_gu[:, :, :f]) * full_gu[:, :, f:]
                baseline = (torch.bmm(full_x, wd.transpose(1, 2)) * weights).sum(dim=0)
                diagnostic = error(mixture, baseline)
                timing, graph = benchmark(run, repetitions=8)
                expected = y.clone()
                expected_mix = mixture.clone()
                y.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(y, expected)
                original_a = a.clone()
                a.zero_()
                graph.replay()
                torch.cuda.synchronize()
                assert bool((y == 0).all())
                a.copy_(original_a)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(y, expected) and torch.equal(mixture, expected_mix)
                original_g = g.clone()
                g.zero_()
                graph.replay()
                torch.cuda.synchronize()
                assert bool((y == 0).all())
                g.copy_(original_g)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(y, expected)
                folder = args.output / f'{label}-M{m}-{impl}'
                export_kernel(gemg, folder / 'gate_up')
                export_kernel(gemd, folder / 'down')
                assembly = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(folder / 'gate_up/kernel.cubin')], text=True)
                assert 'IMMA.16832.S8.S8' in assembly
                case = {'bank': label, 'rows': m, 'implementation': impl, 'timing': timing,
                        'gate_up_arithmetic': eg, 'down_arithmetic': ed,
                        'fused_swiglu_a8': fused_error, 'ffn_incremental_vs_q2': diagnostic,
                        'sass_int8_verified': True, 'global_w8_bytes': 0,
                        'graph_checks': ['poison', 'changed_activation', 'changed_weight', 'restore']}
                report['cases'].append(case)
                save()
                print(label, m, impl, timing['median_ms'], diagnostic['relative_l2'], flush=True)
                del graph, expected, expected_mix, original_a, original_g
                gc.collect()
            del a, aq, sa, gu, xq, sx, y, mixture
            gc.collect()
        del g, d, wg, wd
        gc.collect()
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
