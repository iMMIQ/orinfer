"""Check HC BF16 operand reuse and fused mix against the original FP16 chain."""

import argparse
from pathlib import Path

import torch

from kernels.cuda.model.hc_prefill import (
    hc_combine_norm_prefill,
    hc_norm_prefill,
    hc_projection_prefill,
    hc_up_mix_prefill,
)
from kernels.cuda.model.hyperconnection import hc_combine_norm, hc_mix, hc_norm, hc_projection
from orinfer_build.compiler.cuda.common import configure, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-tile-only", action="store_true")
    args = parser.parse_args()
    configure()
    cases = []
    geometries = [
        (1, 64, 64, False),
        (3, 64, 320, True),
        (17, 64, 64, False),
        (1, 2560, 320, True),
        (129, 64, 320, True),
    ]
    if args.production_tile_only:
        geometries = geometries[-1:]
    for m, h, rank, combined in geometries:
        residual = torch.randn((m, 4, h), device="cuda", dtype=torch.float16)
        original = residual.clone()
        weight = torch.randn((4, h), device="cuda") / 10
        block = torch.randn((m, h), device="cuda", dtype=torch.float16)
        previous = torch.randn((m, 4), device="cuda", dtype=torch.float16) / 10
        previous[0] = torch.tensor([0.02734375, -0.025390625, 0, 1], device="cuda")
        dw = torch.randn((rank, 4 * h), device="cuda", dtype=torch.bfloat16) / 16
        uw = torch.randn((4 * h, rank), device="cuda", dtype=torch.bfloat16) / 16
        iw = torch.randn((4, 4 * h), device="cuda", dtype=torch.bfloat16) / 16
        normed = torch.empty_like(residual)
        activation = torch.empty((m, rank), device="cuda", dtype=torch.float16)
        up = torch.empty_like(residual)
        reference = torch.empty((m, h), device="cuda", dtype=torch.float16)
        injection = torch.empty((m, 4), device="cuda", dtype=torch.float16)
        bf = torch.empty_like(residual, dtype=torch.bfloat16)
        abf = torch.empty_like(activation, dtype=torch.bfloat16)
        inv = torch.empty((m, 4), device="cuda")
        arena = torch.full((m + 2, h), 123, device="cuda", dtype=torch.float16)
        out = arena[:m]
        new_injection = torch.empty_like(injection)
        old_norm = (hc_combine_norm if combined else hc_norm)(m, h, dtype="float16")
        new_norm = (hc_combine_norm_prefill if combined else hc_norm_prefill)(m, h, dtype="float16")
        down_m = 64 if m >= 128 else 16
        up_m = 128 if m >= 128 else 16
        group_m = 4 if m >= 128 else 0
        old_down = hc_projection(
            m, rank, 4 * h, dtype="float16", block_m=down_m, block_n=128, group_m=group_m, silu=True
        )
        new_down = hc_projection_prefill(
            m,
            rank,
            4 * h,
            block_m=down_m,
            block_n=128,
            group_m=group_m,
            silu=True,
            output_dtype="bfloat16",
        )
        old_up = hc_projection(m, 4 * h, rank, dtype="float16", block_m=up_m, block_n=128)
        new_up = hc_up_mix_prefill(m, h, rank, block_m=up_m)
        old_mix = hc_mix(m, h, dtype="float16")
        old_inj = hc_projection(m, 4, 4 * h, dtype="float16", block_m=16, block_n=16)
        new_inj = hc_projection_prefill(m, 4, 4 * h, block_m=16, block_n=16, stages=8)

        def baseline():
            residual.copy_(original)
            if combined:
                old_norm(block, residual, previous, weight, residual, normed)
            else:
                old_norm(residual, weight, normed)
            old_down(normed, dw, activation)
            old_up(activation, uw, up)
            old_mix(normed, up, reference)
            old_inj(normed, iw, injection)

        def run():
            residual.copy_(original)
            if combined:
                new_norm(block, residual, previous, weight, residual, bf, inv)
            else:
                new_norm(residual, weight, bf, inv)
            new_down(bf, dw, abf)
            new_up(abf, uw, residual, weight, inv, out)
            new_inj(bf, iw, new_injection)

        def verify():
            torch.cuda.synchronize()
            for label, actual, expected in [
                ("norm", bf, normed.to(torch.bfloat16)),
                ("activation", abf, activation.to(torch.bfloat16)),
                ("mix", out, reference),
                ("injection", new_injection, injection),
            ]:
                assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16)), (
                    label,
                    m,
                    h,
                    rank,
                    int((actual.view(torch.uint16) != expected.view(torch.uint16)).sum()),
                    float((actual.float() - expected.float()).abs().max()),
                )
            assert bool((arena[m:] == 123).all())

        baseline()
        run()
        verify()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        saved = original.clone()
        original.zero_()
        baseline()
        graph.replay()
        verify()
        original.copy_(saved)
        baseline()
        graph.replay()
        verify()
        cases.append(
            dict(
                rows=m,
                hidden=h,
                rank=rank,
                combined=combined,
                exact=True,
                changed_graph=True,
                tail_guard=True,
            )
        )
        print(cases[-1], flush=True)
    write_json(args.output / "results.json", dict(complete=True, seed=20261002, cases=cases))


if __name__ == "__main__":
    main()
