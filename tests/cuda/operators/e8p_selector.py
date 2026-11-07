"""Check exact E8P selector decoding, routed tails, patches and graph replay."""

import argparse
from pathlib import Path

import numpy as np
import torch

from kernels.cuda.model.integer_vq import integer_vq, integer_vq_grouped
from orinfer_build.compiler.cuda.common import benchmark, configure, write_json
from tests.cuda.operators.quantized_reference import reference, upload
from orinfer_build.quantization.vq import Weights, e8p_selector_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    rng = np.random.default_rng(20261002)
    torch.manual_seed(20261002)
    cases = []
    for bm, m, patched, shared in (
        (8, 1, False, True),
        (8, 3, True, False),
        (16, 17, True, True),
        (32, 33, False, False),
        (64, 65, False, True),
    ):
        e, n, k = 3, 65, 256
        bank = []
        for expert in range(e):
            table = rng.choice(np.array([-10, -6, -2, 0, 2, 6, 10], np.int8), (256, 8))
            if shared and expert:
                table = bank[0].table
            patch = rng.integers(0, 65536, (n, k // 128), dtype=np.uint16) if patched else None
            if patch is not None:
                patch[0, 0] = 32768 | (128 << 7) | 127
            bank.append(
                Weights(
                    "e8p",
                    rng.integers(0, 65536, (n, k // 8), dtype=np.uint16),
                    table,
                    np.full(n, 0.015625, np.float16),
                    np.empty(0, np.int8),
                    patch,
                )
            )
        pp, _, ws = upload(bank)
        book = torch.from_numpy(
            np.stack([e8p_selector_table(w.table) for w in (bank[:1] if shared else bank)])
        ).cuda()
        patch = (
            torch.from_numpy(np.stack([w.patches.T for w in bank]).copy()).cuda()
            if patched
            else torch.zeros(1, device="cuda", dtype=torch.uint16)
        )
        x = torch.randint(-128, 128, (e, m, k), device="cuda", dtype=torch.int8)
        sa = torch.full((e, m), 0.0078125, device="cuda", dtype=torch.float16)
        integer = torch.from_numpy(np.stack([w.integer_weights() for w in bank])).cuda()
        expected = reference(x, integer, ws, sa).view(e * m, n)
        out = torch.full((e * m + 1, n), 91.0, device="cuda", dtype=torch.float16)
        counts = torch.full((e,), m, device="cuda", dtype=torch.int32)
        offsets = torch.arange(e, device="cuda", dtype=torch.int32) * m
        tiles = [(expert, row) for expert in range(e) for row in range(0, m, bm)]
        te = torch.tensor([t[0] for t in tiles], device="cuda", dtype=torch.int32)
        tr = torch.tensor([t[1] for t in tiles], device="cuda", dtype=torch.int32)
        tc = torch.tensor([len(tiles)], device="cuda", dtype=torch.int32)
        for routed in (False, True):
            config = dict(
                kind="e8p",
                shared_table=shared,
                patched=patched,
                block_m=bm,
                block_n=128,
                shortbook=True,
                selectorbook=True,
                num_stages=2 if bm in (8, 64) else 1,
                byte_permute=True,
            )
            op = (
                integer_vq_grouped(e * m, e, len(tiles), n, k, **config)
                if routed
                else integer_vq(e, m, n, k, **config)
            )

            def run():
                operands = [x.view(e * m, k), pp, book, patch, ws, sa.view(-1)]
                if routed:
                    operands += [counts, offsets, te, tr, tc]
                op(*operands, out[: e * m])

            timing, graph = benchmark(run, repetitions=4)
            assert torch.equal(out[: e * m], expected), (bm, m, patched, shared, routed)
            assert bool((out[-1] == 91).all())
            saved_x, saved_book = x.clone(), book.clone()
            x.neg_()
            book.copy_(
                torch.from_numpy(
                    np.stack(
                        [
                            e8p_selector_table(np.zeros_like(w.table))
                            for w in (bank[:1] if shared else bank)
                        ]
                    )
                ).cuda()
            )
            changed = torch.from_numpy(
                np.stack(
                    [
                        Weights(
                            w.kind, w.indices, np.zeros_like(w.table), w.scales, w.signs, w.patches
                        ).integer_weights()
                        for w in bank
                    ]
                )
            ).cuda()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[: e * m], reference(x, changed, ws, sa).view(e * m, n))
            book.copy_(saved_book)
            x.copy_(saved_x)
            if patched:
                saved_patch = patch.clone()
                patch.zero_()
                changed = torch.from_numpy(
                    np.stack(
                        [
                            Weights(w.kind, w.indices, w.table, w.scales, w.signs).integer_weights()
                            for w in bank
                        ]
                    )
                ).cuda()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out[: e * m], reference(x, changed, ws, sa).view(e * m, n))
                patch.copy_(saved_patch)
            if routed:
                counts[0] = 0
                out.fill_(91)
                graph.replay()
                torch.cuda.synchronize()
                assert bool((out[:m] == 91).all())
                assert torch.equal(out[m : e * m], expected[m:])
                counts[0] = m
                tc.zero_()
                out.fill_(91)
                graph.replay()
                torch.cuda.synchronize()
                assert bool((out == 91).all())
                tc.fill_(len(tiles))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[: e * m], expected)
            assert bool((out[-1] == 91).all())
            cases.append(
                dict(
                    block_m=bm,
                    rows=m,
                    routed=routed,
                    patched=patched,
                    shared=shared,
                    exact=True,
                    changed_graph=True,
                    timing=timing,
                )
            )
            write_json(args.output / "results.json", dict(complete=False, cases=cases))
    write_json(args.output / "results.json", dict(complete=True, cases=cases))


if __name__ == "__main__":
    main()
