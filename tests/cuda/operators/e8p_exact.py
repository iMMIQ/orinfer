"""Validate exact E8P/A8 nibble arithmetic, routing tails and the shared K64 bank."""

import argparse
from pathlib import Path

import numpy as np
import torch

from kernels.cuda.model.e8p_exact import e8p_exact_grouped, split_a8
from kernels.cuda.model.e8p_prefill import e8p_prefill_grouped, split_prefill_a8
from kernels.cuda.model.integer_vq import integer_vq_grouped
from orinfer_build.compiler.cuda.common import configure, write_json
from orinfer_build.quantization.vq import (
    e8p_nibble_table,
    e8p_repack_k64,
    e8p_selector_table,
    e8p_sign_table,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    rng = np.random.default_rng(20261002)
    basis = rng.choice(np.array([-10, -6, -2, 2, 6, 10], np.int8), (256, 8))
    selector = e8p_selector_table(basis)
    book = torch.from_numpy(selector[None]).cuda()
    nibble = torch.from_numpy(e8p_nibble_table(selector)[None]).cuda()
    patch = torch.zeros(1, dtype=torch.uint16, device="cuda")
    results = []
    shapes = [
        (8, [1, 0], 160, 640),
        (8, [7, 8], 160, 640),
        (16, [17, 16], 160, 640),
        (32, [33, 32], 160, 640),
        (64, [65, 64], 160, 640),
        (16, [3, 0], 129, 128),
        (16, [5, 9], 257, 2560),
        (8, [9, 17], 1280, 2560),
        (16, [17, 33], 2560, 640),
    ]
    for bm, counts, n, k in shapes:
        rows, e = sum(counts), 2
        tiles = rows // bm + e + 1
        a = torch.randint(-128, 128, (rows, k), dtype=torch.int8, device="cuda")
        # Force A8 extrema and negative values into every input row.
        a[:, :4] = torch.tensor([-128, -1, 0, 127], dtype=torch.int8, device="cuda")
        raw = rng.integers(0, 65536, (e, k // 128, n, 16), dtype=np.uint16)
        p = torch.from_numpy(raw).cuda()
        pp = torch.from_numpy(e8p_repack_k64(raw)).cuda()
        ws = torch.rand((e, n), dtype=torch.float16, device="cuda") / 16
        scale = torch.rand(rows, dtype=torch.float16, device="cuda") / 16
        cs = torch.tensor(counts, dtype=torch.int32, device="cuda")
        offsets = torch.tensor([0, counts[0]], dtype=torch.int32, device="cuda")
        items = [
            (expert, row) for expert, count in enumerate(counts) for row in range(0, count, bm)
        ]
        te = torch.zeros(tiles, dtype=torch.int32, device="cuda")
        tr = torch.zeros_like(te)
        te[: len(items)] = torch.tensor([item[0] for item in items], device="cuda")
        tr[: len(items)] = torch.tensor([item[1] for item in items], device="cuda")
        tc = torch.tensor([len(items)], dtype=torch.int32, device="cuda")
        routing = [cs, offsets, te, tr, tc]
        out = torch.full((rows, n), float("nan"), dtype=torch.float16, device="cuda")
        ref = torch.empty_like(out)
        opts = dict(
            kind="e8p",
            shared_table=True,
            block_m=bm,
            block_n=128,
            shortbook=True,
            selectorbook=True,
            num_stages=1 if bm <= 32 else 2,
        )
        baseline = integer_vq_grouped(rows, e, tiles, n, k, **opts)
        fallback = integer_vq_grouped(rows, e, tiles, n, k, packed_k=64, **opts)
        candidates = [("k64-int8", fallback, book, p, book, baseline)]
        if bm == 8:
            warp = dict(opts, warp_decode=True, block_n=32, parity_xor=True, split_k=2)
            candidates.append(
                (
                    "k64-warp-int8",
                    integer_vq_grouped(rows, e, tiles, n, k, packed_k=64, **warp),
                    book,
                    p,
                    book,
                    integer_vq_grouped(rows, e, tiles, n, k, **warp),
                )
            )
        if bm <= 16:
            lo = torch.empty((rows, k // 8), dtype=torch.uint32, device="cuda")
            hi = torch.empty_like(lo)
            total = torch.empty(rows, dtype=torch.int32, device="cuda")
            pack = split_a8(rows, k)
            for stages in (2, 3, 4):
                if k // 64 < stages - 1:
                    continue
                exact = e8p_exact_grouped(rows, e, tiles, n, k, block_m=bm, stages=stages)
                candidates.append(
                    (f"k64-exact-int4-stages{stages}", exact, nibble, p, book, baseline)
                )
        # All table forms read the identical code bank. Zero-valued bases use
        # the ordinary INT8 fallback, never the nonzero-basis nibble decoder.
        if bm == 16:
            lp = torch.empty((k // 64, rows, 8), dtype=torch.uint32, device="cuda")
            hp = torch.empty_like(lp)
            streamed_pack = split_prefill_a8(rows, k)
            candidates.append(
                (
                    "k64-prefill-int4",
                    e8p_prefill_grouped(rows, e, tiles, n, k),
                    nibble,
                    p,
                    book,
                    baseline,
                )
            )
            zeros = basis.copy()
            zeros[0] = 0
            for label, table, short, selectors in [
                ("basis", zeros.view(np.uint32).reshape(1, 256, 2), False, False),
                ("short", e8p_sign_table(zeros)[None], True, False),
                ("selector-zero", e8p_selector_table(zeros)[None], True, True),
            ]:
                bk = torch.from_numpy(table.copy()).cuda()
                more = dict(
                    opts, shortbook=short, selectorbook=selectors, num_stages=2 if not short else 1
                )
                candidates.append(
                    (
                        label,
                        integer_vq_grouped(rows, e, tiles, n, k, packed_k=64, **more),
                        bk,
                        p,
                        bk,
                        integer_vq_grouped(rows, e, tiles, n, k, **more),
                    )
                )
        for label, kernel, bk, original_p, original_b, reference in candidates:

            def run():
                if label.startswith("k64-exact-int4") or label == "k64-prefill-int4":
                    planes = (lp, hp) if label == "k64-prefill-int4" else (lo, hi)
                    splitter = streamed_pack if label == "k64-prefill-int4" else pack
                    splitter(a, *planes, total)
                    kernel(
                        planes[0].view(torch.int8),
                        planes[1].view(torch.int8),
                        total,
                        pp,
                        bk,
                        ws,
                        scale,
                        *routing,
                        out,
                    )
                else:
                    kernel(a, pp, bk, patch, ws, scale, *routing, out)

            def original():
                reference(a, original_p, original_b, patch, ws, scale, *routing, ref)

            original()
            run()
            torch.cuda.synchronize()
            assert torch.equal(ref, out), (bm, label, "initial")
            if label.startswith("k64-exact-int4") or label == "k64-prefill-int4":
                assert torch.equal(total, a.to(torch.int32).sum(1).to(torch.int32))
                shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int64)
                lplane, hplane = lo, hi
                if label == "k64-prefill-int4":
                    lplane = lp.permute(1, 0, 2).reshape(rows, k // 8)
                    hplane = hp.permute(1, 0, 2).reshape(rows, k // 8)
                low = ((lplane.to(torch.int64).unsqueeze(-1) >> shifts) & 15).flatten(1)
                high = ((hplane.to(torch.int64).unsqueeze(-1) >> shifts) & 15).flatten(1)
                high = torch.where(high >= 8, high - 16, high)
                permuted = a.reshape(rows, k // 8, 8)[:, :, [0, 2, 1, 3, 4, 6, 5, 7]].flatten(1)
                assert torch.equal(low + 16 * high, permuted.to(torch.int64))
            saved = a.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            a.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert bool((out == 0).all())
            a.copy_(torch.randint(-128, 128, a.shape, dtype=torch.int8, device="cuda"))
            original()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(ref, out), (bm, label, "changed")
            a.copy_(saved)
            original()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(ref, out), (bm, label, "restored")
            del graph
            result = dict(
                block_m=bm,
                rows=rows,
                counts=counts,
                N=n,
                K=k,
                candidate=label,
                exact=True,
                changed_input_graph=True,
            )
            results.append(result)
            print(result, flush=True)
    write_json(args.output / "results.json", dict(complete=True, seed=20261002, cases=results))


if __name__ == "__main__":
    main()
