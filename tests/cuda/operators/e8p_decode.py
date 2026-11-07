"""Exact E8P warp decode, routing tails and changed-input graph validation."""

import argparse
from pathlib import Path
import subprocess

import numpy as np
import torch

from kernels.cuda.model.integer_vq import integer_vq_grouped
from orinfer_build.models.flash_next.kernel_policy import expert_projection_config
from orinfer_build.contracts.abi import parse_host
from orinfer_build.compiler.cuda.common import configure, export_kernel, write_json
from orinfer_build.quantization.vq import e8p_decode, e8p_selector_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short", action="store_true", help="Omit the exhaustive code case")
    parser.add_argument(
        "--parity-xor", action="store_true", help="Validate the nonzero-basis decoder"
    )
    args = parser.parse_args()
    configure()
    rng = np.random.default_rng(20261002)
    records = []
    for m, n, k in ((1, 65, 640), (3, 65, 640), (8, 128, 256), (17, 65, 640), (8, 4096, 128)):
        if args.short and n == 4096:
            continue
        e = 1 if n == 4096 else 3
        codes = rng.integers(0, 65536, (e, k // 128, n, 16), dtype=np.uint16)
        if n == 4096:
            codes[0, 0] = np.arange(65536, dtype=np.uint16).reshape(n, 16)
        values = [-10, -6, -2, 2, 6, 10] if args.parity_xor else [-10, -6, -2, 0, 2, 6, 10]
        basis = rng.choice(np.array(values, np.int8), (256, 8))
        integers = e8p_decode(codes, basis).transpose(0, 2, 1, 3, 4).reshape(e, n, k).copy()
        weights = torch.from_numpy(integers).cuda().double()
        x = torch.randint(-128, 128, (e * m, k), device="cuda", dtype=torch.int8)
        ws = (torch.rand((e, n), device="cuda") / 64).half()
        sa = (torch.rand(e * m, device="cuda") / 128).half()
        book = torch.from_numpy(e8p_selector_table(basis)).cuda().view(1, 256, 1)
        packed = torch.from_numpy(codes).cuda()
        patch = torch.zeros(1, device="cuda", dtype=torch.uint16)
        out = torch.full((e * m + 1, n), 91.0, device="cuda", dtype=torch.float16)
        counts = torch.full((e,), m, device="cuda", dtype=torch.int32)
        offsets = torch.arange(e, device="cuda", dtype=torch.int32) * m
        pairs = [(expert, row) for expert in range(e) for row in range(0, m, 8)]
        te = torch.tensor([expert for expert, _ in pairs], device="cuda", dtype=torch.int32)
        tr = torch.tensor([row for _, row in pairs], device="cuda", dtype=torch.int32)
        tc = torch.tensor([len(pairs)], device="cuda", dtype=torch.int32)

        def reference():
            integer = torch.bmm(x.view(e, m, k).double(), weights.transpose(1, 2)).float()
            return (
                ((integer * ws[:, None, :].float()) * sa.view(e, m, 1).float())
                .half()
                .view(e * m, n)
            )

        expected = reference()
        for bn, split in ((32, 2), (64, 1)):
            plan = expert_projection_config(1, k, parity_xor=args.parity_xor)
            plan.update(block_n=bn, split_k=split)
            kernel = integer_vq_grouped(
                e * m, e, len(pairs), n, k, kind="e8p", shared_table=True, **plan
            )
            folder = args.output / f"m{m}-n{n}-k{k}-bn{bn}-sk{split}"
            exported = export_kernel(kernel, folder)
            (host,) = parse_host((folder / "host.txt").read_text())
            roles = [a["value"].removesuffix(".data_ptr()") for a in host["ordered_arguments"]]
            assert roles == [
                "A",
                "AS",
                "Book",
                "C",
                "Counts",
                "Offsets",
                "P",
                "TileCount",
                "TileExpert",
                "TileRow",
                "WS",
            ]
            sass = subprocess.check_output(
                ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(folder / "kernel.cubin")],
                text=True,
            )
            assert "IMMA.16832.S8.S8" in sass

            def run(kernel=kernel):
                kernel(x, packed, book, patch, ws, sa, counts, offsets, te, tr, tc, out[: e * m])

            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[: e * m], expected) and bool((out[-1] == 91).all())
            x.neg_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[: e * m], reference())
            x.neg_()
            counts[0] = 0
            out.fill_(91)
            graph.replay()
            torch.cuda.synchronize()
            assert bool((out[:m] == 91).all()) and torch.equal(out[m : e * m], expected[m:])
            counts[0] = m
            tc.fill_(0)
            out.fill_(91)
            graph.replay()
            torch.cuda.synchronize()
            assert bool((out == 91).all())
            tc.fill_(len(pairs))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[: e * m], expected)
            # Read changed LUT data on replay; XOR parity uses a nonzero basis.
            replacement = -basis if args.parity_xor else np.zeros_like(basis)
            book.copy_(torch.from_numpy(e8p_selector_table(replacement)).cuda().view_as(book))
            zero_integer = (
                e8p_decode(codes, replacement).transpose(0, 2, 1, 3, 4).reshape(e, n, k).copy()
            )
            original_weights = weights
            weights = torch.from_numpy(zero_integer).cuda().double()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[: e * m], reference())
            weights = original_weights
            book.copy_(torch.from_numpy(e8p_selector_table(basis)).cuda().view_as(book))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[: e * m], expected)
            record = dict(
                m=m,
                n=n,
                k=k,
                block_n=bn,
                split_k=split,
                exact=True,
                parity_xor=args.parity_xor,
                changed_input_and_table_graph=True,
                routing_masks=True,
                output_sentinel=True,
                all_65536_codes=n == 4096,
                export=exported,
            )
            records.append(record)
            print({key: value for key, value in record.items() if key != "export"}, flush=True)
            write_json(args.output / "results.json", dict(complete=False, cases=records))
            del graph, kernel
    write_json(args.output / "results.json", dict(complete=True, cases=records))


if __name__ == "__main__":
    main()
