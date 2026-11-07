"""Exact two-stage E8P K-major projections, tails and mutable Graph inputs."""

import argparse
from pathlib import Path
import numpy as np
import torch
from kernels.cuda.model.e8p_q4 import projection, unpack
from orinfer_build.compiler.compile import primitive
from orinfer_build.compiler.cuda.common import configure, write_json
from orinfer_build.quantization.e8p import basis
from orinfer_build.quantization.vq import e8p_decode, e8p_selector_table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure()
    rng = np.random.default_rng(20261002)
    results = []

    def launch(kernel, *values):
        kernel.adapter.func(*values, stream=torch.cuda.current_stream().cuda_stream)

    def integers(codes, table, coefficients):
        a, b = coefficients
        return (
            a * e8p_decode(codes.astype(np.uint16), table).astype(np.int32)
            + b * e8p_decode((codes >> 16).astype(np.uint16), table).astype(np.int32)
        ).reshape(codes.shape[0], -1)

    for packed_k in (64, 128):
        for coefficients in ((3, 1), (4, 1), (7, 2)):
            first = np.arange(65536, dtype=np.uint32).reshape(4096, 16)
            codes = first | (((first * 32771 + 97) & 65535) << 16)
            table = basis()
            book = torch.from_numpy(e8p_selector_table(table)).cuda()

            def physical(c):
                return np.ascontiguousarray(
                    c.reshape(c.shape[0], -1, packed_k // 8).transpose(1, 0, 2)
                )

            weight = torch.from_numpy(physical(codes)).cuda()
            output = torch.empty((4096, 128), device="cuda", dtype=torch.int8)
            decoder = primitive(unpack, 4096, 128, coefficients=coefficients, packed_k=packed_k)
            launch(decoder, weight, book, output)
            np.testing.assert_array_equal(
                output.cpu().numpy(), integers(codes, table, coefficients)
            )
            results.append(dict(kind="all_codes", packed_k=packed_k, coefficients=coefficients))

            for bn in (32, 64, 128):
                x_all = torch.randint(-128, 128, (1, 128), device="cuda", dtype=torch.int8)
                scale_all = torch.ones(1, device="cuda", dtype=torch.float16)
                ws_all = torch.ones(4096, device="cuda", dtype=torch.float16)
                out_all = torch.empty((1, 4096), device="cuda")
                direct = primitive(
                    projection,
                    1,
                    4096,
                    128,
                    coefficients=coefficients,
                    algorithm="int4_direct",
                    block_n=bn,
                    block_k=128,
                    packed_k=packed_k,
                    basis_nonzero=True,
                    output_dtype="float32",
                )
                launch(direct, x_all, weight, book, ws_all, scale_all, out_all)
                all_int = torch.from_numpy(
                    integers(codes, table, coefficients).astype(np.float32)
                ).cuda()
                torch.testing.assert_close(out_all, x_all.float() @ all_int.T, rtol=0, atol=0)
                results.append(
                    dict(
                        kind="all_codes_direct",
                        packed_k=packed_k,
                        coefficients=coefficients,
                        block_n=bn,
                    )
                )

            n, k = 65, 256
            codes = rng.integers(0, 2**32, (n, k // 8), dtype=np.uint32)
            weight = torch.from_numpy(physical(codes)).cuda()
            ws = torch.linspace(0.0001, 0.001, n, device="cuda", dtype=torch.float16)
            for algorithm, bm, bn, bk, sk in (
                ("tensor_core", 8, 64, 128, 1),
                ("dp4a", 4, 16, 128, 1),
                ("int4_direct", 8, 32, 128, 1),
                ("int4_direct", 8, 32, 128, 2),
                ("int4_direct", 8, 64, 128, 1),
                ("int4_direct", 8, 64, 128, 2),
                ("int4_direct", 8, 128, 128, 1),
                ("int4_direct", 8, 128, 128, 2),
            ):
                for dynamic in (False, True):
                    kernel = primitive(
                        projection,
                        4096 if dynamic else 3,
                        n,
                        k,
                        coefficients=coefficients,
                        algorithm=algorithm,
                        block_m=bm,
                        block_n=bn,
                        block_k=bk,
                        split_k=sk,
                        dynamic_rows=dynamic,
                        output_dtype="float32",
                        output_stride=n + 3,
                        packed_k=packed_k,
                        basis_nonzero=True,
                    )
                    for rows in (1, 2, 3, 4, 8, 17, 129) if dynamic else (3,):
                        x = torch.randint(-128, 128, (rows, k), device="cuda", dtype=torch.int8)
                        x[0, ::2], x[0, 1::2] = -128, 127
                        scales = torch.linspace(
                            0.001, 0.002, rows, device="cuda", dtype=torch.float16
                        )
                        out = torch.full((rows + 1, n + 3), 91.0, device="cuda")

                        def run():
                            values = (
                                x.view(torch.int32) if algorithm == "dp4a" else x,
                                weight,
                                book,
                                ws,
                                scales,
                                out,
                            )
                            launch(kernel, *values, *([rows] if dynamic else []))

                        def check():
                            decoded = torch.from_numpy(
                                integers(codes, table, coefficients).astype(np.float32)
                            ).cuda()
                            expected = (
                                (x.float() @ decoded.T) * ws.float()[None]
                            ) * scales.float()[:, None]
                            torch.testing.assert_close(out[:rows, :n], expected, rtol=0, atol=0)
                            assert bool(torch.all(out[rows:] == 91)) and bool(
                                torch.all(out[:, n:] == 91)
                            )

                        run()
                        check()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                        x.random_(-128, 128)
                        codes ^= np.uint32(0x01000100)
                        weight.copy_(torch.from_numpy(physical(codes)).cuda())
                        table = table[:, ::-1].copy()
                        book.copy_(torch.from_numpy(e8p_selector_table(table)).cuda())
                        out.fill_(91)
                        graph.replay()
                        check()
                        results.append(
                            dict(
                                kind="projection",
                                packed_k=packed_k,
                                coefficients=coefficients,
                                algorithm=algorithm,
                                split_k=sk,
                                dynamic=dynamic,
                                rows=rows,
                                changed_input_weight_book_graph=True,
                            )
                        )
                        print(results[-1], flush=True)
            # Zero coordinates remain supported by the shared INT8 schedules.
            table[:, ::3] = 0
            book.copy_(torch.from_numpy(e8p_selector_table(table)).cuda())
            for algorithm, bm, bn in (("tensor_core", 8, 64), ("dp4a", 4, 16)):
                kernel = primitive(
                    projection,
                    3,
                    n,
                    k,
                    coefficients=coefficients,
                    algorithm=algorithm,
                    block_m=bm,
                    block_n=bn,
                    packed_k=packed_k,
                    output_dtype="float32",
                )
                x = torch.randint(-128, 128, (3, k), device="cuda", dtype=torch.int8)
                scales = torch.ones(3, device="cuda", dtype=torch.float16)
                out = torch.empty((3, n), device="cuda")
                launch(
                    kernel,
                    x.view(torch.int32) if algorithm == "dp4a" else x,
                    weight,
                    book,
                    ws,
                    scales,
                    out,
                )
                decoded = torch.from_numpy(
                    integers(codes, table, coefficients).astype(np.float32)
                ).cuda()
                torch.testing.assert_close(
                    out, (x.float() @ decoded.T) * ws.float()[None], rtol=0, atol=0
                )
                results.append(
                    dict(
                        kind="zero_basis",
                        packed_k=packed_k,
                        coefficients=coefficients,
                        algorithm=algorithm,
                    )
                )
    results.extend(partition_checks(rng))
    results.extend(staged_tensor_checks(rng))
    write_json(args.output / "result.json", dict(passed=True, seed=20261002, checks=results))


def partition_checks(rng):
    """Full-K INT64 oracle checks for four-warp accumulation and FP16 output."""
    records = []
    for pk in [64, 128]:
        for k in [512, 17408]:
            n = 65
            codes = rng.integers(0, 2**32, (n, k // 8), dtype=np.uint32)
            table = basis()
            book = torch.from_numpy(e8p_selector_table(table)).cuda()
            w = torch.from_numpy(
                np.ascontiguousarray(codes.reshape(n, k // pk, pk // 8).transpose(1, 0, 2))
            ).cuda()
            decoded = (
                4 * e8p_decode(codes.astype(np.uint16), table).astype(np.int64)
                + e8p_decode((codes >> 16).astype(np.uint16), table).astype(np.int64)
            ).reshape(n, k)
            ws = torch.full((n,), 0.001, device="cuda", dtype=torch.float16)
            for bn in [32, 64]:
                ker = primitive(
                    projection,
                    129,
                    n,
                    k,
                    algorithm="int4_direct",
                    block_n=bn,
                    block_k=128,
                    split_k=4,
                    packed_k=pk,
                    basis_nonzero=True,
                    dynamic_rows=True,
                    output_stride=n + 3,
                )
                for rows in [1, 3, 17, 129]:
                    x = torch.randint(-128, 128, (rows, k), device="cuda", dtype=torch.int8)
                    scale = torch.full((rows,), 0.001, device="cuda", dtype=torch.float16)
                    out = torch.full((rows + 1, n + 3), 91.0, device="cuda", dtype=torch.float16)

                    def run():
                        ker.adapter.func(
                            x,
                            w,
                            book,
                            ws,
                            scale,
                            out,
                            rows,
                            stream=torch.cuda.current_stream().cuda_stream,
                        )

                    def check():
                        dot = (x.cpu().numpy().astype(np.int64) @ decoded.T).astype(np.float32)
                        expected = (
                            dot * ws.cpu().numpy().astype(np.float32)[None]
                        ) * scale.cpu().numpy().astype(np.float32)[:, None]
                        torch.testing.assert_close(
                            out[:rows, :n],
                            torch.from_numpy(expected.astype(np.float16)).cuda(),
                            rtol=0,
                            atol=0,
                        )
                        assert torch.all(out[rows:] == 91) and torch.all(out[:, n:] == 91)

                    run()
                    check()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        run()
                    x.random_(-128, 128)
                    out.fill_(91)
                    g.replay()
                    check()
                    records.append(dict(pk=pk, k=k, bn=bn, rows=rows, exact=True, graph=True))
                    print(records[-1], flush=True)
    return records


def staged_tensor_checks(rng):
    """Async compressed slabs cover partial N/M tiles and multiple K128 slabs."""
    n, k = 65, 256
    codes = rng.integers(0, 2**32, (n, k // 8), dtype=np.uint32)
    table = basis()
    book = torch.from_numpy(e8p_selector_table(table)).cuda()
    physical = np.ascontiguousarray(codes.reshape(n, -1, 16).transpose(1, 0, 2))
    weight = torch.from_numpy(physical).cuda()
    decoded = torch.from_numpy(
        (
            4 * e8p_decode(codes.astype(np.uint16), table).astype(np.int32)
            + e8p_decode((codes >> 16).astype(np.uint16), table).astype(np.int32)
        )
        .reshape(n, k)
        .astype(np.float32)
    ).cuda()
    ws = torch.linspace(0.001, 0.002, n, device="cuda", dtype=torch.float16)
    records = []
    for bm, bn, bk, stages in (
        (128, 128, 128, 2),
        (128, 128, 256, 2),
        (64, 64, 256, 1),
        (256, 64, 128, 1),
    ):
        kernel = primitive(
            projection,
            4096,
            n,
            k,
            block_m=bm,
            block_n=bn,
            block_k=bk,
            stages=stages,
            packed_k=128,
            dynamic_rows=True,
            output_dtype="float32",
            output_stride=n + 3,
        )
        for rows in (1, 127, 128, 129, 513):
            x = torch.randint(-128, 128, (rows, k), device="cuda", dtype=torch.int8)
            scale = torch.full((rows,), 0.001, device="cuda", dtype=torch.float16)
            out = torch.full((rows + 1, n + 3), 91.0, device="cuda")

            def run():
                kernel.adapter.func(
                    x,
                    weight,
                    book,
                    ws,
                    scale,
                    out,
                    rows,
                    stream=torch.cuda.current_stream().cuda_stream,
                )

            def check():
                expected = ((x.float() @ decoded.T) * ws.float()[None]) * scale.float()[:, None]
                torch.testing.assert_close(out[:rows, :n], expected, rtol=0, atol=0)
                assert bool(torch.all(out[rows:] == 91)) and bool(torch.all(out[:, n:] == 91))

            run()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            x.random_(-128, 128)
            out.fill_(91)
            graph.replay()
            check()
            records.append(
                dict(
                    kind="staged_tensor_tail",
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    stages=stages,
                    rows=rows,
                    exact=True,
                    graph=True,
                )
            )
    return records


if __name__ == "__main__":
    main()
