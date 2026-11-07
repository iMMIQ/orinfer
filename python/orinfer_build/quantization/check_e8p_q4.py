"""SM87 checks for two-stage Q4: integer decoder, row tails and graph replay."""

import argparse
from pathlib import Path

import numpy as np
import torch

from kernels.cuda.model.e8p_q4 import projection, unpack
from kernels.cuda.model.int8_projection import int8_projection, int8_small_projection
from kernels.cuda.model.int8_swiglu import int8_swiglu
from kernels.cuda.model.q4_projection import q4_projection, q4_gemv
from orinfer_build.compiler.compile import primitive
from orinfer_build.quantization.e8p import basis
from orinfer_build.quantization.e8p_q4 import Weights, fit
from orinfer_build.quantization.e8p_q4_gpu import Encoder
from orinfer_build.quantization.vq import e8p_decode, e8p_selector_table, rotate
from orinfer_build.compiler.cuda.common import configure, benchmark, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure()
    rng = np.random.default_rng(20261002)
    book = torch.from_numpy(e8p_selector_table(basis())).cuda()
    results = []

    def launch(kernel, *values):
        kernel.adapter.func(*values, stream=torch.cuda.current_stream().cuda_stream)

    for coefficients in [(3, 1), (4, 1), (7, 2)]:
        first = np.arange(65536, dtype=np.uint32).reshape(4096, 16)
        second = (first * 32771 + 97) & 65535
        fixture = Weights(
            first | (second << 16), np.ones(4096, np.float16), np.ones(128, np.int8), coefficients
        )
        weight = torch.from_numpy(fixture.packed).cuda()
        out = torch.empty((4096, 128), device="cuda", dtype=torch.int8)
        decoder = primitive(unpack, 4096, 128, coefficients=coefficients)
        launch(decoder, weight, book, out.view(torch.uint32))
        torch.cuda.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), fixture.integers())
        timing, _ = benchmark(lambda: launch(decoder, weight, book, out.view(torch.uint32)))
        results.append(
            dict(
                kind="exhaustive_decoder",
                coefficients=coefficients,
                codes_per_stage=65536,
                ms=timing["median_ms"],
            )
        )
        print(results[-1], flush=True)

    # The hoisted sign/parity table must also preserve zero basis coordinates,
    # whose odd-parity offset cannot be represented by a parity XOR.
    zero_basis = basis().copy()
    zero_basis[:, ::3] = 0
    zero_book = torch.from_numpy(e8p_selector_table(zero_basis)).cuda()
    first_np, second_np = first, second
    for coefficients in [(3, 1), (4, 1), (7, 2)]:
        decoder = primitive(unpack, 4096, 128, coefficients=coefficients)
        expected = (
            (
                coefficients[0]
                * e8p_decode(first_np.astype(np.uint16), zero_basis).astype(np.int16)
                + coefficients[1]
                * e8p_decode(second_np.astype(np.uint16), zero_basis).astype(np.int16)
            )
            .astype(np.int8)
            .reshape(4096, 128)
        )
        launch(decoder, weight, zero_book, out.view(torch.uint32))
        np.testing.assert_array_equal(out.cpu().numpy(), expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(decoder, weight, zero_book, out.view(torch.uint32))
        zero_book.copy_(book)
        graph.replay()
        np.testing.assert_array_equal(
            out.cpu().numpy(),
            Weights(
                first_np | (second_np << 16),
                np.ones(4096, np.float16),
                np.ones(128, np.int8),
                coefficients,
            ).integers(),
        )
        zero_book.copy_(torch.from_numpy(e8p_selector_table(zero_basis)).cuda())
        results.append(
            dict(kind="zero_basis_decoder", coefficients=coefficients, changed_book_graph=True)
        )

    encoder = Encoder()
    for n, k in [(3, 128), (65, 256), (64, 5120)]:
        raw = rng.normal(0, 0.02, (n, k)).astype(np.float32)
        raw[0] = 0
        signs = rng.choice(np.array([-1, 1], np.int8), k)
        for coefficients in [(3, 1), (4, 1), (7, 2)]:
            candidate = encoder.fit(raw, signs, coefficients=coefficients)
            repeated = encoder.fit(raw, signs, coefficients=coefficients)
            np.testing.assert_array_equal(candidate.packed, repeated.packed)
            np.testing.assert_array_equal(candidate.scales, repeated.scales)
            oracle = fit(raw[:3], signs, coefficients=coefficients)
            target = rotate(raw[:3], signs)
            error = np.square(candidate.dequantize()[:3] - target).sum()
            reference = np.square(oracle.dequantize() - target).sum()
            assert error <= reference * 1.003 + 1e-10, (n, k, coefficients, error, reference)
            results.append(
                dict(
                    kind="fit",
                    n=n,
                    k=k,
                    coefficients=coefficients,
                    sse=float(error),
                    cpu_sse=float(reference),
                    deterministic=True,
                )
            )
            print(results[-1], flush=True)

    n, k = 65, 256
    raw = rng.normal(0, 0.02, (n, k)).astype(np.float32)
    signs = rng.choice(np.array([-1, 1], np.int8), k)
    fixture = encoder.fit(raw, signs)
    weight = torch.from_numpy(fixture.packed).cuda()
    ws = torch.from_numpy(fixture.scales).cuda()
    decoded = torch.from_numpy(fixture.integers().astype(np.float32)).cuda()
    for algorithm, bm, bn in [("tensor_core", 8, 64), ("dp4a", 4, 16)]:
        kernel = primitive(
            projection,
            4096,
            n,
            k,
            algorithm=algorithm,
            block_m=bm,
            block_n=bn,
            dynamic_rows=True,
            output_dtype="float32",
            output_stride=n + 3,
        )
        for rows in [1, 2, 3, 4, 8, 17, 129]:
            x = torch.randint(-128, 128, (rows, k), device="cuda", dtype=torch.int8)
            scales = torch.rand(rows, device="cuda", dtype=torch.float16) * 0.001 + 0.0001
            output = torch.full((rows + 1, n + 3), 91.0, device="cuda")

            def run():
                a = x.view(torch.int32) if algorithm == "dp4a" else x
                launch(kernel, a, weight, book, ws, scales, output[:rows], rows)

            def check():
                expected = (x.float() @ decoded.T) * ws.float()[None] * scales.float()[:, None]
                torch.testing.assert_close(output[:rows, :n], expected, rtol=1e-5, atol=1e-7)
                assert bool(torch.all(output[rows:] == 91)) and bool(torch.all(output[:, n:] == 91))

            run()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            x.random_(-128, 128)
            graph.replay()
            check()
            weight.view(torch.int32).bitwise_xor_(0x01000100)
            decoded.copy_(
                torch.from_numpy(
                    Weights(weight.cpu().numpy(), fixture.scales, signs)
                    .integers()
                    .astype(np.float32)
                ).cuda()
            )
            graph.replay()
            check()
            timing, _ = benchmark(run)
            results.append(
                dict(
                    kind="projection",
                    algorithm=algorithm,
                    rows=rows,
                    max_abs_error=float(
                        (
                            output[:rows, :n]
                            - (x.float() @ decoded.T) * ws.float()[None] * scales.float()[:, None]
                        )
                        .abs()
                        .max()
                    ),
                    changed_input_graph=True,
                    ms=timing["median_ms"],
                )
            )
            print(results[-1], flush=True)

    # The shared-loader refactor must preserve both existing Q4 schedules.
    packed = torch.randint(0, 65536, (n, k // 4), device="cuda", dtype=torch.int32).to(torch.uint16)
    bytes_ = packed.cpu().numpy().view(np.uint8)
    from orinfer_build.quantization.row_q4 import Weights as ScalarWeights

    q = ScalarWeights(bytes_, fixture.scales, signs).integers().astype(np.float32)
    x = torch.randint(-128, 128, (3, k), device="cuda", dtype=torch.int8)
    scales = torch.ones(3, device="cuda", dtype=torch.float16)
    for fn, kwargs in [(q4_projection, dict(block_m=8)), (q4_gemv, dict(BN=16, block_m=4))]:
        kernel = primitive(fn, 4096, n, k, dynamic_rows=True, output_dtype="float32", **kwargs)
        output = torch.empty((3, n), device="cuda")
        launch(kernel, x.view(torch.int32) if fn is q4_gemv else x, packed, ws, scales, output, 3)
        torch.testing.assert_close(
            output,
            (x.float() @ torch.from_numpy(q).cuda().T) * ws.float()[None],
            rtol=1e-5,
            atol=1e-7,
        )
        results.append(dict(kind="scalar_q4_regression", algorithm=fn.__name__, passed=True))

    # W8 projections and fused SwiGLU use the same compute schedules.
    x = torch.randint(-128, 128, (17, k), device="cuda", dtype=torch.int8)
    gate = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
    up = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
    ts = torch.full((17,), 0.001, device="cuda", dtype=torch.float16)
    gs = torch.rand(n, device="cuda", dtype=torch.float16) * 0.001 + 0.0001
    us = torch.rand(n, device="cuda", dtype=torch.float16) * 0.001 + 0.0001
    gate_reference = (x.float() @ gate.float().T) * gs.float()[None] * ts.float()[:, None]
    up_reference = (x.float() @ up.float().T) * us.float()[None] * ts.float()[:, None]
    for algorithm in ["tensor_core", "dp4a"]:
        packed = algorithm == "dp4a"
        a, g, u = (v.view(torch.int32) if packed else v for v in (x, gate, up))
        fn = int8_small_projection if packed else int8_projection
        kwargs = dict(block_m=4, block_n=16) if packed else dict(block_m=16, block_k=128)
        kernel = primitive(fn, 4096, n, k, dynamic_rows=True, output_dtype="float32", **kwargs)
        output = torch.empty((17, n), device="cuda")
        launch(kernel, a, g, gs, ts, output, 17)
        torch.testing.assert_close(output, gate_reference, rtol=1e-5, atol=1e-7)
        kernel = primitive(
            int8_swiglu,
            4096,
            n,
            k,
            algorithm=algorithm,
            block_m=4 if packed else 16,
            block_n=16 if packed else 64,
            dynamic_rows=True,
        )
        output = torch.empty((17, n), device="cuda", dtype=torch.float16)
        launch(kernel, a, g, u, gs, us, ts, output, 17)
        gr, ur = gate_reference.half().float(), up_reference.half().float()
        expected = (gr * torch.sigmoid(gr) * ur).half()
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-6)
        results.append(dict(kind="w8_and_swiglu_regression", algorithm=algorithm, passed=True))
    write_json(args.output / "result.json", dict(passed=True, seed=20261002, checks=results))


if __name__ == "__main__":
    main()
