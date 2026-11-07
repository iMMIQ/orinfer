"""Exact E8P selectors and one-row GDN convolution/history GPU checks."""

import argparse
import json
from pathlib import Path

import numpy as np
import tilelang.language as T
import torch

from kernels.cuda.model.e8p_core import SOURCE, DENSE_SOURCE
from kernels.cuda.operators.op08_gdn_conv_prep import gdn_conv_prep
from orinfer_build.compiler.compile import primitive
from orinfer_build.compiler.cuda.common import SEED, configure, orin_jit
from orinfer_build.quantization.e8p import basis
from orinfer_build.quantization.vq import e8p_decode, e8p_selector_table


def launch(kernel, *args):
    kernel.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)


@orin_jit
def selector_words():
    @T.prim_func
    def main(
        Codes: T.Tensor((65536,), T.uint32),
        Book: T.Tensor((256, 1), T.uint32),
        Out: T.Tensor((65536,), T.uint32),
    ):
        with T.Kernel(256, threads=256) as block:
            T.import_source(SOURCE + DENSE_SOURCE)
            tid = T.get_thread_binding()
            code = Codes[block * 256 + tid]
            b = T.call_pure_extern(
                "uint32", "direct_nibble_basis", Book[T.cast(code >> 8, T.int32), 0]
            )
            Out[block * 256 + tid] = T.call_pure_extern("uint32", "nib_fast", code, b)

    return main


def selector_check():
    table = basis()
    values = e8p_decode(np.arange(65536, dtype=np.uint16), table)
    nibbles = ((values[:, [0, 2, 1, 3, 4, 6, 5, 7]].astype(np.int16) - 1) // 2) & 15
    expected = np.bitwise_or.reduce(
        nibbles.astype(np.uint32) << np.arange(0, 32, 4, dtype=np.uint32), axis=1
    )
    codes = torch.arange(65536, device="cuda", dtype=torch.int64).to(torch.uint32)
    book = torch.from_numpy(e8p_selector_table(table)).cuda()
    output = torch.empty_like(codes)
    kernel = primitive(selector_words)
    launch(kernel, codes, book, output)
    np.testing.assert_array_equal(output.cpu().numpy(), expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(kernel, codes, book, output)
    codes.copy_(torch.arange(65535, -1, -1, device="cuda", dtype=torch.int64).to(torch.uint32))
    graph.replay()
    np.testing.assert_array_equal(output.cpu().numpy(), expected[::-1])
    return dict(codewords=65536, exact=True, changed_input_graph=True)


def convolution_checks():
    results = []
    for heads in (16, 32, 48):
        channels = (32 + heads) * 128
        x = torch.randn((1, 1, channels), device="cuda", dtype=torch.float16)
        w = torch.randn((channels, 4), device="cuda", dtype=torch.float16) * 0.1
        history = torch.randn((1, 3, channels), device="cuda", dtype=torch.float16)
        lengths = torch.ones(1, device="cuda", dtype=torch.int32)
        positions = torch.zeros_like(lengths)
        initial = history.clone()

        def outputs():
            return [
                torch.empty((1, 16, 1, 128), device="cuda", dtype=torch.float16),
                torch.empty((1, 16, 1, 128), device="cuda", dtype=torch.float16),
                torch.empty((1, heads, 1, 128), device="cuda", dtype=torch.float16),
                torch.empty_like(history),
                torch.empty_like(positions),
            ]

        expected, actual = outputs(), outputs()
        reference = primitive(gdn_conv_prep, 1, 1, value_heads=heads)
        compact = primitive(
            gdn_conv_prep, 1, 1, compact_decode=True, inplace_history=True, value_heads=heads
        )

        def reference_run():
            launch(reference, x, w, initial, lengths, positions, *expected)

        def run():
            launch(compact, x, w, history, lengths, positions, *actual[:3], actual[-1])

        def check():
            for left, right in zip(actual[:3], expected[:3]):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            torch.testing.assert_close(history, expected[3], rtol=0, atol=0)
            torch.testing.assert_close(actual[-1], expected[-1], rtol=0, atol=0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for position in (0, 1, 2, 3, 512):
            for length in (0, 1):
                positions.fill_(position)
                lengths.fill_(length)
                x.neg_()
                history.copy_(initial)
                reference_run()
                graph.replay()
                check()
                # The next decode must consume the newly committed raw tail.
                initial.copy_(expected[3])
                positions.add_(length)
                reference_run()
                graph.replay()
                check()
        results.append(dict(heads=heads, exact=True, changed_input_graph=True, sequential=True))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(selectors=selector_check(), convolution=convolution_checks())
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", report, flush=True)


if __name__ == "__main__":
    main()
