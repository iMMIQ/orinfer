"""Validate ASR convolution and window attention, including changed-input graphs."""

import argparse
import json
from pathlib import Path

from kernels.cuda.audio import encoder as audio
from kernels.cuda.vision import encoder as common
from orinfer_build.compiler.compile import primitive
from orinfer_build.tuning.row_tuning import launch
from orinfer_build.compiler.cuda.common import SEED, configure


def compare(kernel, args, reference, output, change, tolerance):
    import torch

    launch(kernel, *args)
    torch.testing.assert_close(output, reference(), atol=tolerance, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(kernel, *args)
    change()
    output.fill_(91)
    graph.replay()
    expected = reference()
    torch.testing.assert_close(output, expected, atol=tolerance, rtol=0)
    return float((output - expected).abs().max())


def run():
    import torch

    configure()
    torch.manual_seed(SEED)
    checks = []
    for channels, height, width in [(1, 128, 100), (480, 64, 50), (480, 32, 25)]:
        x = torch.randn(1, height, width, channels, device="cuda", dtype=torch.float16) * 0.1
        weight = torch.randn(480, 3, 3, channels, device="cuda", dtype=torch.float16) * 0.01
        bias = torch.randn(480, device="cuda", dtype=torch.float16) * 0.01

        def reference():
            return (
                torch.nn.functional.gelu(
                    torch.nn.functional.conv2d(
                        x.permute(0, 3, 1, 2), weight.permute(0, 3, 1, 2), bias, stride=2, padding=1
                    )
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )

        output = torch.empty_like(reference())
        kernel = primitive(audio.convolution, channels, height, width, bm=128, bn=128, bk=32)
        error = compare(kernel, [x, weight, bias, output], reference, output, x.neg_, 0.002)
        checks.append(dict(kind="convolution", channels=channels, max_abs=error, graph=True))
    for rows in [1, 103, 104, 105, 211]:
        q = torch.randn(rows, 1024, device="cuda", dtype=torch.float16) * 0.1
        k, v = torch.randn_like(q), torch.randn_like(q)
        output = torch.empty_like(q)
        length = torch.tensor([rows], device="cuda", dtype=torch.int32)

        def reference():
            expected = torch.empty_like(q)
            for first in range(0, rows, 104):
                last = min(first + 104, rows)
                qs, ks, vs = [
                    value[first:last].reshape(-1, 16, 64).transpose(0, 1).unsqueeze(0)
                    for value in (q, k, v)
                ]
                expected[first:last] = (
                    torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
                    .squeeze(0)
                    .transpose(0, 1)
                    .reshape(-1, 1024)
                )
            return expected

        kernel = primitive(common.attention, 1024, 16, window=104)
        error = compare(kernel, [q, k, v, length, output], reference, output, q.neg_, 0.004)
        checks.append(dict(kind="window_attention", rows=rows, max_abs=error, graph=True))
    # Independent recordings occupy whole windows with distinct random K/V;
    # changing valid lengths checks tail masking during graph replay.
    rows = 222
    q = torch.randn(rows, 1024, device="cuda", dtype=torch.float16) * 0.1
    k, v, output = torch.randn_like(q), torch.randn_like(q), torch.empty_like(q)
    length = torch.tensor([104, 1, 14], device="cuda", dtype=torch.int32)

    def reference():
        expected = torch.zeros_like(q)
        for segment, count in enumerate(length.tolist()):
            first, last = segment * 104, segment * 104 + count
            qs, ks, vs = [
                value[first:last].reshape(-1, 16, 64).transpose(0, 1).unsqueeze(0)
                for value in (q, k, v)
            ]
            expected[first:last] = (
                torch.nn.functional.scaled_dot_product_attention(qs, ks, vs)
                .squeeze(0)
                .transpose(0, 1)
                .reshape(-1, 1024)
            )
        return expected

    def change():
        q.neg_()
        length.copy_(torch.tensor([104, 7, 6], device="cuda", dtype=torch.int32))

    kernel = primitive(common.attention, 1024, 16, window=104, windowed_lengths=True)
    error = compare(kernel, [q, k, v, length, output], reference, output, change, 0.004)
    checks.append(dict(kind="packed_window_attention", rows=rows, max_abs=error, graph=True))
    return dict(passed=True, checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = run()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", len(report["checks"]), flush=True)


if __name__ == "__main__":
    main()
