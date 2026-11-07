"""Check independent encoder owners, padding, outputs and changed-boundary Graph replay."""

import argparse
import json
from pathlib import Path

from kernels.cuda.model import encoder_rows
from kernels.cuda.vision.encoder import attention
from orinfer_build.compiler.compile import primitive
from orinfer_build.tuning.row_tuning import launch
from orinfer_build.compiler.cuda.common import configure


def run():
    import torch
    import torch.nn.functional as F

    configure()
    torch.manual_seed(20261002)
    device = dict(device="cuda")
    records = []
    layouts = [(True, 16, 1), (True, 16, 2), (False, 24, 1), (False, 24, 3)]
    for causal, heads, group, bn, stages, base2, full_row in [
        (*layout, *tile)
        for layout in layouts
        for tile in [(32, 0, False, False), (32, 0, False, True), (64, 1, True, True)]
    ]:
        rows = 224
        q = torch.randn(rows, heads * 128, **device).half() * 0.2
        k = torch.randn(rows, 1024, **device).half() * 0.2
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        ranges = torch.zeros(rows, 2, dtype=torch.int32, **device)
        kernel = primitive(
            attention,
            heads * 128,
            heads,
            bn=bn,
            head_group=group,
            stages=stages,
            base2=base2,
            full_row=full_row,
            kv_heads=8,
            causal=causal,
            segmented=True,
        )

        def boundaries(lengths):
            ranges.zero_()
            spans = []
            start = 0
            for n in lengths:
                padded = (n + 31) // 32 * 32
                ranges[start : start + padded] = torch.tensor([start, start + n], **device)
                spans.append((start, start + n))
                start += padded
            assert start == rows
            return spans

        spans = boundaries([1, 33, 65, 31])
        launch(kernel, q, k, v, ranges, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(kernel, q, k, v, ranges, out)
        for lengths in [[1, 33, 65, 31], [31, 65, 33, 1], [97, 65], [224]]:
            spans = boundaries(lengths)
            q.neg_()
            out.fill_(91)
            graph.replay()
            for start, end in spans:
                n = end - start
                qs = q[start:end].view(n, heads, 128).transpose(0, 1)[None]
                ks, vs = [
                    x[start:end]
                    .view(n, 8, 128)
                    .transpose(0, 1)
                    .repeat_interleave(heads // 8, 0)[None]
                    for x in (k, v)
                ]
                expected = (
                    F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal)[0]
                    .transpose(0, 1)
                    .reshape(n, heads * 128)
                )
                torch.testing.assert_close(out[start:end], expected, atol=0.003, rtol=0.003)
            padding = torch.ones(rows, device="cuda", dtype=torch.bool)
            for start, end in spans:
                padding[start:end] = False
            assert bool(torch.all(out[padding] == 0))
            records.append(
                dict(
                    causal=causal,
                    heads=heads,
                    head_group=group,
                    bn=bn,
                    stages=stages,
                    base2=base2,
                    full_row=full_row,
                    lengths=lengths,
                    graph=True,
                    padding_zero=True,
                )
            )
        if causal:
            saved = out[:-1].clone()
            k[-1].neg_()
            v[-1].add_(10)
            graph.replay()
            torch.testing.assert_close(out[:-1], saved, atol=0, rtol=0)
        # Changed neighboring owners and poisoned padding cannot affect owner zero.
        spans = boundaries([1, 33, 65, 31])
        graph.replay()
        saved = out[0].clone()
        k[1:].fill_(100)
        v[1:].fill_(-100)
        graph.replay()
        torch.testing.assert_close(out[0], saved, atol=0, rtol=0)

    segments = torch.zeros(128, 2, dtype=torch.int32, **device)
    count = torch.zeros(1, dtype=torch.int32, **device)
    x = torch.randn(224, 2048, **device).half()
    mean = torch.empty(128, 2048, **device)
    output = torch.empty_like(mean)
    weight = torch.randn(2, 1024, **device).half()
    logits = torch.empty(128, 2, **device)
    pool = primitive(encoder_rows.pool, 2048)
    normalize = primitive(encoder_rows.normalize, 2048)
    classifier = primitive(encoder_rows.classifier)
    head = x[:, :1024].contiguous()
    segments[:4] = torch.tensor([(0, 1), (32, 65), (96, 161), (192, 223)], **device)
    count.fill_(4)
    launch(pool, x, segments, count, mean)
    launch(normalize, mean, count, output)
    launch(classifier, head, weight, segments, count, logits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(pool, x, segments, count, mean)
        launch(normalize, mean, count, output)
        launch(classifier, head, weight, segments, count, logits)
    for spans in [[(0, 1), (32, 65), (96, 161), (192, 223)], [(0, 97), (128, 193)], [(0, 224)]]:
        count.fill_(len(spans))
        segments[: len(spans)] = torch.tensor(spans, **device)
        x.neg_()
        head.copy_(x[:, :1024])
        graph.replay()
        for owner, (start, end) in enumerate(spans):
            expected = F.normalize(x[start:end].float().mean(0), dim=0)
            torch.testing.assert_close(output[owner], expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(
                logits[owner], head[end - 1].float() @ weight.float().T, atol=1e-4, rtol=1e-5
            )
        records.append(dict(outputs=True, spans=spans, graph=True))
    return dict(passed=True, checks=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = run()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", len(report["checks"]), flush=True)


if __name__ == "__main__":
    main()
