"""Numerics, nonaligned lengths, and changed-input graph replay for embeddings."""

import argparse
import json
from pathlib import Path

from kernels.cuda.model import embedding as e
from kernels.cuda.vision.encoder import attention
from orinfer_build.compiler.compile import primitive
from orinfer_build.tuning.row_tuning import launch
from orinfer_build.compiler.cuda.common import configure


def check(kernel, args, output, reference, change, atol=0.003):
    import torch

    launch(kernel, *args)
    torch.testing.assert_close(output, reference(), atol=atol, rtol=0.001)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(kernel, *args)
    change()
    output.fill_(91)
    graph.replay()
    expected = reference()
    torch.testing.assert_close(output, expected, atol=atol, rtol=0.001)
    return float((output - expected).abs().max())


def run():
    import torch
    import torch.nn.functional as F

    configure()
    records = []
    for n in [1, 7, 33, 65]:
        q = torch.randn(n, 3072, device="cuda", dtype=torch.float16) * 0.2
        k = torch.randn(n, 1024, device="cuda", dtype=torch.float16) * 0.2
        v, out = torch.randn_like(k), torch.empty_like(q)
        length = torch.tensor([n], device="cuda", dtype=torch.int32)

        def reference_attention():
            qs = q.view(n, 24, 128).transpose(0, 1)[None]
            ks, vs = [
                x.view(n, 8, 128).transpose(0, 1).repeat_interleave(3, 0)[None] for x in (k, v)
            ]
            return (
                F.scaled_dot_product_attention(qs, ks, vs)
                .squeeze(0)
                .transpose(0, 1)
                .reshape(n, 3072)
            )

        kernel = primitive(attention, 3072, 24, kv_heads=8)
        error = check(kernel, [q, k, v, length, out], out, reference_attention, q.neg_)
        records.append(dict(kind="bidirectional_gqa", rows=n, max_abs=error, graph=True))
    n, h = 33, 2048
    x = torch.randn(n, h, device="cuda", dtype=torch.float16)
    w = torch.randn(h, device="cuda", dtype=torch.float16)
    out = torch.empty_like(x)
    kernel = primitive(e.rms_norm, 32768, h, dynamic_rows=True)

    def reference():
        return (
            x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        ).half() * w

    error = check(kernel, [x, w, out, n], out, reference, x.neg_, 0.01)
    records.append(dict(kind="rms", max_abs=error, graph=True))
    for hidden in [1009, 3073]:
        x = torch.randn(7, hidden * 2, device="cuda", dtype=torch.float16)
        residual = torch.randn(7, hidden, device="cuda", dtype=torch.float16)
        out = torch.empty_like(residual)
        for block_size in [256, 512, 1024]:
            for kind in ["add", "swiglu"]:
                kernel = primitive(
                    e.add if kind == "add" else e.activation,
                    32768,
                    hidden,
                    dynamic_rows=True,
                    block_size=block_size,
                )
                source = x[:, :hidden].contiguous() if kind == "add" else x
                inputs = [source, residual, out, 7] if kind == "add" else [source, out, 7]

                def reference():
                    if kind == "add":
                        return (source.float() + residual.float()).half()
                    return F.silu(source[:, :hidden]).half() * source[:, hidden:]

                check(kernel, inputs, out, reference, source.neg_, 0)
                torch.testing.assert_close(out, reference(), atol=0, rtol=0)
                records.append(
                    dict(
                        kind=kind,
                        hidden=hidden,
                        block_size=block_size,
                        nonaligned=True,
                        bit_exact=True,
                        graph=True,
                    )
                )
    grid = torch.tensor([6, 10], device="cuda", dtype=torch.int32)
    x = torch.randn(60, 1024, device="cuda", dtype=torch.float16)
    out = torch.empty(15, 4096, device="cuda", dtype=torch.float16)

    def reference():
        return F.unfold(x.view(6, 10, 1024).permute(2, 0, 1)[None], 2, stride=2).squeeze(0).t()

    kernel = primitive(e.merge, 32768, dynamic_rows=True)
    error = check(kernel, [x, grid, out, 15], out, reference, x.neg_, 0)
    records.append(dict(kind="merge_unfold", max_abs=error, graph=True))
    from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
    from transformers.models.pixtral.modeling_pixtral import (
        PixtralVisionRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    x = torch.randn(60, 3072, device="cuda", dtype=torch.float16)
    outputs = [torch.empty(60, 1024, device="cuda", dtype=torch.float16) for _ in range(3)]
    ids = (
        torch.stack(torch.meshgrid(torch.arange(6), torch.arange(10), indexing="ij"), -1)
        .reshape(-1, 2)
        .cuda()
    )
    rope = PixtralVisionRotaryEmbedding(
        PixtralVisionConfig(hidden_size=1024, num_attention_heads=16)
    ).cuda()

    def reference_rope():
        cos, sin = rope(x, ids)
        q, k, v = [a.view(60, 16, 64).transpose(0, 1) for a in x.split(1024, -1)]
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)
        return [a.transpose(0, 1).reshape(60, 1024) for a in (q, k, v)]

    kernel = primitive(e.vision_rope, 32768, dynamic_rows=True)
    launch(kernel, x, grid, *outputs, 60)
    for actual, expected in zip(outputs, reference_rope()):
        torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.002)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(kernel, x, grid, *outputs, 60)
    x.neg_()
    graph.replay()
    for actual, expected in zip(outputs, reference_rope()):
        torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.002)
    records.append(dict(kind="pixtral_axial_rope", graph=True))
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
    from transformers.models.ministral3.modeling_ministral3 import (
        Ministral3RotaryEmbedding,
        get_llama_4_attn_scale,
    )
    from orinfer_build.models.nemotron_embed.test_config import config

    cfg = Ministral3Config(**dict(config()["text_config"], max_position_embeddings=262144))
    rotary = Ministral3RotaryEmbedding(cfg).cuda()
    for n in [65, 16401]:
        x = torch.randn(n, 5120, device="cuda", dtype=torch.float16) * 0.1
        positions = torch.arange(n, device="cuda")[None]
        cos, sin = [a[0].contiguous() for a in rotary(x, positions)]
        scale = get_llama_4_attn_scale(positions, 0.1, 16384).reshape(-1).half()
        output = [
            torch.empty(n, width, device="cuda", dtype=torch.float16)
            for width in (3072, 1024, 1024)
        ]

        def expected_text_rope():
            q, k, v = x.split([3072, 1024, 1024], -1)
            q, k = [a.view(n, heads, 128).transpose(0, 1) for a, heads in [(q, 24), (k, 8)]]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)
            return [
                (q * scale[None, :, None]).transpose(0, 1).reshape(n, 3072),
                k.transpose(0, 1).reshape(n, 1024),
                v,
            ]

        ranges = torch.zeros(n, 2, device="cuda", dtype=torch.int32)
        for group in [1, 4, 8]:
            kernel = primitive(e.text_rope, n, dynamic_rows=True, head_group=group)
            launch(kernel, ranges, x, cos, sin, scale, *output, n)
            for actual, expected in zip(output, expected_text_rope()):
                torch.testing.assert_close(actual, expected, atol=0.0005, rtol=0.002)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(kernel, ranges, x, cos, sin, scale, *output, n)
            x.neg_()
            graph.replay()
            for actual, expected in zip(output, expected_text_rope()):
                torch.testing.assert_close(actual, expected, atol=0.0005, rtol=0.002)
            records.append(
                dict(kind="text_yarn_and_attention_scale", rows=n, head_group=group, graph=True)
            )
    x = torch.randn(65, 2048, device="cuda", dtype=torch.float16)
    mean = torch.empty(2048, device="cuda", dtype=torch.float32)
    out = torch.empty_like(mean)
    mean_kernel = primitive(e.pool, 32768, 2048, dynamic_rows=True)
    norm_kernel = primitive(e.normalize, 2048)
    launch(mean_kernel, x, mean, 65)
    launch(norm_kernel, mean, out)
    torch.testing.assert_close(out, F.normalize(x.float().mean(0), dim=-1), atol=1e-6, rtol=1e-5)
    records.append(dict(kind="mean_pool_and_normalize", rows=65))
    return dict(passed=True, checks=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run()
    (a.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS", len(result["checks"]), flush=True)


if __name__ == "__main__":
    main()
