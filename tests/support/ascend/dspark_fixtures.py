"""Focused numerical fixtures for DSpark's shared packet boundary kernels."""

import torch

from kernels.ascend import kv_int8, model_ops, topk, vector
from orinfer_build.compiler.ascend.build import operand

NAMES = (
    "mean_sections",
    "confidence",
    "block_first",
    "reduce_rank_best",
    "scatter_tail",
    "add_bias_argmax",
)


def reference(kind, inputs, p):
    if kind == "add_bias_argmax":
        base, bias, proposals, step_index = inputs
        proposals = proposals.clone()
        scores = base.view(p["rows"], 5, -1)[:, int(step_index[0])] + bias
        proposals.view(p["rows"], 5, -1)[:, int(step_index[0])] = scores
        candidates = []
        for first in range(0, p["vocab"], p["tile"]):
            values, ids = scores[:, first : first + p["tile"]].max(-1)
            candidates.append(torch.stack((values, (ids + first + p["vocab_first"]).float()), -1))
        return [proposals, torch.stack(candidates)]
    if kind == "scatter_tail":
        source, indices, cache, scales = inputs
        cache, scales = cache.clone(), scales.clone()
        errors = ((indices < -1) | (indices >= p["logical_capacity"])).int()
        for row in range(max(0, p["rows"] - p["table_rows"]), p["rows"]):
            if 0 <= indices[row] < p["logical_capacity"]:
                value = source[row].double()
                scale = (value.abs().amax(-1) / 127).clamp(2**-24, 65504).half()
                quantized = (
                    (value / scale.double()[:, None]).round().clamp(-127, 127).to(torch.int8)
                )
                slot = int(indices[row]) % p["table_rows"]
                cache[slot], scales[slot] = quantized, scale
        return [cache, scales, errors]
    if kind == "mean_sections":
        residual, features = inputs
        features = features.clone()
        features[:, p["section"]] = residual.float().sum(1).div(4).bfloat16()
        return [features]
    if kind == "confidence":
        hidden, embedding, weight, probabilities, step_index = inputs
        value = torch.cat(
            (hidden.view(p["rows"], 5, -1)[:, int(step_index[0])], embedding), dim=1
        ).float()
        probabilities = probabilities.clone()
        probabilities.view(-1, 5)[:, int(step_index[0])] = ((value * weight).sum(-1)).sigmoid()
        return [probabilities]
    if kind == "block_first":
        return [inputs[0].view(-1, 5)[:, 0].contiguous()]
    if kind == "reduce_rank_best":
        packed, previous, tokens, values, step_index = inputs
        packed = packed.clone()
        packed[..., 1] += (torch.arange(p["ranks"]) // p["rank_parts"] * p["rank_vocab"])[:, None]
        tokens, values = tokens.clone(), values.clone()
        scores = packed[..., 0].amax(0)
        ids = torch.where(packed[..., 0] == scores, packed[..., 1], torch.inf).amin(0).int()
        tokens.view(-1, 5)[:, int(step_index[0])] = ids
        values.view(-1, 5)[:, int(step_index[0])] = scores
        return [ids, tokens, values]
    raise ValueError(kind)


def cases(rows):
    p = dict(rows=rows, vocab=257, vocab_first=7 * 16160, block_size=5, tile=256)
    base, bias = torch.randn(rows * 5, 257), torch.randn(rows, 257)
    base[3].zero_()
    bias[0].zero_()
    yield (
        "add_bias_argmax",
        topk.add_bias_argmax(**p),
        p,
        [
            operand("base", "f32", (rows * 5, 257)),
            operand("bias", "f32", (rows, 257)),
            operand("proposals", "f32", (rows * 5, 257), "read_write"),
            operand("packed", "f32", (2, rows, 2), "write"),
            operand("step_index", "i32", (1,)),
        ],
        [base, bias, torch.randn(rows * 5, 257), torch.tensor([3], dtype=torch.int32)],
    )
    p = dict(rows=257, table_rows=128, logical_capacity=262144, retain_tail=True)
    yield (
        "scatter_tail",
        kv_int8.scatter(**p),
        p,
        [
            operand("source", "bf16", (257, 8, 64)),
            operand("indices", "i32", (257,)),
            operand("cache", "i8", (128, 8, 64), "read_write"),
            operand("scales", "f16", (128, 8), "read_write"),
            operand("errors", "i32", (257,), "write"),
        ],
        [
            torch.randn(257, 8, 64).bfloat16(),
            torch.arange(257, dtype=torch.int32) + 121,
            torch.randint(-127, 128, (128, 8, 64), dtype=torch.int8),
            torch.rand(128, 8).half(),
        ],
    )
    width = 5120
    for section in (2,):
        p = dict(rows=rows, section=section)
        yield (
            "mean_sections",
            vector.mean_sections(**p),
            p,
            [
                operand("residual", "bf16", (rows, 4, width)),
                operand("features", "bf16", (rows, 3, width), "read_write"),
            ],
            [torch.randn(rows, 4, width).bfloat16(), torch.randn(rows, 3, width).bfloat16()],
        )
    p = dict(rows=rows)
    yield (
        "confidence",
        vector.confidence(**p),
        p,
        [
            operand("hidden", "bf16", (rows * 5, width)),
            operand("embedding", "bf16", (rows, 256)),
            operand("weight", "f32", (width + 256,)),
            operand("probabilities", "f32", (rows * 5,), "read_write"),
            operand("step_index", "i32", (1,)),
        ],
        [
            torch.randn(rows * 5, width).bfloat16(),
            torch.randn(rows, 256).bfloat16(),
            torch.randn(width + 256) * 0.01,
            torch.rand(rows * 5),
            torch.tensor([3], dtype=torch.int32),
        ],
    )
    p = dict(rows=rows)
    yield (
        "block_first",
        model_ops.block_first(**p),
        p,
        [operand("source", "i32", (rows * 5,)), operand("out", "i32", (rows,), "write")],
        [torch.randint(129280, (rows * 5,), dtype=torch.int32)],
    )
    p = dict(rows=rows, ranks=64, rank_parts=8, rank_vocab=16160)
    pairs = torch.stack((torch.randn(64, rows), torch.randint(16160, (64, rows)).float()), dim=-1)
    # Include an exact tie so token ordering is part of every replay check.
    pairs[:, 0, 0] = 1.0
    yield (
        "reduce_rank_best",
        topk.reduce_rank_best(**p),
        p,
        [
            operand("packed", "f32", (64, rows, 2)),
            operand("previous", "i32", (rows,), "read_write"),
            operand("tokens", "i32", (rows * 5,), "read_write"),
            operand("values", "f32", (rows * 5,), "read_write"),
            operand("step_index", "i32", (1,)),
        ],
        [
            pairs,
            torch.zeros(rows, dtype=torch.int32),
            torch.zeros(rows * 5, dtype=torch.int32),
            torch.randn(rows * 5),
            torch.tensor([3], dtype=torch.int32),
        ],
    )
