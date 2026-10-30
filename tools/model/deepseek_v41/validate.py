"""Independent references for the model-level A2 precision/layout boundaries."""

import torch

from tools.operators.ascend import dense, indexer, memory, model_ops, moe, vector
from tools.operators.ascend import validate as common
from tools.operators.ascend.build import operand

BASE_REFERENCE = common.reference


def reference(kind, inputs, parameters):
    if kind == "model_norm":
        x, weight = inputs
        a = x.float()
        y = a * (a.square().mean(-1, keepdim=True) + 1e-20).rsqrt() * weight
        return [y.bfloat16(), y]
    if kind == "query_quant":
        x = inputs[0].float()
        scale = (x.abs().amax(-1) / 127).half().float().clamp_min(2**-24)
        return [(x / scale[:, None]).round().clamp(-127, 127).to(torch.int8), scale.half()]
    if kind == "convert_scale":
        return [inputs[0].half()]
    if kind.startswith("partial_rope"):
        x, co, si = inputs
        result = x.clone()
        result[..., -64::2] = (
            x[..., -64::2].float() * co[:, None] - x[..., -63::2].float() * si[:, None]
        ).bfloat16()
        result[..., -63::2] = (
            x[..., -63::2].float() * co[:, None] + x[..., -64::2].float() * si[:, None]
        ).bfloat16()
        return [result]
    if kind == "split":
        return [inputs[0][:, :128].contiguous(), inputs[0][:, 128:].contiguous()]
    if kind == "packed_swiglu":
        return BASE_REFERENCE("swiglu", [inputs[0][:, :128], inputs[0][:, 128:]], {"limit": 10.0})
    if kind in ("scatter_i8", "scatter_f16"):
        return BASE_REFERENCE("scatter_rows", inputs, parameters)
    if kind == "gather_attention":
        swa, long_kv, swa_ids, long_ids = inputs
        ids = torch.cat((swa_ids, long_ids), dim=1)
        output = torch.zeros(ids.shape[0], 640, 512, dtype=torch.bfloat16)
        mask = torch.full(ids.shape, -1, dtype=torch.int32)
        for row in range(ids.shape[0]):
            for slot in range(640):
                table = swa if slot < 128 else long_kv
                index = int(ids[row, slot])
                if 0 <= index < table.shape[0]:
                    output[row, slot] = table[index]
                    mask[row, slot] = 0
        return [output, mask]
    if kind == "index_quantized":
        query, qs, keys, ks, weights, ids = inputs
        valid = (ids >= 0) & (ids < keys.shape[0])
        gathered = keys[ids.clamp(0, keys.shape[0] - 1).long()].float()
        dot = torch.einsum("bhd,bsd->bhs", query.float(), gathered).clamp_min(0)
        dot *= qs.float()[:, :, None]
        dot *= ks[ids.clamp(0, keys.shape[0] - 1).long()].float()[:, None, :]
        # The primitive applies one combined head/dimension scale before FP16.
        # Two FP32 multiplications can move subnormal halfway values across a tie.
        w = (weights.float() * (128**-0.5 * 32**-0.5)).half().float()
        total = torch.zeros(ids.shape)
        for head in range(32):
            total += dot[:, head] * w[:, head, None]
        return [total.masked_fill(~valid, -torch.inf)]
    if kind == "fp32_activation":
        return BASE_REFERENCE("fp32_partials", inputs, parameters)
    if kind == "combine_fp32":
        experts, shared, indices, weights = inputs
        out = torch.zeros_like(shared, dtype=torch.float32)
        errors = torch.zeros(shared.shape[0], dtype=torch.int32)
        for row in range(shared.shape[0]):
            for route in range(6):
                slot = int(indices[row, route])
                if slot >= 0:
                    out[row] += experts[slot].float() * weights[row, route]
        return [out + shared.float(), errors]
    if kind == "add_shared":
        return [(inputs[0].bfloat16().float() + inputs[1].float()).bfloat16()]
    raise ValueError(kind)


def cases(rows, width, index_length=35, *_):
    yield (
        "model_norm",
        model_ops.norm(rows, width),
        {"rows": rows, "width": width},
        [
            operand("x", "bf16", (rows, width)),
            operand("weight", "f32", (width,)),
            operand("out", "bf16", (rows, width), "write"),
            operand("precise", "f32", (rows, width), "write"),
        ],
        [torch.randn(rows, width).bfloat16(), torch.randn(width)],
    )
    yield (
        "query_quant",
        vector.quantize(rows * 32, 128, True),
        {"rows": rows * 32, "width": 128},
        [
            operand("x", "bf16", (rows * 32, 128)),
            operand("out", "i8", (rows * 32, 128), "write"),
            operand("scales", "f16", (rows * 32,), "write"),
        ],
        [torch.randn(rows * 32, 128).bfloat16()],
    )
    yield (
        "convert_scale",
        model_ops.convert(rows, 1, "float32", "float16"),
        {"rows": rows},
        [operand("x", "f32", (rows, 1)), operand("out", "f16", (rows, 1), "write")],
        [torch.rand(rows, 1)],
    )
    for dimension in (128, 512):
        angles = torch.randn(rows, 32) * 1000
        yield (
            f"partial_rope{dimension}",
            model_ops.partial_rope(rows, 8, dimension),
            {"rows": rows},
            [
                operand("x", "bf16", (rows, 8, dimension)),
                operand("cosine", "f32", (rows, 32)),
                operand("sine", "f32", (rows, 32)),
                operand("out", "bf16", (rows, 8, dimension), "write"),
            ],
            [torch.randn(rows, 8, dimension).bfloat16(), angles.cos(), angles.sin()],
        )
    for name, kernel in [
        ("split", model_ops.split(rows, 128, 256)),
        ("packed_swiglu", model_ops.swiglu_packed(rows, 128)),
    ]:
        if name == "split":
            specs = [
                operand("packed", "bf16", (rows, 384)),
                operand("first", "bf16", (rows, 128), "write"),
                operand("second", "bf16", (rows, 256), "write"),
            ]
            data = [torch.randn(rows, 384).bfloat16()]
        else:
            specs = [
                operand("packed", "bf16", (rows, 256)),
                operand("out", "bf16", (rows, 128), "write"),
            ]
            data = [(torch.randn(rows, 256) * 8).bfloat16()]
        yield name, kernel, {"rows": rows}, specs, data
    for name, dtype, kernel_dtype, size in [
        ("scatter_i8", torch.int8, "int8", 128),
        ("scatter_f16", torch.float16, "float16", 1),
    ]:
        values = (
            torch.randint(-127, 127, (rows, size), dtype=torch.int8)
            if dtype == torch.int8
            else torch.rand(rows, size).half()
        )
        table = torch.zeros(rows + 4, size, dtype=dtype)
        ids = torch.arange(rows, dtype=torch.int32) + 1
        if rows > 1:
            ids[-1] = -1
        abi = "i8" if dtype == torch.int8 else "f16"
        yield (
            name,
            memory.scatter_rows(rows, rows + 4, size, kernel_dtype, "int32"),
            {"rows": rows},
            [
                operand("values", abi, (rows, size)),
                operand("indices", "i32", (rows,)),
                operand("table", abi, (rows + 4, size), "read_write"),
                operand("errors", "i32", (rows,), "write"),
            ],
            [values, ids, table],
        )
    sid = torch.randint(-1, 128, (rows, 128), dtype=torch.int32)
    lid = torch.randint(-1, 64, (rows, 512), dtype=torch.int32)
    yield (
        "gather_attention",
        model_ops.gather_attention(rows, 128, 64),
        {"rows": rows},
        [
            operand("swa", "bf16", (128, 512)),
            operand("long_kv", "bf16", (64, 512)),
            operand("swa_indices", "i32", (rows, 128)),
            operand("long_indices", "i32", (rows, 512)),
            operand("out", "bf16", (rows, 640, 512), "write"),
            operand("mask", "i32", (rows, 640), "write"),
        ],
        [torch.randn(128, 512).bfloat16(), torch.randn(64, 512).bfloat16(), sid, lid],
    )
    length, table_rows = index_length, index_length
    ids = torch.randint(-1, table_rows + 1, (rows, length), dtype=torch.int32)
    yield (
        "index_quantized",
        indexer.quantized_logits(rows, length, table_rows),
        {"rows": rows},
        [
            operand("query", "i8", (rows, 32, 128)),
            operand("query_scale", "f16", (rows, 32)),
            operand("keys", "i8", (table_rows, 128)),
            operand("key_scale", "f16", (table_rows,)),
            operand("weights", "bf16", (rows, 32)),
            operand("indices", "i32", (rows, length)),
            operand("out", "f32", (rows, length), "write"),
        ],
        [
            torch.randint(-127, 127, (rows, 32, 128), dtype=torch.int8),
            (torch.rand(rows, 32) / 127).half(),
            torch.randint(-127, 127, (table_rows, 128), dtype=torch.int8),
            (torch.rand(table_rows) / 127).half(),
            torch.randn(rows, 32).bfloat16(),
            ids,
        ],
    )
    yield (
        "fp32_activation",
        dense.fp32_partials(rows, 64, width, activation_dtype="float32"),
        {"rows": rows, "splits": 8},
        [
            operand("activation", "f32", (rows, width)),
            operand("weight", "f32", (64, width)),
            operand("out", "f32", (rows, 2, 8, 32), "write"),
        ],
        [torch.randn(rows, width), torch.randn(64, width)],
    )
    row_map = torch.full((rows, 8), -1, dtype=torch.int32)
    row_map[:, :6] = torch.arange(rows * 6).view(rows, 6).int()
    yield (
        "combine_fp32",
        moe.combine(rows, rows * 6, width, output_dtype="float32"),
        {"rows": rows},
        [
            operand("experts", "bf16", (rows * 6, width)),
            operand("shared", "bf16", (rows, width)),
            operand("row_map", "i32", (rows, 8)),
            operand("weights", "f32", (rows, 8)),
            operand("out", "f32", (rows, width), "write"),
            operand("errors", "i32", (rows,), "write"),
        ],
        [
            torch.randn(rows * 6, width).bfloat16(),
            torch.randn(rows, width).bfloat16(),
            row_map,
            torch.rand(rows, 8),
        ],
    )
    yield (
        "add_shared",
        model_ops.add_shared(rows, width),
        {"rows": rows},
        [
            operand("routed", "f32", (rows, width)),
            operand("shared", "bf16", (rows, width)),
            operand("out", "bf16", (rows, width), "write"),
        ],
        [torch.randn(rows, width), torch.randn(rows, width).bfloat16()],
    )


if __name__ == "__main__":
    common.cases = cases
    common.reference = reference
    common.EXACT_OPS.update(
        {"query_quant", "convert_scale", "split", "scatter_i8", "scatter_f16", "gather_attention"}
    )
    common.main()
