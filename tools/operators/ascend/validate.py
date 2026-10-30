"""Offline reference checks and byte fixtures for the Rust A2 kernel validator."""

import argparse
import ctypes
import json
from pathlib import Path

import torch

from tools.build.files import atomic_json, file_hash
from tools.operators.ascend.build import compile_kernel, operand
from tools.operators.ascend import (
    attention,
    compressor,
    dense,
    engram,
    indexer,
    memory,
    moe,
    routing,
    vector,
)
from tools.operators.ascend import topk as select

EXACT_OPS = {
    "mhc_post_pre",
    "gather_quantized_rows",
    "gather_columns",
    "mhc_expand",
    "gather_rows",
    "scatter_rows",
    "gather_activation",
    "gather_scales",
    "c2_commit",
    "select_local",
    "select_merge",
    "select_finalize",
    "regroup",
}


def reference(kind, inputs, parameters):
    if kind == "mhc_post_pre":
        layer, residual, post, combination, mix = inputs
        updated = reference("mhc_post", [layer, residual, post, combination], parameters)[0]
        collapsed = reference("mhc_pre_apply", [updated, mix], parameters)[0]
        return [updated, collapsed]
    if kind == "gather_columns":
        source = inputs[0]
        width = parameters.get("output_width", 0) or parameters["width"]
        logical, physical = width // source.shape[0], source.shape[-1]
        return [
            torch.cat(
                [
                    source[
                        rank,
                        :,
                        max(0, rank * logical + physical - width) : max(
                            0, rank * logical + physical - width
                        )
                        + logical,
                    ]
                    for rank in range(source.shape[0])
                ],
                dim=1,
            ).contiguous()
        ]
    if kind == "gather_quantized_rows":
        table, scales, indices = inputs
        valid = (indices >= 0) & (indices < table.shape[0])
        out = torch.zeros(indices.numel(), table.shape[1], dtype=table.dtype)
        out_scale = torch.zeros(indices.numel(), dtype=scales.dtype)
        out[valid], out_scale[valid] = table[indices[valid]], scales[indices[valid]]
        return [out, out_scale, ((indices < -1) | (indices >= table.shape[0])).int()]
    if kind == "rms_norm":
        x, w = inputs
        a = x.float()
        return [
            (
                a * (a.square().mean(-1, keepdim=True) + parameters["eps"]).rsqrt() * w.float()
            ).bfloat16()
        ]
    if kind == "quantize":
        x = inputs[0].float()
        scale = x.abs().amax(-1).clamp_min(1e-12) / 127.0
        return [(x / scale[:, None]).round().clamp(-127, 127).to(torch.int8), scale]
    if kind == "swiglu":
        g = inputs[0].float().clamp(max=parameters["limit"])
        u = inputs[1].float().clamp(-parameters["limit"], parameters["limit"])
        return [(g / (1.0 + (-g).exp()) * u).bfloat16()]
    if kind == "mhc_split_sinkhorn":
        mix, scale, base = inputs
        eps = parameters["eps"]
        scaled = (
            mix[:, :24] * torch.cat((scale[0].expand(4), scale[1].expand(4), scale[2].expand(16)))
            + base[:24]
        )
        pre = torch.zeros(mix.shape[0], 8)
        post = torch.zeros_like(pre)
        pre[:, :4] = scaled[:, :4].sigmoid() + eps
        post[:, :4] = scaled[:, 4:8].sigmoid() * 2.0
        a = scaled[:, 8:24].view(-1, 4, 4).softmax(-1) + eps
        a = a / (a.sum(-2, keepdim=True) + eps)
        for _ in range(parameters["iterations"] - 1):
            a = a / (a.sum(-1, keepdim=True) + eps)
            a = a / (a.sum(-2, keepdim=True) + eps)
        return [pre, post, a.flatten(1)]
    if kind == "mhc_expand":
        return [inputs[0][:, None, :].expand(-1, 4, -1).contiguous()]
    if kind == "mhc_pre_apply":
        residual, mix = inputs
        x = residual.float() * mix[:, :4, None]
        return [(((x[:, 0] + x[:, 1]) + x[:, 2]) + x[:, 3]).bfloat16()]
    if kind == "mhc_post":
        layer, residual, post, combination = inputs
        c = combination.view(-1, 4, 4)
        term = residual[:, 0].float()[:, None, :] * c[:, 0, :, None]
        for path in range(1, 4):
            term = term + residual[:, path].float()[:, None, :] * c[:, path, :, None]
        return [(layer.float()[:, None, :] * post[:, :4, None] + term).bfloat16()]
    if kind == "mhc_prenorm_partials":
        residual, fn = inputs
        splits = parameters["splits"]
        chunk = residual.shape[1] // splits
        out = torch.zeros(residual.shape[0], splits, 32)
        for split in range(splits):
            # Independent high-precision oracle avoids accepting/rejecting a
            # vector reduction based on the CPU BLAS FP32 accumulation order.
            a = residual[:, split * chunk : (split + 1) * chunk].double()
            w = fn[:24, split * chunk : (split + 1) * chunk].double()
            out[:, split, :24] = a @ w.T
            out[:, split, 24] = a.square().sum(-1)
        return [out]
    if kind == "mhc_prenorm_reduce":
        total = inputs[0].sum(1)
        inverse = (total[:, 24] / (4 * parameters["width"]) + parameters["eps"]).rsqrt()
        out = torch.zeros(total.shape[0], 32)
        out[:, :24] = total[:, :24] * inverse[:, None]
        return [out]
    if kind in ("gather_rows", "gather_activation", "gather_scales"):
        table, indices = inputs
        valid = (indices >= 0) & (indices < table.shape[0])
        out = torch.zeros(indices.numel(), table.shape[1], dtype=table.dtype)
        out[valid] = table[indices[valid]]
        return [out, ((indices < -1) | (indices >= table.shape[0])).int()]
    if kind == "scatter_rows":
        values, indices, original = inputs
        out = original.clone()
        valid = (indices >= 0) & (indices < original.shape[0])
        out[indices[valid]] = values[valid]
        return [out, ((indices < -1) | (indices >= original.shape[0])).int()]
    if kind == "dequant_rows":
        values, scales = inputs
        return [(values.float() * scales.repeat_interleave(parameters["group"], dim=1)).bfloat16()]
    if kind == "rope":
        x, cosine, sine = inputs
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        c, s = cosine[:, None, :], sine[:, None, :]
        out = torch.empty_like(x)
        out[..., 0::2] = (even * c - odd * s).bfloat16()
        out[..., 1::2] = (odd * c + even * s).bfloat16()
        return [out]
    if kind == "route":
        logits, bias = inputs
        score = torch.nn.functional.softplus(logits).sqrt()
        chosen = (score + bias).argsort(dim=-1, descending=True, stable=True)[
            :, : parameters["selected"]
        ]
        values = score.gather(1, chosen)
        weights = torch.zeros(logits.shape[0], 8)
        indices = torch.full((logits.shape[0], 8), -1, dtype=torch.int32)
        indices[:, : parameters["selected"]] = chosen.int()
        weights[:, : parameters["selected"]] = (
            values / values.sum(-1, keepdim=True) * parameters["routed_scale"]
        )
        return [indices, weights]
    if kind == "regroup":
        indices, original_errors = inputs
        rows, selected = parameters["rows"], parameters["selected"]
        gather = torch.full((rows * selected,), -1, dtype=torch.int32)
        restore = torch.full_like(indices, -1)
        counts = torch.zeros(parameters["local"], dtype=torch.int64)
        position = 0
        for expert in range(parameters["local"]):
            for row in range(rows):
                for lane in range(selected):
                    if indices[row, lane] == parameters["base"] + expert:
                        gather[position] = row
                        restore[row, lane] = position
                        counts[expert] += 1
                        position += 1
        if parameters["pad_last"]:
            counts[-1] += gather.numel() - position
        errors = original_errors.clone()
        errors[0] = (
            (indices[:, :selected] < 0) | (indices[:, :selected] >= parameters["experts"])
        ).any()
        errors[rows] = position
        return [gather, restore, counts, errors]
    if kind in ("w8_projection", "bf16_projection"):
        x, w, sa, sw = inputs
        raw = (
            (x.long() @ w.long().T).float() if kind == "w8_projection" else x.float() @ w.float().T
        )
        return [(raw * sa[:, None] * sw[None, :]).bfloat16()]
    if kind == "rotate_blocks":
        hidden, rotation = inputs
        shape = hidden.shape
        return [
            (hidden.float().view(*shape[:-1], -1, parameters["block"]) @ rotation.T).reshape(shape)
        ]
    if kind == "engram_gate":
        hidden, original, key, value, channel_weight, token_mask = inputs
        k = key.float()
        inverse = (original.square().mean(-1) + parameters["eps"]).rsqrt()
        inverse *= (k.square().mean(-1) + parameters["eps"]).rsqrt()
        dot = (original * channel_weight * k).sum(-1) * inverse * parameters["width"] ** -0.5
        magnitude = dot.abs().clamp_min(1e-6).sqrt()
        gate = torch.where(torch.signbit(dot), -magnitude, magnitude).sigmoid()
        gate = gate.masked_fill(~token_mask.bool()[:, None], 0)
        return [(hidden.float() + gate[:, :, None] * value.float()[:, None, :]).bfloat16()]
    if kind == "sparse_attention":
        q, kv, indices, sink = inputs
        score = torch.einsum("bhd,bsd->bhs", q.float(), kv.float()) * parameters["scale"]
        score = score.masked_fill(indices[:, None, :] < 0, -torch.inf)
        with_sink = torch.cat((score, sink[None, :, None].expand(q.shape[0], -1, -1)), dim=-1)
        prob = with_sink.softmax(-1)[..., :-1]
        return [torch.einsum("bhs,bsd->bhd", prob, kv.float()).bfloat16()]
    if kind == "fp32_partials":
        x, w = inputs
        splits = parameters["splits"]
        chunk = x.shape[1] // splits
        out = torch.empty(x.shape[0], w.shape[0] // 32, splits, 32)
        for split in range(splits):
            part = (
                x[:, split * chunk : (split + 1) * chunk].double()
                @ w[:, split * chunk : (split + 1) * chunk].double().T
            )
            out[:, :, split, :] = part.view(x.shape[0], -1, 32)
        return [out]
    if kind == "fp32_reduce":
        return [inputs[0].sum(2).flatten(1)]
    if kind in ("c2_pool", "c2_commit"):
        width = parameters["width"]
        if kind == "c2_pool":
            kv, scores, ring, metadata = inputs
        else:
            kv, scores, metadata, ring = inputs
        errors = torch.zeros(kv.shape[0], dtype=torch.int32)
        out = torch.zeros_like(kv, dtype=torch.bfloat16) if kind == "c2_pool" else ring.clone()
        for row, data in enumerate(metadata.tolist()):
            position, page, previous, active, last, commit, _, _ = data
            if not active or (kind == "c2_commit" and not commit):
                continue
            if (
                position < 0
                or not 0 < page < ring.shape[0]
                or (kind == "c2_commit" and last < position)
            ):
                errors[row] = 1
                continue
            if kind == "c2_commit":
                if last - position < 32:
                    out[page, position % 32] = torch.cat((kv[row], scores[row]))
                continue
            if previous < -1 or previous >= kv.shape[0]:
                errors[row] = 1
                continue
            if position % 2 == 0:
                continue
            if previous >= 0:
                pm = metadata[previous].tolist()
                if pm[0] != position - 1 or pm[1] != page or not pm[3]:
                    errors[row] = 1
                    continue
                old_k, old_s = kv[previous], scores[previous]
            else:
                old_k, old_s = ring[page, (position - 1) % 32].split(width)
            probabilities = torch.stack((old_s, scores[row])).softmax(0)
            out[row] = (torch.stack((old_k, kv[row])) * probabilities).sum(0).bfloat16()
        return [out, errors]
    if kind == "select_local":
        scores, lengths = inputs
        block, selected = parameters["block"], parameters["selected"]
        chunks = (scores.shape[1] + block - 1) // block
        out = torch.full((scores.shape[0], chunks, 2 * selected), -torch.inf)
        out[..., 1::2] = -1
        for row in range(scores.shape[0]):
            for chunk in range(chunks):
                start = chunk * block
                values = scores[row, start : min(start + block, int(lengths[row]))]
                chosen = values.argsort(descending=True, stable=True)[:selected]
                out[row, chunk, : 2 * chosen.numel() : 2] = values[chosen]
                out[row, chunk, 1 : 2 * chosen.numel() : 2] = chosen.float() + start
        return [out]
    if kind == "select_merge":
        candidates = inputs[0]
        selected = parameters["selected"]
        out = torch.empty(candidates.shape[0], (candidates.shape[1] + 3) // 4, 2 * selected)
        for row in range(candidates.shape[0]):
            for group in range(out.shape[1]):
                pairs = candidates[row, group * 4 : (group + 1) * 4].reshape(-1, 2)
                chosen = pairs[:, 0].argsort(descending=True, stable=True)[:selected]
                out[row, group] = pairs[chosen].flatten()
        return [out]
    if kind == "select_finalize":
        pairs = inputs[0]
        indices = pairs[:, 1::2].int()
        indices[pairs[:, 0::2] == -torch.inf] = -1
        return [indices, pairs[:, 0::2].contiguous()]
    if kind == "mhc_head_mix":
        mixes, scale, base = inputs
        out = (mixes * scale[0] + base).sigmoid() + parameters["eps"]
        out[:, 4:] = 0
        return [out]
    if kind == "moe_combine":
        experts, shared, row_map, weights = inputs
        out = torch.zeros_like(shared, dtype=torch.float32)
        errors = torch.zeros(shared.shape[0], dtype=torch.int32)
        for row in range(shared.shape[0]):
            for route in range(parameters["selected"]):
                index = int(row_map[row, route])
                if 0 <= index < experts.shape[0]:
                    out[row] += experts[index].float() * weights[row, route]
                elif index != -1:
                    errors[row] = 1
        return [(out + shared.float()).bfloat16(), errors]
    if kind == "indexer_logits":
        q, keys, weights, indices = inputs
        valid = (indices >= 0) & (indices < keys.shape[0])
        safe = indices.clamp(0, keys.shape[0] - 1).long()
        selected_keys = keys[safe].float()
        dots = torch.einsum("bhd,bnd->bhn", q.float(), selected_keys) * parameters["scale"]
        out = (dots.relu() * weights[:, : q.shape[1], None]).sum(1)
        out = out.masked_fill(~valid, -torch.inf)
        errors = (
            ((indices < -1) | (indices >= keys.shape[0]))
            .view(indices.shape[0], -1, 16)
            .any(-1)
            .int()
        )
        return [out, errors]
    raise ValueError(kind)


def cases(rows, width, index_length=8192, attention_selected=32, select_k=512):
    eps = 1e-20
    yield (
        "rms_norm",
        vector.rms_norm(rows, width, eps),
        {"rows": rows, "width": width, "eps": eps},
        [
            operand("x", "bf16", (rows, width)),
            operand("weight", "bf16", (width,)),
            operand("out", "bf16", (rows, width), "write"),
        ],
        [torch.randn(rows, width).bfloat16(), torch.randn(width).bfloat16()],
    )
    yield (
        "quantize",
        vector.quantize(rows, width),
        {"rows": rows, "width": width},
        [
            operand("x", "bf16", (rows, width)),
            operand("out", "i8", (rows, width), "write"),
            operand("scales", "f32", (rows,), "write"),
        ],
        [torch.randn(rows, width).bfloat16()],
    )
    yield (
        "swiglu",
        vector.swiglu(rows, width),
        {"rows": rows, "width": width, "limit": 10.0},
        [
            operand("gate", "bf16", (rows, width)),
            operand("up", "bf16", (rows, width)),
            operand("out", "bf16", (rows, width), "write"),
        ],
        [(torch.randn(rows, width) * 12).bfloat16(), (torch.randn(rows, width) * 12).bfloat16()],
    )
    yield (
        "mhc_split_sinkhorn",
        vector.mhc_split_sinkhorn(rows),
        {"rows": rows, "iterations": 20, "eps": 1e-6},
        [
            operand("mixes", "f32", (rows, 32)),
            operand("scale", "f32", (8,)),
            operand("base", "f32", (32,)),
            operand("pre", "f32", (rows, 8), "write"),
            operand("post", "f32", (rows, 8), "write"),
            operand("combination", "f32", (rows, 16), "write"),
        ],
        [torch.randn(rows, 32) * 4, torch.randn(8), torch.randn(32)],
    )
    yield (
        "mhc_expand",
        vector.mhc_expand(rows, width),
        {"rows": rows, "width": width},
        [operand("x", "bf16", (rows, width)), operand("out", "bf16", (rows, 4, width), "write")],
        [torch.randn(rows, width).bfloat16()],
    )
    yield (
        "mhc_pre_apply",
        vector.mhc_pre_apply(rows, width),
        {"rows": rows, "width": width},
        [
            operand("residual", "bf16", (rows, 4, width)),
            operand("mix", "f32", (rows, 8)),
            operand("out", "bf16", (rows, width), "write"),
        ],
        [torch.randn(rows, 4, width).bfloat16(), torch.rand(rows, 8)],
    )
    yield (
        "mhc_post",
        vector.mhc_post(rows, width),
        {"rows": rows, "width": width},
        [
            operand("layer", "bf16", (rows, width)),
            operand("residual", "bf16", (rows, 4, width)),
            operand("post", "f32", (rows, 8)),
            operand("combination", "f32", (rows, 16)),
            operand("out", "bf16", (rows, 4, width), "write"),
        ],
        [
            torch.randn(rows, width).bfloat16(),
            torch.randn(rows, 4, width).bfloat16(),
            torch.rand(rows, 8),
            torch.rand(rows, 16),
        ],
    )
    fn = torch.randn(32, 4 * width)
    fn[24:] = 0
    yield (
        "mhc_prenorm_partials",
        vector.mhc_prenorm_partials(rows, width),
        {"rows": rows, "width": width, "splits": 8},
        [
            operand("residual", "bf16", (rows, 4 * width)),
            operand("fn", "f32", (32, 4 * width)),
            operand("out", "f32", (rows, 8, 32), "write"),
        ],
        [torch.randn(rows, 4 * width).bfloat16(), fn],
    )
    partials = torch.randn(rows, 8, 32)
    partials[:, :, 24] = torch.rand(rows, 8) + 1
    partials[:, :, 25:] = 0
    yield (
        "mhc_prenorm_reduce",
        vector.mhc_prenorm_reduce(rows, width),
        {"rows": rows, "width": width, "splits": 8, "eps": eps},
        [operand("partials", "f32", (rows, 8, 32)), operand("out", "f32", (rows, 32), "write")],
        [partials],
    )
    table_rows = max(17, rows * 2 + 1)
    ids = torch.arange(rows, dtype=torch.int64) * 2
    if rows > 1:
        ids[0] = -1
        ids[-1] = table_rows
    yield (
        "gather_rows",
        memory.gather_rows(rows, table_rows, width),
        {"rows": rows, "table_rows": table_rows, "width": width},
        [
            operand("table", "bf16", (table_rows, width)),
            operand("indices", "i64", (rows,)),
            operand("out", "bf16", (rows, width), "write"),
            operand("errors", "i32", (rows,), "write"),
        ],
        [torch.randn(table_rows, width).bfloat16(), ids],
    )
    yield (
        "scatter_rows",
        memory.scatter_rows(rows, table_rows, width),
        {"rows": rows, "table_rows": table_rows, "width": width, "unique_indices": True},
        [
            operand("values", "bf16", (rows, width)),
            operand("indices", "i64", (rows,)),
            operand("table", "bf16", (table_rows, width), "read_write"),
            operand("errors", "i32", (rows,), "write"),
        ],
        [torch.randn(rows, width).bfloat16(), ids, torch.randn(table_rows, width).bfloat16()],
    )
    yield (
        "dequant_rows",
        memory.dequant_rows(rows, width),
        {"rows": rows, "width": width, "group": 32},
        [
            operand("values", "i8", (rows, width)),
            operand("scales", "f32", (rows, width // 32)),
            operand("out", "bf16", (rows, width), "write"),
        ],
        [torch.randint(-128, 128, (rows, width), dtype=torch.int8), torch.rand(rows, width // 32)],
    )
    angles = torch.randn(rows, 32) * 1024
    yield (
        "rope",
        memory.rope(rows, 8),
        {"rows": rows, "heads": 8, "rotary": 64},
        [
            operand("x", "bf16", (rows, 8, 64)),
            operand("cosine", "f32", (rows, 32)),
            operand("sine", "f32", (rows, 32)),
            operand("out", "bf16", (rows, 8, 64), "write"),
        ],
        [torch.randn(rows, 8, 64).bfloat16(), angles.cos(), angles.sin()],
    )
    yield (
        "route",
        routing.route(rows),
        {"rows": rows, "experts": 384, "selected": 6, "routed_scale": 1.5},
        [
            operand("logits", "f32", (rows, 384)),
            operand("bias", "f32", (384,)),
            operand("indices", "i32", (rows, 8), "write"),
            operand("weights", "f32", (rows, 8), "write"),
        ],
        [torch.randn(rows, 384) * 8, torch.randn(384) * 0.1],
    )
    output = 64 if width < 2304 else 2304
    route_ids = torch.arange(rows * 8, dtype=torch.int32).view(rows, 8) % 48
    route_ids[0, 0] = -2
    route_ids[-1, 5] = 384
    regroup_parameters = {
        "rows": rows,
        "experts": 384,
        "local": 48,
        "selected": 6,
        "base": 0,
        "pad_last": False,
    }
    yield (
        "regroup",
        routing.regroup(**regroup_parameters),
        regroup_parameters,
        [
            operand("indices", "i32", (rows, 8)),
            operand("gather", "i32", (rows * 6,), "write"),
            operand("restore", "i32", (rows, 8), "write"),
            operand("counts", "i64", (48,), "write"),
            operand("errors", "i32", (rows + 1,), "read_write"),
        ],
        [route_ids, torch.zeros(rows + 1, dtype=torch.int32)],
    )
    for integer in (True, False):
        kind = "w8_projection" if integer else "bf16_projection"
        dtype = "i8" if integer else "bf16"
        x = (
            torch.randint(-128, 128, (rows, width), dtype=torch.int8)
            if integer
            else torch.randn(rows, width).bfloat16()
        )
        w = (
            torch.randint(-128, 128, (output, width), dtype=torch.int8)
            if integer
            else torch.randn(output, width).bfloat16()
        )
        sa = torch.rand(rows) / 127 if integer else torch.ones(rows)
        sw = torch.rand(output) / 127 if integer else torch.ones(output)
        yield (
            kind,
            dense.projection(rows, output, width, integer),
            {"rows": rows, "output": output, "reduction": width},
            [
                operand("activation", dtype, (rows, width)),
                operand("weight", dtype, (output, width)),
                operand("token_scale", "f32", (rows,)),
                operand("channel_scale", "f32", (output,)),
                operand("out", "bf16", (rows, output), "write"),
                operand(
                    "workspace",
                    "i32" if integer else "f32",
                    ((rows + 15) // 16 * 16, output),
                    "scratch",
                ),
            ],
            [x, w, sa, sw],
        )
    rotation, _ = torch.linalg.qr(torch.randn(128, 128))
    yield (
        "rotate_blocks",
        engram.rotate_blocks(rows, width),
        {"rows": rows, "width": width, "block": 128},
        [
            operand("hidden", "bf16", (rows, 4, width)),
            operand("rotation", "f32", (128, 128)),
            operand("out", "f32", (rows, 4, width), "write"),
        ],
        [torch.randn(rows, 4, width).bfloat16(), rotation],
    )
    mask = torch.ones(rows, dtype=torch.int8)
    if rows > 1:
        mask[0] = 0
    yield (
        "engram_gate",
        engram.gate(rows, width),
        {"rows": rows, "width": width, "eps": eps},
        [
            operand("hidden", "bf16", (rows, 4, width)),
            operand("original", "f32", (rows, 4, width)),
            operand("key", "bf16", (rows, 4, width)),
            operand("value", "bf16", (rows, width)),
            operand("channel_weight", "f32", (4, width)),
            operand("token_mask", "i8", (rows,)),
            operand("out", "bf16", (rows, 4, width), "write"),
        ],
        [
            torch.randn(rows, 4, width).bfloat16(),
            torch.randn(rows, 4, width),
            torch.randn(rows, 4, width).bfloat16(),
            torch.randn(rows, width).bfloat16(),
            torch.randn(4, width),
            mask,
        ],
    )
    selected, dimension, heads = attention_selected, 512, 8
    sparse_ids = torch.arange(selected, dtype=torch.int32).expand(rows, -1).clone()
    sparse_ids[:, -7:] = -1
    yield (
        "sparse_attention",
        attention.sparse_attention(rows, heads, selected, dimension),
        {
            "rows": rows,
            "heads": heads,
            "selected": selected,
            "dimension": dimension,
            "scale": dimension**-0.5,
        },
        [
            operand("query", "bf16", (rows, heads, dimension)),
            operand("gathered_kv", "bf16", (rows, selected, dimension)),
            operand("indices", "i32", (rows, selected)),
            operand("sink", "f32", (heads,)),
            operand("out", "bf16", (rows, heads, dimension), "write"),
        ],
        [
            torch.randn(rows, heads, dimension).bfloat16(),
            torch.randn(rows, selected, dimension).bfloat16(),
            sparse_ids,
            torch.randn(heads),
        ],
    )
    projection_output = 64
    yield (
        "fp32_partials",
        dense.fp32_partials(rows, projection_output, width),
        {"rows": rows, "output": projection_output, "reduction": width, "splits": 8},
        [
            operand("activation", "bf16", (rows, width)),
            operand("weight", "f32", (projection_output, width)),
            operand("out", "f32", (rows, projection_output // 32, 8, 32), "write"),
        ],
        [torch.randn(rows, width).bfloat16(), torch.randn(projection_output, width)],
    )
    yield (
        "fp32_reduce",
        dense.fp32_reduce(rows, projection_output),
        {"rows": rows, "output": projection_output, "splits": 8},
        [
            operand("partials", "f32", (rows, projection_output // 32, 8, 32)),
            operand("out", "f32", (rows, projection_output), "write"),
        ],
        [torch.randn(rows, projection_output // 32, 8, 32)],
    )
    metadata = torch.zeros(rows, 8, dtype=torch.int32)
    last_rows = {}
    for row in range(rows):
        page = row % 2 + 1
        position = (31 if page == 1 else 63) + row // 2
        previous = last_rows.get(page, -1)
        metadata[row, :6] = torch.tensor([position, page, previous, 1, position, 1])
        last_rows[page] = row
    if rows > 1:
        metadata[-1].zero_()
    for page in (1, 2):
        chosen = (metadata[:, 1] == page) & metadata[:, 3].bool() & metadata[:, 5].bool()
        if chosen.any():
            metadata[metadata[:, 1] == page, 4] = metadata[chosen, 0].max()
    c2_width, pages = 512, 4
    kv = torch.randn(rows, c2_width)
    scores = torch.randn(rows, c2_width) * 8
    ring = torch.randn(pages, 32, 2 * c2_width)
    yield (
        "c2_pool",
        compressor.pool(rows, pages, c2_width),
        {"tokens": rows, "pages": pages, "width": c2_width},
        [
            operand("kv", "f32", (rows, c2_width)),
            operand("scores", "f32", (rows, c2_width)),
            operand("ring", "f32", (pages, 32, 2 * c2_width)),
            operand("metadata", "i32", (rows, 8)),
            operand("out", "bf16", (rows, c2_width), "write"),
            operand("errors", "i32", (rows,), "write"),
        ],
        [kv, scores, ring, metadata],
    )
    yield (
        "c2_commit",
        compressor.commit(rows, pages, c2_width),
        {"tokens": rows, "pages": pages, "width": c2_width},
        [
            operand("kv", "f32", (rows, c2_width)),
            operand("scores", "f32", (rows, c2_width)),
            operand("metadata", "i32", (rows, 8)),
            operand("ring", "f32", (pages, 32, 2 * c2_width), "read_write"),
            operand("errors", "i32", (rows,), "write"),
        ],
        [kv, scores, metadata, ring],
    )
    chosen = select_k
    select_block = 4096 if chosen <= 512 else 2048
    chunks = (index_length + select_block - 1) // select_block
    score_rows = torch.stack([torch.randperm(index_length).float() for _ in range(rows)])
    lengths = torch.full((rows,), index_length, dtype=torch.int32)
    yield (
        "select_local",
        select.local(rows, index_length, chosen),
        {"rows": rows, "length": index_length, "block": select_block, "selected": chosen},
        [
            operand("scores", "f32", (rows, index_length)),
            operand("lengths", "i32", (rows,)),
            operand("out", "f32", (rows, chunks, 2 * chosen), "write"),
        ],
        [score_rows, lengths],
    )
    sources = 5
    pairs = torch.empty(rows, sources, 2 * chosen)
    for row in range(rows):
        ranks = torch.randperm(sources * chosen).float().view(sources, chosen)
        for source in range(sources):
            order = ranks[source].argsort(descending=True)
            pairs[row, source, 0::2] = ranks[source, order]
            pairs[row, source, 1::2] = order.float() + source * chosen
    yield (
        "select_merge",
        select.merge(rows, sources, chosen),
        {"rows": rows, "sources": sources, "selected": chosen},
        [
            operand("candidates", "f32", (rows, sources, 2 * chosen)),
            operand("out", "f32", (rows, (sources + 3) // 4, 2 * chosen), "write"),
        ],
        [pairs],
    )
    final_pairs = pairs[:, 0].clone()
    if rows > 1:
        final_pairs[-1, chosen::2] = -torch.inf
    yield (
        "select_finalize",
        select.finalize(rows, chosen),
        {"rows": rows, "selected": chosen},
        [
            operand("candidates", "f32", (rows, 2 * chosen)),
            operand("indices", "i32", (rows, chosen), "write"),
            operand("values", "f32", (rows, chosen), "write"),
        ],
        [final_pairs],
    )
    yield (
        "mhc_head_mix",
        vector.mhc_head_mix(rows),
        {"rows": rows, "eps": 1e-6},
        [
            operand("mixes", "f32", (rows, 8)),
            operand("scale", "f32", (8,)),
            operand("base", "f32", (8,)),
            operand("out", "f32", (rows, 8), "write"),
        ],
        [torch.randn(rows, 8), torch.randn(8), torch.randn(8)],
    )
    for kind, dtype, kernel_dtype, size in (
        ("gather_activation", torch.int8, "int8", width),
        ("gather_scales", torch.float32, "float32", 1),
    ):
        bank = (
            torch.randint(-128, 128, (table_rows, size), dtype=dtype)
            if dtype == torch.int8
            else torch.rand(table_rows, size)
        )
        abi_type = "i8" if dtype == torch.int8 else "f32"
        yield (
            kind,
            memory.gather_rows(rows, table_rows, size, kernel_dtype, "int32"),
            {"rows": rows, "table_rows": table_rows, "width": size},
            [
                operand("table", abi_type, (table_rows, size)),
                operand("indices", "i32", (rows,)),
                operand("out", abi_type, (rows, size), "write"),
                operand("errors", "i32", (rows,), "write"),
            ],
            [bank, ids.int()],
        )
    expert_rows = rows * 6
    row_map = torch.full((rows, 8), -1, dtype=torch.int32)
    row_map[:, :6] = torch.randperm(expert_rows).view(rows, 6).int()
    if rows > 1:
        row_map[0, 1] = -1
        row_map[-1, 5] = expert_rows
    yield (
        "moe_combine",
        moe.combine(rows, expert_rows, width),
        {"rows": rows, "expert_rows": expert_rows, "width": width, "selected": 6},
        [
            operand("experts", "bf16", (expert_rows, width)),
            operand("shared", "bf16", (rows, width)),
            operand("row_map", "i32", (rows, 8)),
            operand("weights", "f32", (rows, 8)),
            operand("out", "bf16", (rows, width), "write"),
            operand("errors", "i32", (rows,), "write"),
        ],
        [
            torch.randn(expert_rows, width).bfloat16(),
            torch.randn(rows, width).bfloat16(),
            row_map,
            torch.rand(rows, 8),
        ],
    )
    logit_length, table = 32, 64
    key_ids = torch.randint(0, table, (rows, logit_length), dtype=torch.int32)
    key_ids[:, -3:] = -1
    key_ids[0, 0] = table
    yield (
        "indexer_logits",
        indexer.logits(rows, logit_length, table, scale=128**-0.5),
        {
            "rows": rows,
            "length": logit_length,
            "table_rows": table,
            "heads": 4,
            "dimension": 128,
            "scale": 128**-0.5,
        },
        [
            operand("query", "bf16", (rows, 4, 128)),
            operand("keys", "bf16", (table, 128)),
            operand("weights", "f32", (rows, 8)),
            operand("indices", "i32", (rows, logit_length)),
            operand("out", "f32", (rows, logit_length), "write"),
            operand("errors", "i32", (rows, logit_length // 16), "write"),
        ],
        [
            torch.randn(rows, 4, 128).bfloat16(),
            torch.randn(table, 128).bfloat16(),
            torch.rand(rows, 8),
            key_ids,
        ],
    )

    yield (
        "mhc_post_pre",
        vector.mhc_post_pre(rows, width),
        {"rows": rows, "width": width},
        [
            operand("layer", "bf16", (rows, width)),
            operand("residual", "bf16", (rows, 4, width)),
            operand("post", "f32", (rows, 8)),
            operand("combination", "f32", (rows, 16)),
            operand("out", "bf16", (rows, 4, width), "write"),
            operand("mix", "f32", (rows, 8)),
            operand("collapsed", "bf16", (rows, width), "write"),
        ],
        [
            torch.randn(rows, width).bfloat16(),
            torch.randn(rows, 4, width).bfloat16(),
            torch.randn(rows, 8),
            torch.randn(rows, 16),
            torch.randn(rows, 8),
        ],
    )
    table_rows = max(1, rows - 1)
    indices = torch.arange(rows, dtype=torch.int32) % table_rows
    indices[0] = -1
    if rows > 1:
        indices[-1] = table_rows
    yield (
        "gather_quantized_rows",
        memory.gather_quantized_rows(rows, table_rows, width),
        {"rows": rows, "table_rows": table_rows, "width": width},
        [
            operand("table", "i8", (table_rows, width)),
            operand("scales", "f32", (table_rows,)),
            operand("indices", "i32", (rows,)),
            operand("out", "i8", (rows, width), "write"),
            operand("out_scales", "f32", (rows,), "write"),
            operand("errors", "i32", (rows,), "write"),
        ],
        [
            torch.randint(-127, 128, (table_rows, width), dtype=torch.int8),
            torch.rand(table_rows),
            indices,
        ],
    )
    yield (
        "gather_columns",
        memory.gather_columns(rows, width),
        {"rows": rows, "width": width},
        [
            operand("gathered", "bf16", (8, rows, width // 8)),
            operand("out", "bf16", (rows, width), "write"),
        ],
        [torch.randn(8, rows, width // 8).bfloat16()],
    )


def write_tensor(path, tensor):
    path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return {"file": path.name, "sha256": file_hash(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compile-only", action="store_true", help="compile without opening an NPU device"
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--index-length", type=int, default=8192)
    parser.add_argument("--attention-selected", type=int, default=32)
    parser.add_argument("--select-k", type=int, choices=(512, 2048), default=512)
    parser.add_argument(
        "--quant-grid",
        action="store_true",
        help="exercise all 128 BF16 mantissas with both signs (rows=128, width=512)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=(
            "rms_norm",
            "mhc_post_pre",
            "gather_quantized_rows",
            "gather_columns",
            "quantize",
            "swiglu",
            "mhc_split_sinkhorn",
            "mhc_expand",
            "mhc_pre_apply",
            "mhc_post",
            "mhc_prenorm_partials",
            "mhc_prenorm_reduce",
            "gather_rows",
            "scatter_rows",
            "dequant_rows",
            "rope",
            "route",
            "regroup",
            "w8_projection",
            "bf16_projection",
            "rotate_blocks",
            "engram_gate",
            "sparse_attention",
            "fp32_partials",
            "fp32_reduce",
            "c2_pool",
            "c2_commit",
            "select_local",
            "select_merge",
            "select_finalize",
            "mhc_head_mix",
            "gather_activation",
            "gather_scales",
            "moe_combine",
            "indexer_logits",
        ),
    )
    args = parser.parse_args()
    if args.rows < 1 or args.width < 128 or args.width % 128:
        parser.error("positive rows and width divisible by 128 are required")
    if not 0 < args.index_length <= 1 << 24:
        parser.error("index length must be in 1..2^24")
    if args.attention_selected < 16 or args.attention_selected % 16:
        parser.error("attention selected length must be a positive multiple of 16")
    if args.quant_grid and (args.rows, args.width) != (128, 512):
        parser.error("quant grid requires rows=128 and width=512")
    if not args.compile_only:
        import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(7)
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for kind, func, params, operands, initial in cases(
        args.rows, args.width, args.index_length, args.attention_selected, args.select_k
    ):
        if args.only and kind not in args.only:
            continue
        if kind == "quantize" and args.quant_grid:
            maxima = torch.arange(128, 256).float()[:, None] / 128
            grid = torch.minimum(torch.arange(256).float()[None, :] / 128, maxima)
            initial = [torch.cat([grid, -grid], dim=1).bfloat16()]
        manifest_path = compile_kernel(func, kind, operands, params, args.output / "cache")
        if args.compile_only:
            print(json.dumps({"op": kind, "manifest": str(manifest_path)}), flush=True)
            continue
        manifest = json.loads(manifest_path.read_text())
        library = ctypes.CDLL(str(manifest_path.parent / manifest["library"]["file"]))
        entry = library.orinfer_ascend_kernel_v1
        entry.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_void_p]
        entry.restype = ctypes.c_int32
        passes = []
        expected = reference(kind, initial, params)
        input_slots = [
            i for i, operand in enumerate(operands) if operand["access"] in ("read", "read_write")
        ]
        output_slots = [
            i for i, operand in enumerate(operands) if operand["access"] in ("write", "read_write")
        ]
        by_input = dict(zip(input_slots, initial))
        by_output = dict(zip(output_slots, expected))
        types = {
            "bf16": torch.bfloat16,
            "f16": torch.float16,
            "f32": torch.float32,
            "i8": torch.int8,
            "i32": torch.int32,
            "i64": torch.int64,
        }
        tensors = []
        for slot, spec_operand in enumerate(operands):
            if slot in by_input:
                tensors.append(by_input[slot].contiguous().npu())
            elif slot in by_output:
                wanted = by_output[slot]
                tensors.append(torch.empty(wanted.shape, dtype=wanted.dtype, device="npu"))
            else:
                tensors.append(
                    torch.empty(
                        spec_operand["shape"], dtype=types[spec_operand["dtype"]], device="npu"
                    )
                )
        device_inputs = [tensors[i] for i in input_slots]
        device_outputs = [tensors[i] for i in output_slots]
        pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])
        for p in range(4):
            inputs = [t.clone() for t in initial]
            if p == 1:
                inputs[0].zero_()
                if kind == "route":
                    inputs[1].zero_()
                if kind == "sparse_attention":
                    inputs[2].fill_(-1)
                if kind == "select_local":
                    inputs[0] = (
                        torch.arange(params["length"], dtype=torch.float32)
                        .expand(args.rows, -1)
                        .contiguous()
                    )
                    inputs[1][0] = 0
                    if args.rows > 1:
                        inputs[1][-1] = max(0, params["length"] - 11)
                if kind in ("select_merge", "select_finalize"):
                    inputs[0][..., 0::2] = -torch.inf
                    inputs[0][..., 1::2] = -1
            elif p >= 2:
                inputs[0] = (inputs[0].float() * (-0.5 if p == 2 else 2.0)).to(inputs[0].dtype)
                if kind == "mhc_prenorm_reduce":
                    inputs[0][:, :, 24] = initial[0][:, :, 24] * (0.25 if p == 2 else 4.0)
                if kind in ("select_merge", "select_finalize"):
                    inputs[0] = initial[0].clone()
                    inputs[0][..., 0::2] *= 0.5 if p == 2 else 2.0
                if kind == "gather_quantized_rows":
                    inputs[2] = (
                        torch.arange(params["rows"] - 1, -1, -1, dtype=torch.int32)
                        % params["table_rows"]
                        if p == 2
                        else torch.full_like(inputs[2], -2)
                    )
            expected = reference(kind, inputs, params)
            for dst, src in zip(device_inputs, inputs):
                dst.copy_(src)
            for slot in output_slots:
                if operands[slot]["access"] == "write":
                    tensors[slot].fill_(17)
            status = entry(pointers, len(tensors), torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(f"{kind}: native launch error {status}")
            torch.npu.synchronize()
            saved_inputs = []
            saved_outputs = []
            errors = []
            for slot, t in zip(input_slots, inputs):
                identity = write_tensor(args.output / f"{kind}-pass{p}-in{slot}.bin", t)
                saved_inputs.append({"slot": slot, "data": identity})
            for j, (actual, wanted) in enumerate(zip(device_outputs, expected)):
                atol, rtol = (
                    (0.0, 0.0)
                    if not wanted.is_floating_point() or kind in EXACT_OPS
                    else ((2e-5, 2e-5) if wanted.dtype == torch.float32 else (0.003, 0.008))
                )
                torch.testing.assert_close(actual.cpu(), wanted, atol=atol, rtol=rtol)
                identity = write_tensor(args.output / f"{kind}-pass{p}-out{j}.bin", wanted)
                saved_outputs.append(
                    {"slot": output_slots[j], "data": identity, "atol": atol, "rtol": rtol}
                )
                a, b = actual.cpu().float(), wanted.float()
                errors.append(float(torch.where(a == b, 0.0, (a - b).abs()).max()))
            passes.append({"inputs": saved_inputs, "outputs": saved_outputs})
            print(json.dumps({"op": kind, "pass": p, "max_abs": errors}), flush=True)
        fixtures.append({"manifest": str(manifest_path.relative_to(args.output)), "passes": passes})
        atomic_json(args.output / "fixtures.json", {"schema_version": 1, "cases": fixtures})
    if not args.compile_only:
        atomic_json(args.output / "fixtures.json", {"schema_version": 1, "cases": fixtures})


if __name__ == "__main__":
    main()
