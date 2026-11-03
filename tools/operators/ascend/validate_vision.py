"""Numerical, bounded-row and changed-input Graph checks for shared vision kernels."""

import argparse
import ctypes
import math
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--part", choices=["core", "layout", "routing"], default="core")
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    from kernels.model.rows import row_count
    from tools.operators.ascend import vision
    from tools.operators.ascend.build import compile_kernel, operand

    torch.manual_seed(20261002)
    torch.set_num_threads(4)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    reports, libraries = [], []

    def entry(function, name, ports, params, capacity):
        manifest = compile_kernel(
            function, name, ports, params, args.output / "cache", row_capacity=capacity
        )
        if args.compile_only:
            return None
        lib = ctypes.CDLL(str(manifest.parent / "kernel.so"))
        libraries.append(lib)
        call = lib.orinfer_ascend_kernel_v2
        call.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        call.restype = ctypes.c_int32
        return call

    if args.part == "routing":
        check_routing(args, torch, row_count, operand, entry, reports)
        atomic_json(
            args.output / "routing-result.json", dict(passed=not args.compile_only, checks=reports)
        )
        return

    if args.part == "layout":
        check_layout(args, torch, vision, row_count, operand, entry, reports)
        atomic_json(
            args.output / "layout-result.json",
            dict(passed=not args.compile_only, compile_only=args.compile_only, checks=reports),
        )
        return

    for activation, biased, residual in [
        ("linear", True, True),
        ("silu_mul", False, False),
        ("gelu", True, False),
    ]:
        capacity, width = 129, 128
        columns = width * (2 if activation == "silu_mul" else 1)
        params = dict(
            rows=capacity, width=width, activation=activation, biased=biased, residual=residual
        )
        ports = [
            operand("source", "f32", [capacity, columns]),
            operand("bias", "bf16", [columns]),
            operand("skip", "bf16", [capacity, width]),
            operand("output", "bf16", [capacity, width], "write"),
        ]
        call = entry(
            vision.epilogue(**dict(params, rows=row_count(capacity, True))),
            "vision.epilogue",
            ports,
            params,
            capacity,
        )
        if args.compile_only:
            continue
        source = torch.randn(capacity, columns) * 2
        bias = (torch.randn(columns) * 0.2).bfloat16()
        skip = torch.randn(capacity, width).bfloat16()
        device_source, device_bias, device_skip = source.npu(), bias.npu(), skip.npu()
        output = torch.full((capacity + 1, width), 13.0, dtype=torch.bfloat16, device="npu")
        pointers = (ctypes.c_void_p * 4)(
            device_source.data_ptr(),
            device_bias.data_ptr(),
            device_skip.data_ptr(),
            output.data_ptr(),
        )
        for rows in [1, 17, 64, capacity]:
            status = call(pointers, 4, rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(("vision epilogue", status))
            torch.npu.synchronize()
            rounded = (source[:rows] + (bias.float() if biased else 0)).bfloat16()
            if activation == "silu_mul":
                gate, up = rounded.chunk(2, dim=-1)
                expected = torch.nn.functional.silu(gate) * up
            elif activation == "gelu":
                expected = torch.nn.functional.gelu(rounded)
            else:
                expected = rounded
            if residual:
                expected = expected + skip[:rows]
            actual = output[:rows].cpu()
            error = (actual.float() - expected.float()).abs()
            # GELU approximation can differ by one BF16 ULP at a rounding tie.
            tolerance = 0.0 if activation == "linear" else 0.03125
            assert error.max().item() <= tolerance, (activation, error.max().item())
            assert torch.all(output[rows:] == 13), "Vision epilogue wrote beyond real rows"
            reports.append(
                dict(op="epilogue", activation=activation, rows=rows, max_abs=error.max().item())
            )
            output.fill_(13)

    capacity, heads, dimension, tile_m, tile_k = 768, 2, 64, 128, 256
    slots = 4
    blocks = slots
    params = dict(
        rows=capacity,
        capacity=capacity,
        heads=heads,
        dimension=dimension,
        tile_m=tile_m,
        tile_k=tile_k,
        workspace_slots=slots,
    )
    ports = [
        operand("qkv", "bf16", [2, capacity, heads * dimension]),
        operand("values", "f16", [capacity, heads * dimension]),
        operand("output", "bf16", [capacity, heads * dimension], "write"),
        operand("scores", "f32", [blocks * 2, tile_m, tile_k], "scratch"),
        operand("probabilities", "f16", [blocks, tile_m, tile_k], "scratch"),
        operand("products", "f32", [slots, tile_m, dimension], "scratch"),
    ]
    call = entry(
        vision.attention(**dict(params, rows=row_count(capacity, True))),
        "vision.attention",
        ports,
        params,
        capacity,
    )
    if not args.compile_only:
        qkv = (torch.randn(2, capacity, heads * dimension) * 0.2).bfloat16().npu()
        values = (torch.randn(capacity, heads * dimension) * 0.2).bfloat16().half().npu()
        output = torch.full(
            (capacity + 1, heads * dimension), 13.0, dtype=torch.bfloat16, device="npu"
        )
        scores = torch.empty(blocks * 2, tile_m, tile_k, device="npu")
        probabilities = torch.empty(blocks, tile_m, tile_k, dtype=torch.float16, device="npu")
        products = torch.empty(slots, tile_m, dimension, device="npu")
        pointers = (ctypes.c_void_p * 6)(
            qkv.data_ptr(),
            values.data_ptr(),
            output.data_ptr(),
            scores.data_ptr(),
            probabilities.data_ptr(),
            products.data_ptr(),
        )

        def launch(rows):
            status = call(pointers, 6, rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(("vision attention", status))

        def reference(rows):
            q, k = (
                qkv[:, :rows]
                .view(2, rows, heads, dimension)
                .permute(0, 2, 1, 3)
                .float()
                .cpu()
                .unbind()
            )
            v = values[:rows].view(rows, heads, dimension).permute(1, 0, 2).float().cpu()
            p = (q @ k.transpose(-2, -1) / dimension**0.5).softmax(-1)
            return (p @ v).transpose(0, 1).reshape(rows, -1)

        for rows in [1, 17, 65, 129, 255, 257, 511, capacity]:
            launch(rows)
            torch.npu.synchronize()
            actual, expected = output[:rows].float().cpu(), reference(rows)
            error = (actual - expected).abs()
            relative_rms = error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(
                1e-8
            )
            assert error.max().item() < 0.003 and relative_rms.item() < 0.02, (
                rows,
                error.max().item(),
                relative_rms.item(),
            )
            assert torch.all(output[rows:] == 13), "Vision attention wrote beyond real rows"
            reports.append(
                dict(
                    op="attention",
                    rows=rows,
                    max_abs=error.max().item(),
                    relative_rms=relative_rms.item(),
                )
            )
            output.fill_(13)
        rows = 65
        launch(rows)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            launch(rows)
        before = output[:rows].clone()
        qkv.mul_(1.25)
        values.mul_(1.25)
        graph.replay()
        torch.npu.synchronize()
        actual, expected = output[:rows].float().cpu(), reference(rows)
        assert (actual - expected).abs().max().item() < 0.003
        assert not torch.equal(before, output[:rows]), "Vision Graph reused stale inputs"
        reports.append(dict(op="attention", graph_changed_inputs=True))
        # A large population of subnormal FP16 probabilities must contribute
        # to PV; a flush here would discard about 1.3% of the probability mass.
        qkv.zero_()
        qkv.view(2, capacity, heads, dimension)[0, :, :, 0] = 8
        qkv.view(2, capacity, heads, dimension)[1, 1:, :, 0] = -11
        values.zero_()
        values[1:].fill_(1)
        launch(capacity)
        torch.npu.synchronize()
        expected = (capacity - 1) * math.exp(-11) / (1 + (capacity - 1) * math.exp(-11))
        error = (output[:capacity].float().cpu() - expected).abs().max().item()
        assert error < 0.0002, ("FP16 subnormal probability mass", error)
        reports.append(dict(op="attention", subnormal_probability_mass=True, max_abs=error))
    atomic_json(
        args.output / "result.json",
        dict(passed=not args.compile_only, compile_only=args.compile_only, checks=reports),
    )


def check_layout(args, torch, vision, row_count, operand, entry, reports):
    capacity, rows_max, heads, dim = 256, 129, 16, 64
    hidden = heads * dim
    ports = [
        operand("source", "f32", [rows_max, 3 * hidden]),
        operand("bias", "bf16", [3 * hidden]),
        operand("phase", "f32", [rows_max, dim]),
        operand("qkv", "bf16", [2, capacity, hidden], "write"),
        operand("values", "f16", [capacity, hidden], "write"),
    ]
    params = dict(rows=rows_max, capacity=capacity, heads=heads, dimension=dim)
    call = entry(
        vision.qkv_rope(**dict(params, rows=row_count(rows_max, True))),
        "vision.qkv_rope",
        ports,
        params,
        rows_max,
    )
    if not args.compile_only:
        source = torch.randn(rows_max, 3 * hidden) * 0.2
        bias = (torch.randn(3 * hidden) * 0.1).bfloat16()
        angles = torch.randn(rows_max, dim // 2)
        phase = torch.cat((angles.cos(), angles.sin()), dim=-1)
        ds, db, dp = source.npu(), bias.npu(), phase.npu()
        out = torch.full((2, capacity, hidden), 13, dtype=torch.bfloat16, device="npu")
        values = torch.full((capacity, hidden), 13, dtype=torch.float16, device="npu")
        pointers = (ctypes.c_void_p * 5)(
            ds.data_ptr(), db.data_ptr(), dp.data_ptr(), out.data_ptr(), values.data_ptr()
        )
        for rows in [1, 17, 65, rows_max]:
            status = call(pointers, 5, rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(("vision QKV/RoPE", status))
            torch.npu.synchronize()
            expected = (
                (source[:rows] + bias.float())
                .bfloat16()
                .view(rows, 3, heads, dim)
                .permute(1, 0, 2, 3)
                .contiguous()
            )
            cosine, sine = phase[:rows].chunk(2, dim=-1)
            for part in [0, 1]:
                left, right = expected[part].float().chunk(2, dim=-1)
                expected[part] = torch.cat(
                    (
                        left * cosine[:, None] - right * sine[:, None],
                        right * cosine[:, None] + left * sine[:, None],
                    ),
                    dim=-1,
                ).bfloat16()
            actual = out[:, :rows].view(2, rows, heads, dim).cpu()
            error = (actual.float() - expected[:2].float()).abs()
            assert error.max().item() <= 0.0078125
            assert torch.equal(values[:rows].cpu(), expected[2].reshape(rows, hidden).half())
            padded = (rows + 127) // 128 * 128
            assert torch.all(out[:, rows:padded] == 0)
            assert torch.all(out[:, padded:] == 13)
            assert torch.all(values[rows:padded] == 0)
            assert torch.all(values[padded:] == 13)
            reports.append(
                dict(op="qkv_rope", rows=rows, max_abs=error.max().item(), cleared_padding=True)
            )
            out.fill_(13)
            values.fill_(13)

    width, maximum = 1024, 32
    params = dict(rows=maximum, capacity=capacity, width=width, ratio=3, channels=512)
    ports = [
        operand("source", "bf16", [capacity, width]),
        operand("grid", "i32", [8]),
        operand("output", "bf16", [maximum, width * 9], "write"),
    ]
    call = entry(
        vision.merge(**dict(params, rows=row_count(maximum, True))),
        "vision.merge",
        ports,
        params,
        maximum,
    )
    if args.compile_only:
        return
    source = torch.randn(capacity, width).bfloat16().npu()
    grid = torch.zeros(8, dtype=torch.int32, device="npu")
    output = torch.full((maximum + 1, width * 9), 13, dtype=torch.bfloat16, device="npu")
    pointers = (ctypes.c_void_p * 3)(source.data_ptr(), grid.data_ptr(), output.data_ptr())

    def launch(rows):
        status = call(pointers, 3, rows, torch.npu.current_stream()._as_parameter_)
        if status:
            raise RuntimeError(("vision merge", status))

    def reference(height, width):
        x = source[: height * width].cpu().view(height, width, -1).permute(2, 0, 1)
        x = torch.nn.functional.pad(x, (0, -width % 3, 0, -height % 3))
        return torch.nn.functional.unfold(x[None].float(), 3, stride=3).squeeze(0).T.bfloat16()

    for height, columns in [(2, 4), (5, 7), (13, 14)]:
        rows = ((height + 2) // 3) * ((columns + 2) // 3)
        grid[:2] = torch.tensor([height, columns], dtype=torch.int32, device="npu")
        launch(rows)
        torch.npu.synchronize()
        assert torch.equal(output[:rows].cpu(), reference(height, columns))
        assert torch.all(output[rows:] == 13)
        reports.append(dict(op="merge", height=height, width=columns, rows=rows, exact=True))
        output.fill_(13)
    grid[:2] = torch.tensor([5, 7], dtype=torch.int32, device="npu")
    launch(6)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch(6)
    grid[:2] = torch.tensor([7, 5], dtype=torch.int32, device="npu")
    source.mul_(1.25)
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(output[:6].cpu(), reference(7, 5))
    reports.append(dict(op="merge", graph_changed_inputs_and_grid=True))


def check_routing(args, torch, row_count, operand, entry, reports):
    from tools.operators.ascend.routing import route

    capacity, experts, selected, stride = 129, 384, 6, 8
    base_ports = [
        operand("logits", "f32", [capacity, experts]),
        operand("bias", "f32", [experts]),
        operand("indices", "i32", [capacity, stride], "write"),
        operand("weights", "f32", [capacity, stride], "write"),
    ]
    calls = []
    for vision_bias in [False, True]:
        ports = base_ports + (
            [operand("image_bias", "f32", [experts]), operand("image_rows", "i32", [capacity])]
            if vision_bias
            else []
        )
        params = dict(rows=capacity, experts=experts, selected=selected, vision_bias=vision_bias)
        calls.append(
            entry(
                route(**dict(params, rows=row_count(capacity, True))),
                "routing.route",
                ports,
                params,
                capacity,
            )
        )
    if args.compile_only:
        return
    logits = torch.randn(capacity, experts, device="npu") * 3
    bias = torch.randn(experts, device="npu") * 0.5
    image_bias = torch.randn(experts, device="npu") * 1.5
    mask = (torch.arange(capacity, device="npu") % 3 == 1).int()
    indices = torch.full((capacity + 1, stride), -13, dtype=torch.int32, device="npu")
    weights = torch.full((capacity + 1, stride), 13.0, device="npu")
    ports = [logits, bias, indices, weights, image_bias, mask]

    def launch(rows, visual=True):
        count = 6 if visual else 4
        pointers = (ctypes.c_void_p * count)(*(x.data_ptr() for x in ports[:count]))
        status = calls[int(visual)](
            pointers, count, rows, torch.npu.current_stream()._as_parameter_
        )
        if status:
            raise RuntimeError(("vision routing", status))

    def reference(rows):
        scores = torch.nn.functional.softplus(logits[:rows].cpu()).sqrt()
        correction = torch.where(
            mask[:rows].cpu()[:, None] != 0, image_bias.cpu()[None], bias.cpu()[None]
        )
        chosen = (scores + correction).topk(selected, dim=-1).indices
        weight = scores.gather(1, chosen)
        return chosen, weight / weight.sum(-1, keepdim=True) * 1.5

    for rows in [1, 17, capacity]:
        launch(rows)
        torch.npu.synchronize()
        chosen, expected = reference(rows)
        assert torch.equal(indices[:rows, :selected].long().cpu(), chosen)
        assert (weights[:rows, :selected].cpu() - expected).abs().max().item() < 2e-5
        assert torch.all(indices[rows:] == -13) and torch.all(weights[rows:] == 13)
        reports.append(dict(op="routing", rows=rows, mixed_image_text=True, indices_exact=True))
        indices.fill_(-13)
        weights.fill_(13)
    graph = torch.npu.NPUGraph()
    launch(17)
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        launch(17)
    mask.copy_(1 - mask)
    graph.replay()
    torch.npu.synchronize()
    chosen, expected = reference(17)
    assert torch.equal(indices[:17, :selected].long().cpu(), chosen)
    assert (weights[:17, :selected].cpu() - expected).abs().max().item() < 2e-5
    mask.zero_()
    launch(17)
    torch.npu.synchronize()
    original_indices, original_weights = indices.clone(), weights.clone()
    launch(17, visual=False)
    torch.npu.synchronize()
    assert torch.equal(indices, original_indices) and torch.equal(weights, original_weights)
    reports.append(dict(op="routing", graph_changed_mask=True, text_abi_exact=True))


if __name__ == "__main__":
    main()
