"""Validate indirect A2 owner pointers, tails, capture guards and changed Graph data."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json, file_hash, source_hash


def measurement_source():
    root = Path(__file__).resolve().parents[3]
    return source_hash(
        root,
        [
            Path(__file__).resolve(),
            root / "tools/model/tuning.py",
            root / "tools/model/deepseek_v41/tune.py",
            root / "tools/model/row_contract.py",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    p.add_argument("--owners", type=int, default=3)
    p.add_argument("--rows", type=int, nargs="+", help="Owner row lengths for gather or Cube PV")
    p.add_argument("--attention-heads", type=int, nargs="+", default=[8])
    p.add_argument("--pv-tile-d", type=int, nargs="+", choices=(32, 64, 128, 256), default=[32])
    p.add_argument("--benchmark-rows", type=int, nargs="+", help="Measure single-owner PV lengths")
    p.add_argument("--kv-compressed", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--bulk-store", action="store_true")
    p.add_argument(
        "--reuse-long", action="store_true", help="Check full gather then SWA-only reuse"
    )
    p.add_argument("--op", nargs="+", help="Restrict validation to named primitive families")
    a = p.parse_args()
    if not 1 <= min(a.attention_heads) <= max(a.attention_heads) <= 64:
        p.error("attention heads must be in1..64")
    if not 1 <= a.owners <= 128:
        p.error("owners must be in 1..128")
    if a.rows and (
        len(a.rows) != a.owners
        or not 1 <= min(a.rows) <= max(a.rows) <= 2048
        or a.op not in (["kv_int8.gather_attention"], ["attention_cube.pv"])
    ):
        p.error("--rows requires one 1..2048 length per owner and a gather or Cube PV --op")
    if a.benchmark_rows and (
        a.op != ["attention_cube.pv"]
        or not 1 <= min(a.benchmark_rows) <= max(a.benchmark_rows) <= 2048
    ):
        p.error("--benchmark-rows requires --op attention_cube.pv and lengths in1..2048")
    if a.reuse_long and (a.op != ["kv_int8.gather_attention"] or a.kv_compressed is not True):
        p.error("--reuse-long requires --op kv_int8.gather_attention --kv-compressed")
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.operators.ascend import (
        attention_cube,
        attention_prefill,
        attention_fused,
        kv_int8,
        compressor,
    )
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.operators.ascend.geometry import physical_shapes
    from tools.operators.ascend.private_rows import indirect

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(a.device)
    capacity, owners = max((a.rows or [16]) + (a.benchmark_rows or [])), 128
    kinds = {
        "int32": ("i32", torch.int32),
        "int8": ("i8", torch.int8),
        "float16": ("f16", torch.float16),
        "bfloat16": ("bf16", torch.bfloat16),
        "float32": ("f32", torch.float32),
    }
    report = []
    for module, name, params in [
        *[
            (attention_cube, "pv", dict(rows=capacity, heads=heads, selected=selected, tile_d=tile))
            for selected in (128, 640)
            for heads in a.attention_heads
            for tile in a.pv_tile_d
        ],
        (kv_int8, "scatter", dict(rows=capacity, table_rows=256)),
        *[
            (
                kv_int8,
                "gather_attention",
                dict(
                    rows=capacity,
                    capacity=1024,
                    compressed_capacity=512,
                    compressed=compressed,
                    swa_storage_rows=256,
                    tile_slots=tile,
                    bulk_store=a.bulk_store,
                    reuse_long=a.reuse_long,
                ),
            )
            for compressed in (False, True)
            for tile in (1, 4, 8, 16)
        ],
        (attention_prefill, "qk", dict(rows=capacity, capacity=256, compressed=False)),
        (attention_prefill, "pv", dict(rows=capacity, capacity=256, compressed=False)),
        (compressor, "pool", dict(tokens=capacity, pages=2)),
        (compressor, "commit", dict(tokens=capacity, pages=2)),
        (
            attention_fused,
            "prefill",
            dict(
                rows=capacity, capacity=256, compressed=False, tile_m=64, tile_k=128, reuse_kv=True
            ),
        ),
    ]:
        if a.op and module.__name__.split(".")[-1] + "." + name not in a.op:
            continue
        if (
            name == "gather_attention"
            and a.kv_compressed is not None
            and params["compressed"] != a.kv_compressed
        ):
            continue
        factory = getattr(module, name)
        row_key = "tokens" if "tokens" in params else "rows"
        function = factory(**dict(params, **{row_key: row_count(capacity, True)}))
        tensors = [function.buffer_map[v] for v in function.params if v in function.buffer_map]
        shapes = physical_shapes(function, capacity)
        columns = {t.name: i for i, t in enumerate(tensors)}
        writes = {
            "scatter": {"cache", "scales", "errors"},
            "gather_attention": {"out", "mask"},
            "qk": {"scores"},
            "pv": {"output", "workspace"},
            "pool": {"out", "errors"},
            "commit": {"ring", "errors"},
            "prefill": {"output", "score_tile", "probability_tile", "output_tile"},
        }[name]
        specs = [
            operand(
                t.name, kinds[str(t.dtype)][0], shape, "read_write" if t.name in writes else "read"
            )
            for t, shape in zip(tensors, shapes, strict=True)
        ]
        reference_params = dict(params)
        if a.reuse_long:
            reference_params["reuse_long"] = False
        if module is attention_cube and name == "pv":
            reference_params["tile_d"] = 32
        reference_function = factory(
            **dict(reference_params, **{row_key: row_count(capacity, True)})
        )
        reference = compile_kernel(
            reference_function,
            module.__name__.split(".")[-1] + "." + name,
            specs,
            reference_params,
            a.output / "cache",
            row_capacity=capacity,
        )
        wrapper, _, grid = indirect(function, columns, owners, capacity)
        metadata = dict(
            params,
            _owner_source_sha256=file_hash(Path(module.__file__)),
            _owner_aliases={t.name + "_owner": str(t.dtype) for t in tensors},
            _owner_writes=sorted(writes),
            _owner_access=[dict(column=i, write=t.name in writes) for i, t in enumerate(tensors)],
        )
        private = compile_kernel(
            wrapper,
            "private_rows." + module.__name__.split(".")[-1] + "_" + name,
            [
                operand("OwnerWork", "i32", [8192, 2]),
                operand("OwnerPointers", "u64", [owners, len(tensors) + 1]),
                operand("PacketActive", "i32", [1]),
            ],
            metadata,
            a.output / "cache",
            row_capacity=4096,
        )
        entries = []
        for path in (reference, private):
            lib = ctypes.CDLL(str((path.parent / "kernel.so").resolve()))
            fn = lib.orinfer_ascend_kernel_v2
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int32
            entries.append((lib, fn))

        def call(index, values, rows):
            pointers = (ctypes.c_void_p * len(values))(*[x.data_ptr() for x in values])
            status = entries[index][1](
                pointers, len(values), rows, torch.npu.current_stream()._as_parameter_
            )
            if status:
                raise RuntimeError(f"{name} status {status}")

        data = []
        expected = []
        for owner in range(a.owners):
            values = []
            for tensor, shape in zip(tensors, shapes, strict=True):
                dtype = kinds[str(tensor.dtype)][1]
                if tensor.name in ("swa_indices", "long_indices"):
                    v = torch.arange(shape[1], dtype=dtype).expand(shape[0], -1).clone()
                    if tensor.name == "swa_indices":
                        v += 512 + owner * 31 + torch.arange(shape[0], dtype=dtype)[:, None] - 127
                        bound = params["capacity"]
                    else:
                        bound = params["compressed_capacity"]
                        v = (
                            v + owner * 13 + torch.arange(shape[0], dtype=dtype)[:, None] * 17
                        ) % bound
                    v[:, ::19] = -1
                    v[:, 3] = bound
                    v[:, 4] = -2
                elif tensor.name in ("indices", "positions"):
                    v = torch.arange(shape[0], dtype=dtype) + ((owner * 19 + 13) % 128)
                elif tensor.name == "metadata":
                    v = torch.zeros(shape, dtype=dtype)
                    for row in range(capacity):
                        v[row, :6] = torch.tensor(
                            [
                                ((owner * 19 + 13) % 128) + row,
                                1,
                                row - 1,
                                1,
                                ((owner * 19 + 13) % 128) + capacity - 1,
                                1,
                            ],
                            dtype=dtype,
                        )
                elif tensor.name in writes:
                    v = torch.full(shape, 17, dtype=dtype)
                elif dtype.is_floating_point:
                    v = (torch.randn(shape) * 0.25).to(dtype)
                    if tensor.name in ("swa_scales", "long_scales"):
                        v.flatten()[:8] = torch.tensor(
                            [0, 2**-24, 2**-14, 65504, -(2**-24), -65504, 0.125, 1],
                            dtype=dtype,
                        )
                elif name == "gather_attention" and dtype == torch.int8:
                    v = torch.randint(-127, 128, shape, dtype=dtype)
                else:
                    v = torch.ones(shape, dtype=dtype)
                assert v.dtype == dtype and list(v.shape) == shape, (tensor.name, v.dtype, shape)
                values.append(v.npu())
            data.append(values)
            expected.append([x.clone() for x in values])
        work = torch.empty((8192, 2), dtype=torch.int32, device="npu")
        pointers = torch.empty((owners, len(tensors) + 1), dtype=torch.int64, device="npu")
        active = torch.zeros(1, dtype=torch.int32, device="npu")

        def tables(order, lengths):
            from tools.model.row_contract import resolve

            table = torch.zeros((owners, len(tensors) + 1), dtype=torch.int64)
            items = []
            for lane, (owner, length) in enumerate(zip(order, lengths, strict=True)):
                table[lane] = torch.tensor(
                    [length] + [v.data_ptr() for v in data[owner]], dtype=torch.int64
                )
                items.extend([[lane, t] for t in range(min(resolve(grid, length), 64))])
            if len(items) % 2:
                items.append([-1, 0])
            work[: len(items)].copy_(torch.tensor(items, dtype=torch.int32))
            pointers.copy_(table)
            return len(items) // 2

        order = list(range(a.owners))
        lengths = a.rows or ([1, 3, 9] if a.owners == 3 else [1 + i % capacity for i in order])
        m = tables(order, lengths)
        call(1, [work, pointers, active], m)
        torch.npu.synchronize()
        for owner in data:
            for tensor, value in zip(tensors, owner, strict=True):
                if tensor.name in writes:
                    assert torch.all(value == 17), (name, "disabled capture")
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            call(1, [work, pointers, active], m)
        active.fill_(1)
        sequences = [(order, lengths), (list(reversed(order)), lengths)]
        if name == "gather_attention" or module is attention_cube:
            sequences.append((list(reversed(order)), lengths))
        for phase, (order, lengths) in enumerate(sequences):
            assert tables(order, lengths) == m
            if name == "gather_attention" and phase == 2:
                # Scatter/quantization updates these addresses between model
                # steps. Exercise Scalar scale loads after actual GM updates.
                for owner in range(a.owners):
                    for tensor, value, want in zip(
                        tensors, data[owner], expected[owner], strict=True
                    ):
                        if tensor.name in ("swa", "long_kv"):
                            value.neg_()
                            want.copy_(value)
                        elif tensor.name in ("swa_scales", "long_scales"):
                            value.mul_(0.5)
                            want.copy_(value)
            if module is attention_cube and phase == 2:
                for owner in range(a.owners):
                    for tensor, value, want in zip(
                        tensors, data[owner], expected[owner], strict=True
                    ):
                        if tensor.name == "kv":
                            value.neg_()
                            want.copy_(value)
                        elif tensor.name == "probabilities":
                            value.mul_(0.5)
                            want.copy_(value)
            for owner in range(a.owners):
                for tensor, value, want in zip(tensors, data[owner], expected[owner], strict=True):
                    if tensor.name in writes:
                        value.fill_(17)
                        want.fill_(17)
            if a.reuse_long:
                # The previous layer initializes all long slots. Its successor
                # changes only SWA and must preserve those converted long rows.
                for owner, length in zip(order, lengths, strict=True):
                    call(0, data[owner], length)
                for owner in range(a.owners):
                    for tensor, value, want in zip(
                        tensors, data[owner], expected[owner], strict=True
                    ):
                        if tensor.name == "swa":
                            value.neg_()
                            want.copy_(value)
                        elif tensor.name == "swa_scales":
                            value.mul_(0.5)
                            want.copy_(value)
            for owner, length in zip(order, lengths, strict=True):
                call(0, expected[owner], length)
            graph.replay()
            torch.npu.synchronize()
            if module is attention_cube:
                by_name = {tensor.name: i for i, tensor in enumerate(tensors)}
                for owner, length in zip(order, lengths, strict=True):
                    if owner not in (order[0], order[-1]):
                        continue
                    probability = (
                        data[owner][by_name["probabilities"]][:length, : params["heads"]]
                        .cpu()
                        .double()
                    )
                    kv = data[owner][by_name["kv"]][:length].cpu().double()
                    wanted = (probability @ kv).to(torch.bfloat16)
                    actual = data[owner][by_name["output"]][:length].cpu()
                    torch.testing.assert_close(actual, wanted, atol=0.02, rtol=0.008)
            if name == "gather_attention":
                by_name = {tensor.name: i for i, tensor in enumerate(tensors)}
                for owner, length in zip(order, lengths, strict=True):
                    source = {
                        key: data[owner][i].cpu() for key, i in by_name.items() if key not in writes
                    }
                    decoded, masks = [], []
                    for cache_name, scale_name, ids_name, limit, circular in [
                        ("swa", "swa_scales", "swa_indices", params["capacity"], True),
                        (
                            "long_kv",
                            "long_scales",
                            "long_indices",
                            params["compressed_capacity"],
                            False,
                        ),
                    ]:
                        if not circular and not params["compressed"]:
                            continue
                        ids = source[ids_name][:length].long()
                        valid = (ids >= 0) & (ids < limit)
                        physical = (ids.clamp_min(0) % 256) if circular else ids.clamp(0, limit - 1)
                        values = source[cache_name][physical].float()
                        scales = source[scale_name][physical].float()
                        values = (values * scales[..., None]).bfloat16()
                        values[~valid] = 0
                        decoded.append(values)
                        masks.append(torch.where(valid, 0, -1).int())
                    actual = data[owner][by_name["out"]][:length].cpu()
                    wanted = torch.cat(decoded, 1).reshape(actual.shape)
                    if not torch.equal(actual, wanted):
                        locations = (actual != wanted).nonzero()[:8]
                        print(
                            dict(
                                parameters=params,
                                owner=owner,
                                length=length,
                                mismatch=[
                                    dict(
                                        at=at.tolist(),
                                        actual=float(actual[tuple(at)]),
                                        expected=float(wanted[tuple(at)]),
                                    )
                                    for at in locations
                                ],
                                same_as_direct=torch.equal(
                                    actual, expected[owner][by_name["out"]][:length].cpu()
                                ),
                                mask=data[owner][by_name["mask"]][0, 128:144].cpu().tolist(),
                                ids=source["long_indices"][0, :16].tolist(),
                            ),
                            flush=True,
                        )
                    assert torch.equal(actual, wanted)
                    assert torch.equal(
                        data[owner][by_name["mask"]][:length].cpu(), torch.cat(masks, 1)
                    )
            for owner in range(a.owners):
                for tensor, value, want in zip(tensors, data[owner], expected[owner], strict=True):
                    if tensor.name in writes:
                        if name == "prefill" and tensor.name != "output":
                            # Padded scratch rows do not carry semantic values.
                            continue
                        actual_cpu, expected_cpu = value.cpu(), want.cpu()
                        if not torch.equal(actual_cpu, expected_cpu):
                            print(
                                dict(
                                    op=name,
                                    owner=owner,
                                    tensor=tensor.name,
                                    actual_finite=bool(torch.isfinite(actual_cpu).all()),
                                    expected_finite=bool(torch.isfinite(expected_cpu).all()),
                                    max_abs=float(
                                        (actual_cpu.float() - expected_cpu.float()).abs().max()
                                    ),
                                    actual=actual_cpu.flatten()[:16].float().tolist(),
                                    expected=expected_cpu.flatten()[:16].float().tolist(),
                                ),
                                flush=True,
                            )
                        assert torch.equal(actual_cpu, expected_cpu), (name, owner, tensor.name)
        del graph
        timing = None
        if name == "gather_attention":
            from tools.model.deepseek_v41.tune import npu_benchmark

            timing, measured = npu_benchmark(lambda: call(1, [work, pointers, active], m), 7, 2)
            del measured
        measurements = []
        identity = None
        if a.benchmark_rows:
            from tools.model.deepseek_v41.tune import npu_benchmark
            from tools.model.tuning import measure, read_record, record_path

            identity = dict(
                manifest_sha256=file_hash(private),
                rows=sorted(set(a.benchmark_rows)),
                device=torch.npu.get_device_name(a.device),
                torch_version=str(torch.__version__),
                seed=20261002,
                budget_ms=25,
                max_repetitions=11,
                measurement_source_sha256=measurement_source(),
            )
            record = record_path(a.output / "tuning", identity)
            record.parent.mkdir(parents=True, exist_ok=True)
            cached = read_record(record, identity)
            if cached is not None:
                measurements = cached["measurements"]
            else:
                for length in identity["rows"]:
                    live = tables([0], [length])
                    elapsed = measure(
                        lambda: call(1, [work, pointers, active], live),
                        budget_ms=identity["budget_ms"],
                        max_repetitions=identity["max_repetitions"],
                        benchmark=npu_benchmark,
                    )
                    measurements.append(dict(rows=length, milliseconds=elapsed))
                atomic_json(record, dict(identity=identity, measurements=measurements))
        report.append(
            dict(
                op=module.__name__.split(".")[-1] + "." + name,
                owners=a.owners,
                lengths=sorted(set(lengths)),
                changed_owners=True,
                capture_disabled=True,
                graph_replay=True,
                exact_reference=True,
                reference_manifest=str(reference),
                private_manifest=str(private),
                source_sha256=file_hash(Path(module.__file__)),
                parameters=params,
                graph_timing=timing,
                measurements=measurements,
                measurement_identity=identity,
            )
        )
        print(report[-1], flush=True)
    if not report:
        raise ValueError("No primitive matched --op")
    atomic_json(a.output / "result.json", report)


if __name__ == "__main__":
    main()
