"""Validate indirect A2 owner pointers, tails, capture guards and changed Graph data."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    p.add_argument("--owners", type=int, default=3)
    a = p.parse_args()
    if not 1 <= a.owners <= 128:
        p.error("owners must be in 1..128")
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.operators.ascend import attention_prefill, kv_int8, compressor
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.operators.ascend.geometry import physical_shapes
    from tools.operators.ascend.private_rows import indirect

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(a.device)
    capacity, owners = 16, 128
    kinds = {
        "int32": ("i32", torch.int32),
        "int8": ("i8", torch.int8),
        "float16": ("f16", torch.float16),
        "bfloat16": ("bf16", torch.bfloat16),
        "float32": ("f32", torch.float32),
    }
    report = []
    for module, name, params in [
        (kv_int8, "scatter", dict(rows=capacity, table_rows=256)),
        (attention_prefill, "qk", dict(rows=capacity, capacity=256, compressed=False)),
        (attention_prefill, "pv", dict(rows=capacity, capacity=256, compressed=False)),
        (compressor, "pool", dict(tokens=capacity, pages=2)),
        (compressor, "commit", dict(tokens=capacity, pages=2)),
    ]:
        factory = getattr(module, name)
        row_key = "tokens" if "tokens" in params else "rows"
        function = factory(**dict(params, **{row_key: row_count(capacity, True)}))
        tensors = [function.buffer_map[v] for v in function.params if v in function.buffer_map]
        shapes = physical_shapes(function, capacity)
        columns = {t.name: i for i, t in enumerate(tensors)}
        writes = {
            "scatter": {"cache", "scales", "errors"},
            "qk": {"scores"},
            "pv": {"output", "workspace"},
            "pool": {"out", "errors"},
            "commit": {"ring", "errors"},
        }[name]
        specs = [
            operand(
                t.name, kinds[str(t.dtype)][0], shape, "read_write" if t.name in writes else "read"
            )
            for t, shape in zip(tensors, shapes, strict=True)
        ]
        reference = compile_kernel(
            function,
            module.__name__.split(".")[-1] + "." + name,
            specs,
            params,
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
                if tensor.name in ("indices", "positions"):
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
                else:
                    v = torch.ones(shape, dtype=dtype)
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
        lengths = [1, 3, 9] if a.owners == 3 else [1 + i % capacity for i in order]
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
        for order, lengths in [(order, lengths), (list(reversed(order)), lengths)]:
            assert tables(order, lengths) == m
            for owner in range(a.owners):
                for tensor, value, want in zip(tensors, data[owner], expected[owner], strict=True):
                    if tensor.name in writes:
                        value.fill_(17)
                        want.fill_(17)
            for owner, length in zip(order, lengths, strict=True):
                call(0, expected[owner], length)
            graph.replay()
            torch.npu.synchronize()
            for owner in range(a.owners):
                for tensor, value, want in zip(tensors, data[owner], expected[owner], strict=True):
                    if tensor.name in writes:
                        assert torch.equal(value.cpu(), want.cpu()), (name, owner, tensor.name)
        del graph
        report.append(
            dict(
                op=module.__name__.split(".")[-1] + "." + name,
                owners=a.owners,
                lengths=sorted(set(lengths)),
                changed_owners=True,
                capture_disabled=True,
                graph_replay=True,
                exact_reference=True,
            )
        )
        print(report[-1], flush=True)
    atomic_json(a.output / "result.json", report)


if __name__ == "__main__":
    main()
