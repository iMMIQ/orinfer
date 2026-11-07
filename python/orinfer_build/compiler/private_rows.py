"""Shared offline binding of request-owned buffers to dynamic row kernels."""

import math


def wrap_private(function, capacity, modes, columns, pointers, table, *, table_offset=0, **options):
    from kernels.cuda.model.independent_rows import independent_rows

    function, inputs = independent_rows(
        function,
        capacity,
        modes,
        {name: i for i, name in enumerate(columns)},
        dynamic=True,
        **options,
    )
    bindings = {name: pointers[name] for name in inputs if name not in (None, "__rows__")}
    bindings["Pointers"] = dict(kind="buffer_slice", name=table, offset=table_offset)
    return function, bindings


class PrivateRows:
    def __init__(self, weights, row_buffers, scratch, identity, capacity, layers, columns=()):
        self.weights, self.row_buffers, self.scratch = weights, set(row_buffers), scratch
        self.identity, self.capacity, self.layers = identity, capacity, layers
        self.columns = set(columns) | {scratch}
        self.sizes, self.offsets = {}, {}

    def bind(self, function, pointers, *, extents=None):
        from tvm import tirx as tir
        from tvm.arith import Analyzer

        used = set()

        def visit(node):
            if isinstance(node, (tir.BufferLoad, tir.BufferStore)):
                used.add(node.buffer.data)
            if isinstance(node, tir.Var):
                used.add(node)

        tir.stmt_functor.post_order_visit(function.body, visit)
        normalized, modes = dict(pointers), {}
        for buf in function.buffer_map.values():
            if buf.data not in used:
                continue
            arg = pointers.get(buf.name) or pointers.get("live_" + buf.name.lower())
            if arg is None:
                raise ValueError(f"Missing private buffer pointer {buf.name}")
            name = arg["name"]
            normalized[buf.name] = arg
            storage = self.weights.buffers[name]
            if storage.get("data") is not None:
                mode, identity = "weight", name
            elif name in self.row_buffers:
                mode, identity = "row", name
            elif self.weights.scopes[name] == "sequence":
                mode, identity = "private", self.identity(name)
                self.columns.add(identity)
            else:
                mode, identity = "private", self.scratch
                shape = tir.stmt_functor.substitute(math.prod(buf.shape), extents or {})
                size = (
                    int(Analyzer().simplify(shape))
                    * {
                        "f16": 2,
                        "bf16": 2,
                        "f32": 4,
                        "i8": 1,
                        "u8": 1,
                        "i32": 4,
                        "u32": 4,
                        "i64": 8,
                        "u64": 8,
                    }[storage["dtype"]]
                )
                self.sizes[name] = max(self.sizes.get(name, 0), size)
            modes[buf.name] = mode, identity
        return modes, normalized

    def layout(self, *, first=()):
        cursor = 0
        self.offsets = {}
        for name in sorted(self.sizes, key=lambda name: (name not in first, name)):
            self.offsets[name] = cursor
            cursor += (self.sizes[name] + 255) // 256 * 256
        self.columns = sorted(self.columns)
        return self.offsets, cursor, self.columns

    def wrap(self, function, modes, pointers, layer, table, **options):
        return wrap_private(
            function,
            self.capacity,
            modes,
            self.columns,
            pointers,
            table,
            table_offset=layer * len(self.columns) * 8,
            row_stride=self.layers * len(self.columns),
            offsets={
                name: self.offsets[arg["name"]]
                for name, arg in pointers.items()
                if arg["name"] in self.offsets
            },
            **options,
        )
