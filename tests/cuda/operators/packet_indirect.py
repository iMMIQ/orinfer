"""Exercise unchanged TileLang bodies through replaceable private address tables."""

import argparse
from pathlib import Path
import torch
import tilelang.language as T
from kernels.cuda.model.independent_rows import indirect_parameters
from kernels.cuda.model.packet_io import copy_rows
from orinfer_build.compiler.compile import compile_function, instantiate
from orinfer_build.compiler.cuda.common import configure, write_json


def arithmetic():
    m = T.dynamic("m")

    @T.prim_func
    def kernel(
        X: T.Tensor((m,), T.float32),
        State: T.Tensor((m + 7,), T.float32),
        Y: T.Tensor((m,), T.float32),
    ):
        with T.Kernel(T.ceildiv(m, 128), threads=128) as block:
            Live = T.decl_buffer((m,), T.float32, data=State.data, elem_offset=7)
            for i in T.Parallel(128):
                index = block * 128 + i
                if index < m:
                    Y[index] = X[index] + Live[index]
                    Live[index] = Live[index] + 1

    from kernels.shared.rows import explicit_rows

    return explicit_rows(kernel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    columns = {"unused0": 0, "unused1": 1, "state": 2, "unused3": 3}
    fn, inputs = indirect_parameters(arithmetic(), columns, {"State": "state"})
    op = compile_function(fn)
    x = torch.arange(263, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    states = [torch.full((270,), float(i * 10), device="cuda") for i in range(3)]
    table = torch.tensor([0, 0, states[0].data_ptr(), 0], dtype=torch.uint64, device="cuda")
    values = {"X": x, "Y": y, None: table, "m": 263, "column_State": 2}

    def run():
        op.adapter.func(
            *(values[name] for name in inputs), stream=torch.cuda.current_stream().cuda_stream
        )

    run()
    states[0].zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for owner in [0, 2, 1, 0]:
        before = [s.clone() for s in states]
        table.copy_(torch.tensor([0, 0, states[owner].data_ptr(), 0], dtype=torch.uint64))
        x.add_(3)
        graph.replay()
        assert torch.equal(y, x + before[owner][7:])
        for i, state in enumerate(states):
            expected = before[i]
            if i == owner:
                expected[7:].add_(1)
            assert torch.equal(state, expected)
    # Exercise byte tails and aligned rows, including partial tiles.
    copy_cases = []
    for width in [259, 2051, 4112, 993280]:
        size = 3 * width
        fn, inputs = indirect_parameters(
            instantiate(copy_rows, width), {"out": 0}, {"Destination": "out"}
        )
        copy = compile_function(fn)
        source_storage = torch.arange(size + 32, device="cuda").to(torch.uint8)
        source = source_storage[16 : size + 16]
        targets = [
            torch.full((size + 32,), fill, device="cuda", dtype=torch.uint8) for fill in [17, 29]
        ]
        views = [target[16 : size + 16] for target in targets]
        pointers = torch.tensor([views[0].data_ptr()], device="cuda", dtype=torch.uint64)
        values = {None: pointers, "Source": source, "m": 3, "column_Destination": 0}

        def run_copy():
            copy.adapter.func(
                *(values[name] for name in inputs), stream=torch.cuda.current_stream().cuda_stream
            )

        run_copy()
        copy_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(copy_graph):
            run_copy()
        pointers.copy_(torch.tensor([views[1].data_ptr()], dtype=torch.uint64))
        source.add_(7)
        copy_graph.replay()
        assert torch.equal(views[1], source)
        assert torch.equal(views[0], source - 7)
        for target, fill in zip(targets, [17, 29]):
            assert bool((target[:16] == fill).all() and (target[-16:] == fill).all())
        copy_cases.append(dict(width=width, bytes=size))
    write_json(
        args.output / "results.json",
        dict(complete=True, owners=[0, 2, 1, 0], rows=263, private_offset=7, copy_cases=copy_cases),
    )


if __name__ == "__main__":
    main()
