"""Dynamic row copies and owner-local length controls for causal segments."""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count


@orin_jit
def copy(capacity, width, dtype="float16"):
    rows = row_count(capacity, True)

    @T.prim_func
    def main(Source: T.Tensor((rows, width), dtype), Destination: T.Tensor((rows, width), dtype)):
        with T.Kernel(rows, T.ceildiv(width, 256), threads=128) as (row, block):
            for j in T.Parallel(256):
                col = block * 256 + j
                if col < width:
                    Destination[row, col] = Source[row, col]

    return main


@orin_jit
def length(capacity=512):
    rows = row_count(capacity, True)

    @T.prim_func
    def main(Length: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                Length[0] = rows

    from kernels.shared.rows import explicit_rows

    return explicit_rows(main)
