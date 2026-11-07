"""Device-resident single-sequence position control for graph replay."""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count


@orin_jit
def prepare(rows: int):
    @T.prim_func
    def kernel(
        Step: T.Tensor((1,), T.int32),
        Positions: T.Tensor((rows,), T.int32),
        SeqLength: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(T.ceildiv(rows, 256), threads=256) as block:
            i = block * 256 + T.get_thread_binding()
            if i < rows:
                Positions[i] = Step[0] + i
            if i == 0:
                SeqLength[0] = Step[0] + rows

    return kernel


@orin_jit
def advance(rows: int, dynamic_rows: bool = False):

    rows = count = row_count(rows, dynamic_rows)

    @T.prim_func
    def kernel(Step: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                Step[0] = Step[0] + count

    if dynamic_rows:
        from kernels.shared.rows import explicit_rows

        kernel = explicit_rows(kernel)
    return kernel
