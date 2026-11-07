"""Overlay image features on CPU-prepared text embeddings, including MTP shift."""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count


@orin_jit
def overlay(
    rows: int,
    hidden: int,
    capacity: int,
    features: int,
    dynamic_rows: bool = False,
):
    if not 1 <= rows <= capacity or hidden <= 0 or features <= 0:
        raise ValueError("Invalid visual embedding dimensions")

    rows = row_count(rows, dynamic_rows)

    @T.prim_func
    def main(
        Embedding: T.Tensor((rows, hidden), T.float16),
        Features: T.Tensor((features, hidden), T.float16),
        Index: T.Tensor((capacity,), T.int32),
        Position: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(rows, threads=128) as row:
            token = Position[0] + row
            if token >= 0 and token < capacity:
                feature = Index[token]
                if feature >= 0 and feature < features:
                    for column in T.Parallel(hidden):
                        Embedding[row, column] = Features[feature, column]

    return main
