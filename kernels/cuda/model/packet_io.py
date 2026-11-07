"""Single-owner packet boundaries, lifted by the shared dynamic-row wrapper."""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit


def output_fragments(vocab, *, batched=False):
    """Share full-logit and selection writes across direct and owner-table I/O."""

    @T.macro
    def logits(Source, Logits, row, block):
        for column in T.Parallel(512):
            index = block * 512 + column
            if index < vocab:
                if batched:
                    Logits[index] = Source[row, index]
                else:
                    Logits[index] = Source[index]

    @T.macro
    def selected(Pair, Token, Status, row):
        if batched:
            Token[0] = Pair[row, 0]
            Status[0] = Pair[row, 1]
        else:
            Token[0] = Pair[0]
            Status[0] = Pair[1]

    return logits, selected


@orin_jit
def token_input():
    @T.prim_func
    def kernel(Token: T.Tensor((1,), T.int32), Input: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                Input[0] = Token[0]

    return kernel


@orin_jit
def outputs(vocab):
    write_logits, write_selected = output_fragments(vocab)

    @T.prim_func
    def kernel(
        Source: T.Tensor((vocab,), T.float32),
        Pair: T.Tensor((2,), T.int32),
        Logits: T.Tensor((vocab,), T.float32),
        Token: T.Tensor((1,), T.int32),
        Status: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(T.ceildiv(vocab, 512), threads=128) as block:
            write_logits(Source, Logits, 0, block)
            if block == 0 and T.get_thread_binding() == 0:
                write_selected(Pair, Token, Status, 0)

    return kernel


@orin_jit
def copy_rows(width):
    rows = T.dynamic("m")

    @T.prim_func
    def kernel(
        Source: T.Tensor((rows, width), T.uint8), Destination: T.Tensor((rows, width), T.uint8)
    ):
        with T.Kernel(T.ceildiv(rows * width, 256), threads=128) as block:
            for i in T.Parallel(256):
                offset = block * 256 + i
                if offset < rows * width:
                    Destination[offset // width, offset % width] = Source[
                        offset // width, offset % width
                    ]

    from kernels.shared.rows import explicit_rows

    return explicit_rows(kernel)
