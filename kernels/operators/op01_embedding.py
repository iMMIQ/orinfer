"""SM87 embedding gather. Runtime tensors are caller-owned, no workspace.

The raw device ABI assumes validated IDs; scheduler must reject invalid IDs
before upload and after changes. Stable addresses and explicit current stream
are required for graph capture/replay. BF16 -> FP16 is a numeric conversion.
"""
import tilelang.language as T
from tools.operators.common import orin_jit


def validate_token_ids(ids, vocab: int, rows: int | None = None):
    """CPU rejection boundary; duplicates and unordered IDs are legal."""
    if type(vocab) is not int or not 0 < vocab <= 2147483647:
        raise ValueError("vocab must be a positive int32 dimension")
    if not isinstance(ids, (tuple, list)) or not ids or len(ids) > 2147483647:
        raise ValueError("ids must be a nonempty CPU list/tuple of int32 IDs")
    if rows is not None and (type(rows) is not int or rows != len(ids)):
        raise ValueError("rows must equal the number of IDs")
    if any(type(i) is not int or not 0 <= i < vocab for i in ids):
        raise ValueError("every token ID must be an integer in [0,vocab)")
    return tuple(ids)


@orin_jit
def embedding_gather(vocab: int = 248320, hidden: int = 5120,
                     dtype: str = "float16", stride: int | None = None,
                     block: int = 1024, threads: int = 128):
    """Build (W, I, Y): row-major W[V,stride], int32 I[M], FP16 Y[M,H].

    FP16 copies exactly, BF16 casts to FP16 (round-to-nearest-even). W/I/Y
    disjoint; positive int32 M. Only selected rows and H columns are read.
    """
    stride = hidden if stride is None else stride
    assert 0 < vocab <= 2147483647 and 0 < hidden <= stride
    assert dtype in ("float16", "bfloat16")
    assert block > 0 and block % threads == 0 and threads in (128, 256)
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(W: T.Tensor((vocab, stride), dtype),
               I: T.Tensor((rows,), T.int32),
               Y: T.Tensor((rows, hidden), T.float16)):
        with T.Kernel(T.ceildiv(rows * hidden, block), threads=threads) as bx:
            for j in T.Parallel(block):
                index = bx * block + j
                if index < rows * hidden:
                    Y[index // hidden, index % hidden] = T.cast(
                        W[I[index // hidden], index % hidden], T.float16)
    return kernel


@orin_jit
def embedding_u4(vocab: int = 248320, hidden: int = 5120,
                 group: int = 128, block: int = 512, threads: int = 128):
    """Build (P,S,Z,I,Y); unpack only requested rows, no whole-table expansion.

    Kp=ceildiv(H,group)*group. P[V,Kp/2] uint8 stores adjacent low/high U4;
    S[V,Kp/group] FP16, Z same shape int8 (values 0..15), Y[M,H] FP16.
    Math: half((float32(q)-float32(z))*float32(s)), one final rounding.
    Each lane reads one packed pair; odd H and group padding are masked.
    """
    assert 0 < vocab <= 2147483647 and hidden > 0
    assert group > 0 and group % 2 == 0
    assert block > 0 and block % threads == 0 and threads in (128, 256)
    padded = ((hidden + group - 1) // group) * group
    pairs = (hidden + 1) // 2
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(P: T.Tensor((vocab, padded // 2), T.uint8),
               S: T.Tensor((vocab, padded // group), T.float16),
               Z: T.Tensor((vocab, padded // group), T.int8),
               I: T.Tensor((rows,), T.int32),
               Y: T.Tensor((rows, hidden), T.float16)):
        with T.Kernel(T.ceildiv(rows * pairs, block), threads=threads) as bx:
            for j in T.Parallel(block):
                index = bx * block + j
                if index < rows * pairs:
                    row = index // pairs
                    col = (index % pairs) * 2
                    token = I[row]
                    packed = T.cast(P[token, col // 2], T.int32)
                    zero = T.cast(Z[token, col // group], T.float32)
                    scale = T.cast(S[token, col // group], T.float32)
                    Y[row, col] = T.cast((T.cast(packed & 15, T.float32) - zero) * scale, T.float16)
                    if col + 1 < hidden:
                        Y[row, col + 1] = T.cast((T.cast(packed >> 4, T.float32) - zero) * scale, T.float16)
    return kernel
