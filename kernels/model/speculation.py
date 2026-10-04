"""Prefix state selection and MTP input normalization on caller-owned buffers."""
import tilelang.language as T
from tools.operators.common import orin_jit


@orin_jit
def select_prefix(tokens: int, elements: int, dtype='float32'):
    """Commit one saved prefix. Caller guarantees 1 <= Count[0] <= tokens."""
    assert tokens > 0 and elements > 0 and dtype in ('float16', 'float32')

    @T.prim_func
    def kernel(Prefix: T.Tensor((tokens, elements), dtype),
               Count: T.Tensor((1,), T.int32),
               State: T.Tensor((elements,), dtype)):
        with T.Kernel(T.ceildiv(elements, 256), threads=256) as b:
            for j in T.Parallel(256):
                index = b * 256 + j
                if index < elements:
                    State[index] = Prefix[Count[0] - 1, index]
    return kernel


@orin_jit
def conv_history_prefixes(tokens: int, channels: int = 10240):
    """Save chronological raw width-4 convolution history for every prefix.

    Snapshot before the regular convolution overwrites History. Logical
    position zero ignores old history, exactly like gdn_conv_prep.
    """
    assert tokens > 0 and channels > 0

    @T.prim_func
    def kernel(X: T.Tensor((tokens, channels), T.float16),
               History: T.Tensor((3, channels), T.float16),
               Position: T.Tensor((1,), T.int32),
               Prefix: T.Tensor((tokens, 3, channels), T.float16)):
        with T.Kernel(T.ceildiv(3 * channels, 256), tokens, threads=256) as (b, t):
            for j in T.Parallel(256):
                index = b * 256 + j
                if index < 3 * channels:
                    tap, channel = index // channels, index % channels
                    source = t + 1 + tap - 3
                    Prefix[t, tap, channel] = 0.0
                    if source >= 0:
                        Prefix[t, tap, channel] = X[source, channel]
                    elif Position[0] + source >= 0:
                        Prefix[t, tap, channel] = History[source + 3, channel]
    return kernel


@orin_jit
def mtp_norm_concat(hidden: int, epsilon: float = 1e-6):
    """Normalize [embedding, normalized target hidden] then concatenate.

    Target is the target model's FINAL normalized hidden output, as passed
    to the native Qwen3_5 MTP model; it is not the pre-final residual sum.
    Norm weights are zero centered. The concatenation order is embedding
    first, hidden second. All input and output buffers must be disjoint.
    """
    assert hidden > 0 and epsilon > 0
    rows = T.dynamic('rows')

    @T.prim_func
    def kernel(Embedding: T.Tensor((rows, hidden), T.float16),
               Target: T.Tensor((rows, hidden), T.float16),
               EmbeddingWeight: T.Tensor((hidden,), T.float16),
               HiddenWeight: T.Tensor((hidden,), T.float16),
               Out: T.Tensor((rows, 2 * hidden), T.float16)):
        with T.Kernel(rows, 2, threads=256) as (r, half):
            values = T.alloc_fragment((hidden,), T.float32)
            squares = T.alloc_fragment((hidden,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(hidden):
                if half == 0:
                    values[j] = T.cast(Embedding[r, j], T.float32)
                else:
                    values[j] = T.cast(Target[r, j], T.float32)
                squares[j] = values[j] * values[j]
            T.reduce_sum(squares, total, dim=0)
            for j in T.Parallel(hidden):
                weight = T.if_then_else(half == 0, T.cast(EmbeddingWeight[j], T.float32),
                                        T.cast(HiddenWeight[j], T.float32))
                Out[r, half * hidden + j] = values[j] * T.rsqrt(total[0] / hidden + epsilon) * (1.0 + weight)
    return kernel


@orin_jit
def capture_target_hidden(hidden: int, max_context: int, epsilon: float = 1e-6, ring: bool = False):
    """Store finalized target hidden, optionally in a chunk-sized ring.

    Step is the position AFTER the target graph. The scheduler captures its
    exact active row count before another graph overwrites Hidden/residual.
    The reduction/rounding matches op23_final_norm; no FP16 residual pre-sum.
    Ring captures must fit the allocation. Consume a chunk before overwriting
    its slots; the final prompt chunk waits for the target's first token.
    """
    assert hidden > 0 and max_context > 0 and epsilon > 0
    rows = T.dynamic('rows')

    @T.prim_func
    def kernel(X: T.Tensor((rows, hidden), T.float16),
               Residual: T.Tensor((rows, hidden), T.float32),
               Weight: T.Tensor((hidden,), T.float16),
               Step: T.Tensor((1,), T.int32),
               Out: T.Tensor((max_context, hidden), T.float16)):
        with T.Kernel(rows, threads=256) as row:
            values = T.alloc_fragment((hidden,), T.float32)
            squares = T.alloc_fragment((hidden,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(hidden):
                values[j] = T.cast(X[row, j], T.float32) + Residual[row, j]
                squares[j] = values[j] * values[j]
            T.reduce_sum(squares, total, dim=0)
            for j in T.Parallel(hidden):
                position = Step[0] - rows + row
                slot = position % max_context if ring else position
                Out[slot, j] = values[j] * T.rsqrt(total[0] / hidden + epsilon) * (1.0 + T.cast(Weight[j], T.float32))
    return kernel


@orin_jit
def gather_target_hidden(tokens: int, hidden: int, max_context: int, ring: bool = False):
    """Pair target h[t] with x[t+1]; ring slots must still hold this prefix."""
    assert tokens > 0 and hidden > 0 and max_context > 0

    @T.prim_func
    def kernel(Target: T.Tensor((max_context, hidden), T.float16),
               Step: T.Tensor((1,), T.int32),
               Out: T.Tensor((tokens, hidden), T.float16)):
        with T.Kernel(T.ceildiv(hidden, 256), tokens, threads=256) as (b, t):
            for j in T.Parallel(256):
                index = b * 256 + j
                if index < hidden:
                    position = Step[0] + t
                    slot = position % max_context if ring else position
                    Out[t, index] = Target[slot, index]
    return kernel
