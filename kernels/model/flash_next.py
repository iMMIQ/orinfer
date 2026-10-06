"""Flash Next dense projections, sigmoid GDN epilogue and QSA preparation.

Floating projections handle critical weights and reference probes; ordinary
native INT8 projections use int8_projection.py and packed E8P experts use
integer_vq.py. qsa_short_attention remains a bounded operator oracle.
Native long QSA indexing, INT8 KV and sparse attention are implemented in
qsa.py and qsa_attention.py; prepared FP16 KV tiles are only temporary.
"""
import tilelang.language as T

from tools.operators.common import orin_jit


@orin_jit
def dense_projection(M: int, N: int, K: int, weight_dtype: str = 'float16',
                     output_dtype: str = 'float16'):
    """FP16 input, FP16/BF16 weights, FP32 accumulation; optional F32 output."""
    if any(type(v) is not int or v <= 0 for v in (M, N, K)) or K % 64:
        raise ValueError('Invalid dense projection dimensions')
    if weight_dtype not in ('float16', 'bfloat16') or output_dtype not in ('float16', 'float32'):
        raise ValueError('Invalid dense projection dtype')
    @T.prim_func
    def main(A: T.Tensor((M, K), T.float16),
             Weight: T.Tensor((N, K), weight_dtype),
             Output: T.Tensor((M, N), output_dtype)):
        with T.Kernel(T.ceildiv(M, 16), T.ceildiv(N, 64), threads=128) as (by, bx):
            a = T.alloc_shared((16, 64), weight_dtype)
            w = T.alloc_shared((64, 64), weight_dtype)
            acc = T.alloc_fragment((16, 64), T.float32)
            T.clear(acc)
            for kg in T.Pipelined(K // 64, num_stages=2):
                for i, j in T.Parallel(16, 64, coalesced_width=T.int32(1)):
                    a[i, j] = T.if_then_else(by * 16 + i < M, A[by * 16 + i, kg * 64 + j], 0.0)
                T.copy(Weight[bx * 64, kg * 64], w)
                T.gemm(a, w, acc, transpose_B=True)
            T.copy(acc, Output[by * 16, bx * 64])
    return main


@orin_jit
def hc_initialize(M: int, H: int = 2560, streams: int = 4):
    """Broadcast token embeddings into every independent residual stream."""
    if any(type(v) is not int or v <= 0 for v in (M, H, streams)):
        raise ValueError('Invalid residual dimensions')
    @T.prim_func
    def main(Embedding: T.Tensor((M, H), T.float16),
             Residual: T.Tensor((M, streams, H), T.float16)):
        with T.Kernel(M, streams, T.ceildiv(H, 256), threads=128) as (row, branch, block):
            for j in T.Parallel(256):
                if block * 256 + j < H:
                    Residual[row, branch, block * 256 + j] = Embedding[row, block * 256 + j]
    return main


@orin_jit
def residual_add(M: int, C: int):
    """FP16 residual update, with one owner per element; Output may alias X."""
    if any(type(v) is not int or v <= 0 for v in (M, C)):
        raise ValueError('Invalid residual dimensions')
    @T.prim_func
    def main(X: T.Tensor((M, C), T.float16),
             Update: T.Tensor((M, C), T.float16),
             Output: T.Tensor((M, C), T.float16)):
        with T.Kernel(T.ceildiv(M * C, 256), threads=128) as block:
            for j in T.Parallel(256):
                index = block * 256 + j
                if index < M * C:
                    Output[index // C, index % C] = T.cast(X[index // C, index % C], T.float32) + T.cast(Update[index // C, index % C], T.float32)
    return main


@orin_jit
def swiglu(M: int, width: int):
    """Separate FP16 gate/up projections into a FP16 shared-expert activation."""
    if any(type(v) is not int or v <= 0 for v in (M, width)):
        raise ValueError('Invalid SwiGLU dimensions')
    @T.prim_func
    def main(Gate: T.Tensor((M, width), T.float16),
             Up: T.Tensor((M, width), T.float16),
             Output: T.Tensor((M, width), T.float16)):
        with T.Kernel(T.ceildiv(M * width, 256), threads=128) as block:
            for j in T.Parallel(256):
                index = block * 256 + j
                if index < M * width:
                    v = T.cast(Gate[index // width, index % width], T.float32)
                    Output[index // width, index % width] = v / (1.0 + T.exp(-v)) * T.cast(Up[index // width, index % width], T.float32)
    return main


@orin_jit
def gdn_sigmoid_norm(M: int, heads: int = 48, width: int = 128, eps: float = 1e-6):
    """(X[M,heads,width], Z, Weight[width] F32, Y), FP16 activations.

    Weight is ordinary RMSNorm gamma, not zero-centered gamma. The output
    gate is sigmoid; reusing Qwen3.5's SiLU gate changes model semantics.
    """
    if any(type(v) is not int or v <= 0 for v in (M, heads, width)) or not 0 < eps < 1:
        raise ValueError('Invalid GDN norm dimensions')
    size = 1 << (width - 1).bit_length()
    @T.prim_func
    def main(X: T.Tensor((M, heads, width), T.float16),
             Z: T.Tensor((M, heads, width), T.float16),
             Weight: T.Tensor((width,), T.float32),
             Y: T.Tensor((M, heads, width), T.float16)):
        with T.Kernel(M, heads, threads=128) as (row, head):
            x = T.alloc_fragment((size,), T.float32)
            square = T.alloc_fragment((size,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(size):
                x[j] = T.if_then_else(j < width, T.cast(X[row, head, j], T.float32), 0.0)
                square[j] = x[j] * x[j]
            T.reduce_sum(square, total, dim=0)
            for j in T.Parallel(width):
                z = T.cast(Z[row, head, j], T.float32)
                Y[row, head, j] = x[j] * T.rsqrt(total[0] / width + eps) * Weight[j] / (1.0 + T.exp(-z))
    return main


@orin_jit
def qsa_prepare(M: int, capacity: int, heads: int = 24, kv_heads: int = 2,
                width: int = 256, rotary: int = 64, theta: float = 1e7,
                eps: float = 1e-6, is_neox_style: bool = False, staged: bool = False):
    """Normalize Q/K, text RoPE, append private KV and copy gate.

    QGate stores interleaved [head,query_or_gate,width] output rows. Position
    is the first input position, shared with all layers and advanced outside
    this kernel. Its entire [position,position+M) interval must fit capacity.
    Text positions are identical on all IMRoPE axes. Original HF Q/K weights
    use NeoX half-width pairs; a permuted GGUF fixture uses adjacent pairs.
    MRoPE frequency-axis interleaving does not select the Q/K pair layout.
    Norm weights are ordinary FP32 gamma. With staged=True only this chunk
    writes FP16 temporary rows; a separate INT8 store owns the persistent cache.
    """
    if any(type(v) is not int or v <= 0 for v in (M, capacity, heads, kv_heads, width, rotary)):
        raise ValueError('Invalid QSA preparation dimensions')
    if M > capacity or heads % kv_heads or rotary > width or rotary % 2 or width % 2:
        raise ValueError('Invalid QSA head/rotary geometry')
    if theta <= 1 or not 0 < eps < 1 or type(is_neox_style) is not bool:
        raise ValueError('Invalid QSA rotation/normalization')
    size = 1 << (width - 1).bit_length()
    cache_rows = M if staged else capacity
    @T.prim_func
    def main(QGate: T.Tensor((M, heads, 2, width), T.float16),
             Key: T.Tensor((M, kv_heads, width), T.float16),
             Value: T.Tensor((M, kv_heads, width), T.float16),
             QWeight: T.Tensor((width,), T.float32),
             KWeight: T.Tensor((width,), T.float32),
             Position: T.Tensor((1,), T.int32),
             Query: T.Tensor((M, heads, width), T.float16),
             KCache: T.Tensor((cache_rows, kv_heads, width), T.float16),
             VCache: T.Tensor((cache_rows, kv_heads, width), T.float16),
             Gate: T.Tensor((M, heads, width), T.float16)):
        with T.Kernel(M, heads + kv_heads, threads=128) as (row, head):
            x = T.alloc_fragment((size,), T.float32)
            square = T.alloc_fragment((size,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            normalized = T.alloc_shared((width,), T.float32)
            for j in T.Parallel(size):
                x[j] = 0.0
                if j < width:
                    if head < heads:
                        x[j] = T.cast(QGate[row, head, 0, j], T.float32)
                    else:
                        x[j] = T.cast(Key[row, head - heads, j], T.float32)
                square[j] = x[j] * x[j]
            T.reduce_sum(square, total, dim=0)
            for j in T.Parallel(width):
                if head < heads:
                    normalized[j] = x[j] * T.rsqrt(total[0] / width + eps) * QWeight[j]
                else:
                    normalized[j] = x[j] * T.rsqrt(total[0] / width + eps) * KWeight[j]
            for pair in T.Parallel(width // 2):
                first_index = T.if_then_else(is_neox_style and pair < rotary // 2, pair, 2 * pair)
                second_index = T.if_then_else(is_neox_style and pair < rotary // 2, pair + rotary // 2, 2 * pair + 1)
                angle = T.cast(Position[0] + row, T.float32) * T.pow(theta, -2.0 * pair / rotary)
                cosine = T.if_then_else(pair * 2 < rotary, T.cos(angle), 1.0)
                sine = T.if_then_else(pair * 2 < rotary, T.sin(angle), 0.0)
                first = normalized[first_index] * cosine - normalized[second_index] * sine
                second = normalized[first_index] * sine + normalized[second_index] * cosine
                if head < heads:
                    Query[row, head, first_index] = first
                    Query[row, head, second_index] = second
                    Gate[row, head, first_index] = QGate[row, head, 1, first_index]
                    Gate[row, head, second_index] = QGate[row, head, 1, second_index]
                elif Position[0] >= 0 and Position[0] + row < capacity:
                    KCache[T.if_then_else(staged,row,Position[0]+row), head - heads, first_index] = first
                    KCache[T.if_then_else(staged,row,Position[0]+row), head - heads, second_index] = second
                    VCache[T.if_then_else(staged,row,Position[0]+row), head - heads, first_index] = Value[row, head - heads, first_index]
                    VCache[T.if_then_else(staged,row,Position[0]+row), head - heads, second_index] = Value[row, head - heads, second_index]
    return main


@orin_jit
def qsa_short_attention(M: int, capacity: int, heads: int = 24, kv_heads: int = 2,
                        width: int = 256, budget: int = 2048):
    """Causal softmax GQA + sigmoid gate while all cells fit QSA's budget.

    This correctness-first SIMT implementation reads only causal live KV.
    No context truncation, sparse approximation or CPU attention computation.
    Private caches and caller-owned outputs have stable addresses for graphs.
    """
    if any(type(v) is not int or v <= 0 for v in (M, capacity, heads, kv_heads, width, budget)):
        raise ValueError('Invalid QSA attention dimensions')
    if capacity > budget or M > capacity or heads % kv_heads:
        raise ValueError('Dense QSA requires the full request to fit the selection budget')
    size = 1 << (width - 1).bit_length()
    @T.prim_func
    def main(Query: T.Tensor((M, heads, width), T.float16),
             KCache: T.Tensor((capacity, kv_heads, width), T.float16),
             VCache: T.Tensor((capacity, kv_heads, width), T.float16),
             Gate: T.Tensor((M, heads, width), T.float16),
             Position: T.Tensor((1,), T.int32),
             Output: T.Tensor((M, heads, width), T.float16)):
        with T.Kernel(M, heads, threads=128) as (row, head):
            query = T.alloc_fragment((size,), T.float32)
            product = T.alloc_fragment((size,), T.float32)
            result = T.alloc_fragment((size,), T.float32)
            score = T.alloc_fragment((1,), T.float32)
            maximum = T.alloc_local((1,), T.float32)
            denominator = T.alloc_local((1,), T.float32)
            previous = T.alloc_local((1,), T.float32)
            probability = T.alloc_local((1,), T.float32)
            maximum[0] = -3.402823466e38
            denominator[0] = 0.0
            T.clear(result)
            for j in T.Parallel(size):
                query[j] = T.if_then_else(j < width, T.cast(Query[row, head, j], T.float32), 0.0)
            for cell in T.serial(T.min(capacity, T.max(0, Position[0] + row + 1))):
                for j in T.Parallel(size):
                    product[j] = T.if_then_else(j < width,
                        query[j] * T.cast(KCache[cell, head // (heads // kv_heads), j], T.float32), 0.0)
                T.reduce_sum(product, score, dim=0)
                score[0] *= width ** -.5
                previous[0] = T.exp(maximum[0] - T.max(maximum[0], score[0]))
                maximum[0] = T.max(maximum[0], score[0])
                probability[0] = T.exp(score[0] - maximum[0])
                denominator[0] = denominator[0] * previous[0] + probability[0]
                for j in T.Parallel(width):
                    result[j] = result[j] * previous[0] + probability[0] * T.cast(VCache[cell, head // (heads // kv_heads), j], T.float32)
            for j in T.Parallel(width):
                Output[row, head, j] = T.if_then_else(denominator[0] > 0,
                    result[j] / denominator[0] / (1.0 + T.exp(-T.cast(Gate[row, head, j], T.float32))), 0.0)
    return main
