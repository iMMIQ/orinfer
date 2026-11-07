"""Causal GQA rows with shared INT8 KV tiles and bounded split workspace.

A row is a query token, independent of request/batch/MTP meaning. The caller
binds one owner's KV and causal positions. Query heads sharing KV are packed
into the MMA row axis. Split count follows the live grid geometry; its target
and tile sizes are selected offline, never from a context-length enumeration.
Short causal packets can instead assign one CTA tile per query, preserving
the single-query split and reduction contract for every owner and width.
"""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.shared.rows import row_count, explicit_rows, explicit_scalar
from kernels.cuda.model.kv_int8 import HALF2_SOURCE, int8_kv_tile


SCALAR_SOURCE = r"""
__device__ __forceinline__ int attn_split_count(int rows,int groups,int bm,int target,int maximum,int instances) {
    int tiles=((rows*groups+bm-1)/bm)*instances;
    return min(maximum,(target+tiles-1)/tiles);
}
__device__ __forceinline__ int attn_max_position(const int* positions, int rows, int qb, int bm, int groups) {
    __shared__ int warp_maximum[32];
    int value=-1;
    for(int i=threadIdx.x; i<bm; i+=blockDim.x) {
        int token=(qb*bm+i)/groups;
        if(token<rows) value=max(value,positions[token]);
    }
    for(int delta=16;delta>0;delta>>=1) value=max(value,__shfl_xor_sync(0xffffffff,value,delta));
    if((threadIdx.x&31)==0) warp_maximum[threadIdx.x>>5]=value;
    __syncthreads();
    value=(threadIdx.x<blockDim.x/32)?warp_maximum[threadIdx.x]:-1;
    if(threadIdx.x<32) {
        for(int delta=16;delta>0;delta>>=1) value=max(value,__shfl_xor_sync(0xffffffff,value,delta));
        if(threadIdx.x==0) warp_maximum[0]=value;
    }
    __syncthreads();
    return warp_maximum[0];
}

"""


def geometry(
    rows, block_m, target_tiles, max_splits, groups, instances=1, independent_queries=False
):
    if independent_queries:
        return rows, min(max_splits, target_tiles)
    tiles = T.ceildiv(rows * groups, block_m)
    splits = T.min(max_splits, T.ceildiv(target_tiles, tiles * instances))
    return tiles, splits


def gate_result(value, raw):
    return T.cast(T.cast(value, T.float16), T.float32) * T.cast(
        T.cast(1 / (1 + T.exp(-T.cast(raw, T.float32))), T.float16), T.float32
    )


@orin_jit
def causal_attention_int8(
    capacity,
    context,
    block_m=32,
    block_n=64,
    target_tiles=16,
    max_splits=8,
    query_heads=24,
    kv_heads=4,
    dimension=256,
    head_group=None,
    threads=128,
    stages=0,
    parallel_instances=False,
    decoded_kv=False,
    gated=True,
    independent_queries=False,
):
    groups = query_heads // kv_heads if head_group is None else head_group
    if (
        min(capacity, context, target_tiles) <= 0
        or query_heads % kv_heads
        or type(decoded_kv) is not bool
        or (decoded_kv and parallel_instances)
        or dimension % 64
        or block_m not in (16, 32, 64, 128)
        or block_n not in (32, 64)
        or threads not in (128, 256, 512)
        or stages not in (0, 1)
        or max_splits not in (1, 2, 4, 8, 16)
        or groups <= 0
        or (query_heads // kv_heads) % groups
        or type(independent_queries) is not bool
        or (independent_queries and groups > block_m)
        or (independent_queries and capacity > 8)
    ):
        raise ValueError("Invalid dynamic GQA geometry")
    m = row_count(capacity, True)
    instances = T.dynamic("independent_instances") if parallel_instances else 1
    work_heads = query_heads // groups
    heads_per_kv = query_heads // kv_heads // groups
    tiles, splits = geometry(
        m, block_m, target_tiles, max_splits, groups, instances, independent_queries
    )
    # splits > 1 implies tiles < target_tiles, and tiles*splits < 2*target_tiles.
    workspace_tiles = 2 * target_tiles
    workspace_heads = m * work_heads if independent_queries else work_heads
    load_kv = int8_kv_tile(block_n, dimension, threads)
    storage_width = kv_heads * dimension // (1 if decoded_kv else 4)
    storage_dtype = T.float16 if decoded_kv else T.uint32

    @T.prim_func
    def main(
        Q: T.Tensor((m, query_heads, dimension), T.float16),
        K: T.Tensor((1, context, storage_width), storage_dtype),
        V: T.Tensor((1, context, storage_width), storage_dtype),
        KS: T.Tensor((1, context, kv_heads * dimension // 64), T.float16),
        VS: T.Tensor((1, context, kv_heads * dimension // 64), T.float16),
        Gate: T.Tensor((m, query_heads, dimension), T.float16),
        Positions: T.Tensor((m,), T.int32),
        Lengths: T.Tensor((1,), T.int32),
        Y: T.Tensor((m, query_heads, dimension), T.float16),
        Partial: T.Tensor((workspace_heads, workspace_tiles, block_m, dimension), T.float32),
        Maximum: T.Tensor((workspace_heads, workspace_tiles, block_m), T.float32),
        Denominator: T.Tensor((workspace_heads, workspace_tiles, block_m), T.float32),
    ):
        with T.Kernel(tiles, work_heads, splits, threads=threads) as (qb, kh, split):
            T.import_source(HALF2_SOURCE + SCALAR_SOURCE)
            split_count = T.alloc_local((1,), T.int32)
            if independent_queries:
                split_count[0] = min(max_splits, target_tiles)
            else:
                split_count[0] = T.call_pure_extern(
                    "int32",
                    "attn_split_count",
                    m,
                    groups,
                    block_m,
                    target_tiles,
                    max_splits,
                    instances,
                )
            q = T.alloc_shared((block_m, dimension), T.float16)
            k = T.alloc_shared((block_n, dimension), T.float16)
            v = T.alloc_shared((block_n, dimension), T.float16)
            p = T.alloc_shared((block_m, block_n), T.float16)
            if stages and not decoded_kv:
                packed_k = T.alloc_shared((1, block_n, dimension // 4), T.uint32)
                packed_v = T.alloc_shared((1, block_n, dimension // 4), T.uint32)
                scale_k = T.alloc_shared((1, block_n, dimension // 64), T.float16)
                scale_v = T.alloc_shared((1, block_n, dimension // 64), T.float16)
            score = T.alloc_fragment((block_m, block_n), T.float32)
            out = T.alloc_fragment((block_m, dimension), T.float32)
            maximum = T.alloc_fragment((block_m,), T.float32)
            previous = T.alloc_fragment((block_m,), T.float32)
            denom = T.alloc_fragment((block_m,), T.float32)
            rowsum = T.alloc_fragment((block_m,), T.float32)
            correction = T.alloc_fragment((block_m,), T.float32)
            pos = T.alloc_fragment((block_m,), T.int32)
            maxpos = (
                T.alloc_local((1,), T.int32) if block_m == 128 else T.alloc_fragment((1,), T.int32)
            )
            T.clear(out)
            T.clear(denom)
            T.fill(maximum, -1e30)
            for i in T.Parallel(block_m):
                token = qb if independent_queries else (qb * block_m + i) // groups
                live = token < m and (i < groups if independent_queries else True)
                pos[i] = T.if_then_else(live, Positions[token], -1)
            if independent_queries:
                maxpos[0] = Positions[qb]
            elif block_m == 128:
                # Keep the CTA-wide position reduction independent of the MMA
                # fragment's row layout for larger query tiles.
                maxpos[0] = T.call_extern(
                    "int32",
                    "attn_max_position",
                    T.access_ptr(Positions[0], "r", m),
                    m,
                    qb,
                    block_m,
                    groups,
                )
            else:
                T.reduce_max(pos, maxpos, dim=0)
            valid = T.max(0, T.min(T.min(Lengths[0], context), maxpos[0] + 1))
            width = T.alloc_local((1,), T.int32)
            width[0] = T.truncdiv(valid + split_count[0] - 1, split_count[0])
            start = split * width[0]
            end = T.min(start + width[0], valid)
            if decoded_kv:
                live_k = T.decl_buffer((1, end, storage_width), storage_dtype, data=K.data)
                live_v = T.decl_buffer((1, end, storage_width), storage_dtype, data=V.data)
            elif stages:
                # Restrict the copy's source view to committed KV. Capacity may
                # reserve uncommitted virtual pages beyond the live sequence.
                live_k = T.decl_buffer((1, end, kv_heads * dimension // 4), T.uint32, data=K.data)
                live_v = T.decl_buffer((1, end, kv_heads * dimension // 4), T.uint32, data=V.data)
                live_ks = T.decl_buffer(
                    (1, end, kv_heads * dimension // 64), T.float16, data=KS.data
                )
                live_vs = T.decl_buffer(
                    (1, end, kv_heads * dimension // 64), T.float16, data=VS.data
                )
            for i, d in T.Parallel(block_m, dimension):
                token = qb if independent_queries else (qb * block_m + i) // groups
                q[i, d] = 0.0
                if token < m and (i < groups if independent_queries else True):
                    head = kh * groups + (i if independent_queries else (qb * block_m + i) % groups)
                    q[i, d] = Q[token, head, d]
            for tile in T.Pipelined(T.ceildiv(T.max(0, end - start), block_n), num_stages=stages):
                base = start + tile * block_n
                if decoded_kv:
                    T.copy(live_k[0, base, (kh // heads_per_kv) * dimension], k)
                    T.copy(live_v[0, base, (kh // heads_per_kv) * dimension], v)
                elif stages:
                    # Prefetch compressed operands, rather than double-buffering
                    # the larger decoded FP16 tiles. T.copy guards the physical
                    # source boundary; load_kv also zeroes the live KV tail.
                    T.copy(live_k[0, base, (kh // heads_per_kv) * (dimension // 4)], packed_k)
                    T.copy(live_v[0, base, (kh // heads_per_kv) * (dimension // 4)], packed_v)
                    T.copy(live_ks[0, base, (kh // heads_per_kv) * (dimension // 64)], scale_k)
                    T.copy(live_vs[0, base, (kh // heads_per_kv) * (dimension // 64)], scale_v)
                    load_kv(packed_k, packed_v, scale_k, scale_v, k, v, 0, 0, 0, end - base)
                else:
                    load_kv(K, V, KS, VS, k, v, 0, kh // heads_per_kv, base, end)
                T.sync_threads()
                T.clear(score)
                # Keep more query rows in each warp. QK and PV use the same
                # row distribution, reducing cross-warp softmax reductions.
                T.gemm(q, k, score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i in T.Parallel(block_m):
                    previous[i] = maximum[i]
                for i, j in T.Parallel(block_m, block_n):
                    score[i, j] = T.if_then_else(
                        base + j < end and base + j <= pos[i],
                        score[i, j] * (dimension**-0.5),
                        -1e30,
                    )
                T.reduce_max(score, maximum, dim=1, clear=False)
                for i, j in T.Parallel(block_m, block_n):
                    score[i, j] = T.if_then_else(
                        base + j < end and base + j <= pos[i],
                        T.exp(score[i, j] - maximum[i]),
                        0,
                    )
                    p[i, j] = score[i, j]
                T.reduce_sum(score, rowsum, dim=1)
                for i in T.Parallel(block_m):
                    correction[i] = T.exp(previous[i] - maximum[i])
                    denom[i] = denom[i] * correction[i] + rowsum[i]
                for i, d in T.Parallel(block_m, dimension):
                    out[i, d] *= correction[i]
                T.gemm(p, v, out, policy=T.GemmWarpPolicy.FullRow)
            if split_count[0] > 1:
                storage_head = qb * work_heads + kh if independent_queries else kh
                storage_tile = split if independent_queries else qb * split_count[0] + split
                for i, d in T.Parallel(block_m, dimension):
                    Partial[storage_head, storage_tile, i, d] = out[i, d]
                for i in T.Parallel(block_m):
                    Maximum[storage_head, storage_tile, i] = maximum[i]
                    Denominator[storage_head, storage_tile, i] = denom[i]
            else:
                for i, d in T.Parallel(block_m, dimension):
                    token = qb if independent_queries else (qb * block_m + i) // groups
                    head = kh * groups + (i if independent_queries else (qb * block_m + i) % groups)
                    if token < m and (i < groups if independent_queries else True):
                        value = T.if_then_else(denom[i] > 0, out[i, d] / T.max(denom[i], 1e-30), 0)
                        Y[token, head, d] = (
                            gate_result(value, Gate[token, head, d])
                            if gated
                            else T.cast(value, T.float16)
                        )

    main = explicit_rows(main)
    return explicit_scalar(main, instances) if parallel_instances else main


@orin_jit
def merge_attention_int8(
    capacity,
    block_m=32,
    target_tiles=16,
    max_splits=8,
    query_heads=24,
    kv_heads=4,
    dimension=256,
    head_group=None,
    parallel_instances=False,
    gated=True,
    independent_queries=False,
):
    m = row_count(capacity, True)
    instances = T.dynamic("independent_instances") if parallel_instances else 1
    groups = query_heads // kv_heads if head_group is None else head_group
    if (
        groups <= 0
        or (query_heads // kv_heads) % groups
        or max_splits not in (1, 2, 4, 8, 16)
        or type(independent_queries) is not bool
        or (independent_queries and groups > block_m)
        or (independent_queries and capacity > 8)
    ):
        raise ValueError("Invalid query-head group")
    reduction_levels = max_splits.bit_length() - 1
    work_heads = query_heads // groups
    _, splits = geometry(
        m, block_m, target_tiles, max_splits, groups, instances, independent_queries
    )
    workspace_tiles = 2 * target_tiles
    workspace_heads = m * work_heads if independent_queries else work_heads
    grid = (
        m * query_heads
        if parallel_instances or independent_queries
        else T.min(m * query_heads, 1 + (splits - 1) * target_tiles * block_m * query_heads)
    )

    @T.prim_func
    def main(
        Partial: T.Tensor((workspace_heads, workspace_tiles, block_m, dimension), T.float32),
        Maximum: T.Tensor((workspace_heads, workspace_tiles, block_m), T.float32),
        Denominator: T.Tensor((workspace_heads, workspace_tiles, block_m), T.float32),
        Gate: T.Tensor((m, query_heads, dimension), T.float16),
        Y: T.Tensor((m, query_heads, dimension), T.float16),
    ):
        with T.Kernel(grid, threads=128) as work:
            T.import_source(SCALAR_SOURCE)
            split_count = T.alloc_local((1,), T.int32)
            if independent_queries:
                split_count[0] = min(max_splits, target_tiles)
            else:
                split_count[0] = T.call_pure_extern(
                    "int32",
                    "attn_split_count",
                    m,
                    groups,
                    block_m,
                    target_tiles,
                    max_splits,
                    instances,
                )
            if split_count[0] > 1:
                token = work // query_heads
                head = work % query_heads
                kh = head // groups
                flat = token * groups + head % groups
                qb = token if independent_queries else flat // block_m
                qi = head % groups if independent_queries else flat % block_m
                storage_head = token * work_heads + kh if independent_queries else kh
                # Each thread owns output columns. Keep only a short split
                # vector live, rather than a split-by-head-dimension fragment.
                # Both sums use a balanced FP32 tree before FP16 gate rounding.
                maxima = T.alloc_local((max_splits,), T.float32)
                factors = T.alloc_local((max_splits,), T.float32)
                weights = T.alloc_local((max_splits,), T.float32)
                values = T.alloc_local((max_splits,), T.float32)
                maximum = T.alloc_local((1,), T.float32)
                maximum[0] = -1e30
                for s in T.unroll(max_splits):
                    maxima[s] = -1e30
                    storage_tile = s if independent_queries else qb * split_count[0] + s
                    if s < split_count[0] and Denominator[storage_head, storage_tile, qi] > 0:
                        maxima[s] = Maximum[storage_head, storage_tile, qi]
                    maximum[0] = T.max(maximum[0], maxima[s])
                for s in T.unroll(max_splits):
                    factors[s] = T.exp(maxima[s] - maximum[0])
                    weights[s] = 0.0
                    if s < split_count[0]:
                        storage_tile = s if independent_queries else qb * split_count[0] + s
                        weights[s] = factors[s] * Denominator[storage_head, storage_tile, qi]
                for level in T.unroll(reduction_levels):
                    for s in T.unroll(max_splits >> (level + 1)):
                        weights[s] = weights[2 * s] + weights[2 * s + 1]
                for column in T.unroll((dimension + 127) // 128):
                    d = column * 128 + T.get_thread_binding()
                    if d < dimension:
                        for s in T.unroll(max_splits):
                            values[s] = 0.0
                            if s < split_count[0]:
                                storage_tile = s if independent_queries else qb * split_count[0] + s
                                values[s] = factors[s] * Partial[storage_head, storage_tile, qi, d]
                        for level in T.unroll(reduction_levels):
                            for s in T.unroll(max_splits >> (level + 1)):
                                values[s] = values[2 * s] + values[2 * s + 1]
                        value = T.if_then_else(
                            weights[0] > 0, values[0] / T.max(weights[0], 1e-30), 0
                        )
                        Y[token, head, d] = (
                            gate_result(value, Gate[token, head, d])
                            if gated
                            else T.cast(value, T.float16)
                        )
            else:
                # Keep a real buffer argument when an unsplit specialization
                # folds the merge away; NVRTC's adapter rejects zero arguments.
                if work == 0 and T.get_thread_binding() == 0:
                    Denominator[0, 0, 0] = 0.0

    main = explicit_rows(main)
    return explicit_scalar(main, instances) if parallel_instances else main
