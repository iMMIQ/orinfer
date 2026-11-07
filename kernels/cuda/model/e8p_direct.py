"""Dense E8P-Q4 schedule with exact A8 planes and direct INT4 MMA operands.

Compressed K-major weights remain packed throughout the pipeline. Both E8P
stages accumulate over the full K before integer reconstruction and rescaling.
All rows share this schedule; tile choices belong to offline tuning.
"""

import tilelang.language as T
from kernels.cuda.model.e8p_core import SOURCE, DENSE_SOURCE, mma_pair, low_plane, high_plane
from kernels.cuda.model.integer_core import rescale


def direct_tensor_core(m, N, K, bm, bn, *, stages, packed_k, coefficients, split_k):
    coarse, residual = coefficients
    threads = bn * split_k
    multiply = mma_pair(bm)

    @T.macro
    def project(A, W, Book, S, Unused, AS, Out, by, bx):
        T.import_source(SOURCE + DENSE_SOURCE)
        tid = T.get_thread_binding()
        lane = tid % 32
        warp = (tid // 32) % (bn // 32)
        kwarp = tid // bn
        packed = T.alloc_shared((stages, split_k, bn, 16), T.uint32)
        partial = T.alloc_shared((split_k, bm, bn), T.int32)
        book = T.alloc_shared((256,), T.uint32)
        lo0 = T.alloc_local((bm,), T.int32)
        hi0 = T.alloc_local((bm,), T.int32)
        lo1 = T.alloc_local((bm,), T.int32)
        hi1 = T.alloc_local((bm,), T.int32)
        v = T.alloc_local((4,), T.uint32)
        a = T.alloc_local((bm // 2,), T.uint32)
        sums = T.alloc_local((bm // 8,), T.int32)
        bits = T.alloc_local((2,), T.uint32)
        for i in T.Parallel(256):
            book[i] = T.call_pure_extern("uint32", "direct_nibble_basis", Book[i, 0])
        for i in T.unroll(bm):
            lo0[i] = 0
            hi0[i] = 0
            lo1[i] = 0
            hi1[i] = 0
        for stripe in T.unroll(bm // 8):
            sums[stripe] = 0
        for initial in T.unroll(min(stages - 1, K // (128 * split_k))):
            for half in T.unroll(4):
                T.ptx_cp_async(
                    T.access_ptr(
                        packed[
                            initial,
                            (half * threads + tid) // (4 * bn),
                            ((half * threads + tid) // 4) % bn,
                            ((half * threads + tid) % 4) * 4,
                        ],
                        "w",
                        4,
                    ),
                    T.access_ptr(
                        W[
                            (
                                (initial * split_k + (half * threads + tid) // (4 * bn)) * 128
                                + ((half * threads + tid) % 4) * 32
                            )
                            // packed_k,
                            T.min(bx * bn + ((half * threads + tid) // 4) % bn, N - 1),
                            (((half * threads + tid) % 4) * 4) % (packed_k // 8),
                        ],
                        "r",
                        4,
                    ),
                    4,
                    predicate=bx * bn + ((half * threads + tid) // 4) % bn < N,
                )
            T.ptx_commit_group()
        T.sync_threads()
        for step in T.serial(K // (128 * split_k)):
            slot = step % stages
            if step + stages - 2 < K // (128 * split_k):
                T.ptx_wait_group(stages - 2)
            else:
                T.ptx_wait_group(0)
            T.sync_threads()
            if step + stages - 1 < K // (128 * split_k):
                nxt = (step + stages - 1) % stages
                for half in T.unroll(4):
                    T.ptx_cp_async(
                        T.access_ptr(
                            packed[
                                nxt,
                                (half * threads + tid) // (4 * bn),
                                ((half * threads + tid) // 4) % bn,
                                ((half * threads + tid) % 4) * 4,
                            ],
                            "w",
                            4,
                        ),
                        T.access_ptr(
                            W[
                                (
                                    (
                                        (step + stages - 1) * split_k
                                        + (half * threads + tid) // (4 * bn)
                                    )
                                    * 128
                                    + ((half * threads + tid) % 4) * 32
                                )
                                // packed_k,
                                T.min(bx * bn + ((half * threads + tid) // 4) % bn, N - 1),
                                (((half * threads + tid) % 4) * 4) % (packed_k // 8),
                            ],
                            "r",
                            4,
                        ),
                        4,
                        predicate=bx * bn + ((half * threads + tid) // 4) % bn < N,
                    )
                T.ptx_commit_group()
            for sub in T.unroll(2):
                for stripe in T.unroll(bm // 8):
                    row = by * bm + stripe * 8 + lane // 4
                    for part in T.unroll(2):
                        bits[0] = 0
                        bits[1] = 0
                        if row < m:
                            bits[0] = T.ldg32(
                                A[
                                    row,
                                    ((step * split_k + kwarp) * 16 + sub * 8 + lane % 4 + part * 4)
                                    * 8,
                                ]
                            )
                            bits[1] = T.ldg32(
                                A[
                                    row,
                                    ((step * split_k + kwarp) * 16 + sub * 8 + lane % 4 + part * 4)
                                    * 8
                                    + 4,
                                ]
                            )
                        a[stripe * 2 + part] = low_plane(bits[0], bits[1])
                        a[bm // 4 + stripe * 2 + part] = high_plane(bits[0], bits[1])
                        sums[stripe] += T.call_pure_extern("int32", "nib_sum", bits[0], bits[1])
                for nt in T.unroll(2):
                    for level in T.unroll(2):
                        for rr in T.unroll(4):
                            n = warp * 32 + nt * 16 + lane // 4 + (rr % 2) * 8
                            col = lane % 4 + (rr // 2) * 4
                            code = (
                                packed[slot, kwarp, n, sub * 8 + col] >> (level * 16)
                            ) & T.uint32(65535)
                            v[rr] = T.call_pure_extern("uint32", "nib_fast", code, book[code >> 8])
                        for stripe in T.unroll(bm // 8):
                            if level == 0:
                                multiply(lo0, hi0, v, a, nt, stripe)
                            else:
                                multiply(lo1, hi1, v, a, nt, stripe)
            T.sync_threads()
        for stripe in T.unroll(bm // 8):
            sums[stripe] = T.call_extern("int32", "direct_reduce4", sums[stripe])
        for nt in T.unroll(2):
            for stripe in T.unroll(bm // 8):
                for r in T.unroll(2):
                    col = bx * bn + warp * 32 + nt * 16 + lane // 4 + r * 8
                    for t in T.unroll(2):
                        row = stripe * 8 + (lane % 4) * 2 + t
                        total = T.call_extern(
                            "int32", "direct_row_sum", sums[stripe], (lane % 4) * 2 + t
                        )
                        j = nt * (bm // 2) + stripe * 4 + r * 2 + t
                        value = (
                            2 * coarse * (lo0[j] + 16 * hi0[j])
                            + 2 * residual * (lo1[j] + 16 * hi1[j])
                            + (coarse + residual) * total
                        )
                        partial[kwarp, row, col - bx * bn] = value
        T.sync_threads()
        if kwarp == 0:
            for nt in T.unroll(2):
                for stripe in T.unroll(bm // 8):
                    for r in T.unroll(2):
                        col = bx * bn + warp * 32 + nt * 16 + lane // 4 + r * 8
                        for t in T.unroll(2):
                            row = stripe * 8 + (lane % 4) * 2 + t
                            combined = T.alloc_local((1,), T.int32)
                            combined[0] = 0
                            for ks in T.unroll(split_k):
                                combined[0] += partial[ks, row, col - bx * bn]
                            if by * bm + row < m and col < N:
                                Out[by * bm + row, col] = rescale(
                                    combined[0], S[col], AS[by * bm + row]
                                )

    return project
