"""Exact single-token BF16 projection with cached inputs and direct MMA operands.

Grouped columns remain in the original row-major weights; no read cache or
expanded representation is created. FP32 accumulates in the original K16 order.
"""

import tilelang.language as T
from orinfer_build.compiler.cuda.common import orin_jit
from kernels.cuda.model.bf16_mma import SOURCE
from kernels.cuda.model.floating_core import mma_loop
from kernels.shared.rows import row_count


@orin_jit(
    pass_configs={"tl.disable_thread_storage_sync": True, "tl.disable_shared_memory_reuse": True}
)
def bf16_single(
    N,
    K,
    groups=1,
    block_n=32,
    block_k=256,
    stages=3,
    *,
    rows=1,
    dynamic_rows=False,
    row_major_grid=False,
):
    if (
        any(type(x) is not int or x <= 0 for x in (N, K, groups, block_n, block_k, stages))
        or groups not in (1, 4)
        or block_n not in (16, 32)
        or block_k not in (128, 256)
        or stages not in (2, 3, 4)
        or N % block_n
        or K % block_k
    ):
        raise ValueError("Unsupported single-token BF16 projection geometry")
    bn, bk = block_n, block_k
    threads = bn * 2
    if type(rows) is not int or not 1 <= rows <= 1024:
        raise ValueError("Unsupported BF16 decode rows")
    if type(row_major_grid) is not bool:
        raise ValueError("Invalid request tile ordering")
    rows = row_count(rows, dynamic_rows)

    @T.prim_func
    def main(
        A: T.Tensor((rows, groups, K), T.float16),
        Weight: T.Tensor((N, groups * K // 2), T.uint32),
        Output: T.Tensor((groups, rows, N), T.float32),
    ):
        with T.Kernel(
            (N // bn) * rows if row_major_grid else N // bn,
            groups,
            1 if row_major_grid else rows,
            threads=threads,
        ) as (block, group, lane):
            bx = block // rows if row_major_grid else block
            row = block % rows if row_major_grid else lane
            T.import_source(SOURCE)
            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            live = T.alloc_shared((K,), T.bfloat16)
            w = T.alloc_shared((stages, bn, bk // 2), T.uint32)
            acc = T.alloc_local((4,), T.float32)
            v = T.alloc_local((4,), T.uint32)
            T.clear(acc)
            for initial in T.unroll(min(stages - 1, K // bk)):
                for it in T.unroll(bn * bk // (threads * 8)):
                    index = (tid + it * threads) * 4
                    n = index // (bk // 2)
                    col = index % (bk // 2)
                    T.ptx_cp_async(
                        T.access_ptr(w[initial, n, col ^ ((n % 8) * 4)], "w", 4),
                        T.access_ptr(
                            Weight[bx * bn + n, group * (K // 2) + initial * (bk // 2) + col],
                            "r",
                            4,
                        ),
                        4,
                    )
                T.ptx_commit_group()
            for j in T.Parallel(K):
                live[j] = A[row, group, j]
            for step in T.serial(K // bk):
                slot = step % stages
                if stages == 4:
                    if step + 2 < K // bk:
                        T.ptx_wait_group(2)
                    elif step + 1 < K // bk:
                        T.ptx_wait_group(1)
                    else:
                        T.ptx_wait_group(0)
                elif stages == 3:
                    if step + 1 < K // bk:
                        T.ptx_wait_group(1)
                    else:
                        T.ptx_wait_group(0)
                else:
                    T.ptx_wait_group(0)
                T.sync_threads()
                if step + stages - 1 < K // bk:
                    nxt = (step + stages - 1) % stages
                    for it in T.unroll(bn * bk // (threads * 8)):
                        index = (tid + it * threads) * 4
                        n = index // (bk // 2)
                        col = index % (bk // 2)
                        T.ptx_cp_async(
                            T.access_ptr(w[nxt, n, col ^ ((n % 8) * 4)], "w", 4),
                            T.access_ptr(
                                Weight[
                                    bx * bn + n,
                                    group * (K // 2) + (step + stages - 1) * (bk // 2) + col,
                                ],
                                "r",
                                4,
                            ),
                            4,
                        )
                    T.ptx_commit_group()
                for kk in T.unroll(bk // 16):
                    b0 = T.call_pure_extern(
                        "uint32",
                        "hc_raw_pair",
                        T.access_ptr(live[step * bk + kk * 16 + (lane % 4) * 2], "r", 2),
                    )
                    b1 = T.call_pure_extern(
                        "uint32",
                        "hc_raw_pair",
                        T.access_ptr(live[step * bk + kk * 16 + (lane % 4) * 2 + 8], "r", 2),
                    )
                    for rr in T.unroll(4):
                        n = warp * 16 + lane // 4 + (rr % 2) * 8
                        col = kk * 8 + lane % 4 + (rr // 2) * 4
                        v[rr] = w[slot, n, col ^ ((n % 8) * 4)]
                    T.evaluate(
                        T.call_extern(
                            "void",
                            "hc_mma_bf16",
                            T.access_ptr(acc[0], "rw", 4),
                            v[0],
                            v[1],
                            v[2],
                            v[3],
                            b0,
                            b1,
                        )
                    )
                T.sync_threads()
            if lane % 4 == 0:
                for rr in T.unroll(2):
                    Output[group, row, bx * bn + warp * 16 + lane // 4 + rr * 8] = acc[rr * 2]

    return main.with_attr("global_symbol", "bf16_single")


@orin_jit
def bf16_rows(
    N, K, groups=4, *, rows=1024, block_m=16, block_n=64, dynamic_rows=True, weight_left=True
):
    """Share weights across token rows, retaining each group's K16 FP32 sum.

    Operand orientation is selectable; neither group reduction nor output
    rounding moves across this primitive's boundary.
    """
    if groups not in (1, 4) or N % block_n or K % 128 or block_m not in (16, 32, 64):
        raise ValueError("Invalid grouped BF16 row tile")
    rows = row_count(rows, dynamic_rows)

    project = mma_loop(
        rows,
        K,
        block_m,
        block_n,
        128,
        weight_left=weight_left,
        grouped_input=True,
        weight_first=True,
    )

    @T.prim_func
    def bf16_rows_kernel(
        A: T.Tensor((rows, groups, K), T.float16),
        Weight: T.Tensor((N, groups * K), T.bfloat16),
        Output: T.Tensor((groups, rows, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(rows, block_m), N // block_n, groups, threads=128) as (
            by,
            bx,
            group,
        ):
            acc = T.alloc_fragment(
                (block_n, block_m) if weight_left else (block_m, block_n), T.float32
            )
            project(A, Weight, acc, by, bx, group)
            for i, j in T.Parallel(block_n, block_m):
                if by * block_m + j < rows:
                    Output[group, by * block_m + j, bx * block_n + i] = (
                        acc[i, j] if weight_left else acc[j, i]
                    )

    return bf16_rows_kernel
