"""Exact partitioned QSA selection on SM87.

Each CTA retains its local top-512 using the original nonnegative FP32 bits.
A bounded union contains every global winner, including lower-ID ties. The
merge preserves greater-then-equal ascending IDs and the incomplete tail.
"""

import tilelang.language as T
from kernels.cuda.model.qsa import geometry
from kernels.shared.rows import row_count
from orinfer_build.compiler.cuda.common import orin_jit


@orin_jit
def select(
    m: int, capacity: int, dynamic_rows=False, tile: int = 512, threads: int = 128, parts: int = 2
):
    blocks = geometry(m, capacity)
    if tile not in (256, 512, 1024) or threads not in (128, 256) or parts not in (2, 4, 8):
        raise ValueError("Invalid QSA selection geometry")
    m = row_count(m, dynamic_rows)

    def histogram(length):

        @T.macro
        def accumulate(Values, Hist, Control, count, digit):
            for i in T.Parallel(length):
                bucket = T.alloc_var(T.int32, 256)
                if i < count:
                    key = Values[i]
                    if digit == 0 or key >> (32 - digit * 8) == Control[1] >> (32 - digit * 8):
                        bucket = (key >> (24 - digit * 8)) & 255
                peers = T.call_pure_extern(
                    "uint32", "__match_any_sync", T.uint32(4294967295), bucket
                )
                if (
                    bucket < 256
                    and T.get_thread_binding() % 32
                    == T.call_pure_extern("int32", "__ffs", peers) - 1
                ):
                    T.atomic_add(Hist[bucket], T.call_pure_extern("int32", "__popc", peers))

        return accumulate

    tile_histogram = histogram(tile)

    @T.macro
    def choose(Hist, Control, digit):
        cumulative = T.alloc_fragment((256,), T.int32)
        T.sync_threads()
        for i in T.Parallel(256):
            cumulative[i] = Hist[255 - i]
        T.cumsum(cumulative, dim=0)
        for i in T.Parallel(256):
            before = cumulative[i] - Hist[255 - i]
            if before < Control[2] and cumulative[i] >= Control[2]:
                Control[1] = Control[1] | ((255 - i) << (24 - digit * 8))
                Control[6] = Control[2] - before
        T.sync_threads()
        if T.get_thread_binding() == 0:
            Control[2] = Control[6]
        T.sync_threads()

    @T.prim_func
    def main(
        Scores: T.Tensor((blocks, m), T.float32),
        Position: T.Tensor((1,), T.int32),
        Selected: T.Tensor((m, parts * 1024), T.int32),
    ):
        with T.Kernel(m, parts, threads=threads) as (row, part):
            hist = T.alloc_shared((256,), T.int32)
            control = T.alloc_shared((7,), T.int32)
            equal_ids = T.alloc_shared((512,), T.int32)
            values = T.alloc_fragment((tile,), T.int32)
            flags = T.alloc_fragment((tile,), T.int32)
            ranks = T.alloc_fragment((tile,), T.int32)
            equals = T.alloc_fragment((tile,), T.int32)
            equal_ranks = T.alloc_fragment((tile,), T.int32)
            causal_full = (Position[0] + row + 1) // 4
            start = part * T.ceildiv(causal_full, parts)
            full = T.max(0, T.min(T.ceildiv(causal_full, parts), causal_full - start))
            tiles = T.ceildiv(full, tile)
            T.clear(control)
            if T.get_thread_binding() == 0:
                control[3] = -1
            T.sync_threads()
            if full > 0:
                if T.get_thread_binding() == 0:
                    control[2] = T.min(512, full)
                T.sync_threads()
                for digit in T.unroll(4):
                    T.clear(hist)
                    T.sync_threads()
                    for block in T.serial(tiles):
                        for i in T.Parallel(tile):
                            values[i] = 0
                            if block * tile + i < full:
                                values[i] = T.reinterpret(
                                    T.int32, Scores[start + block * tile + i, row]
                                )
                        tile_histogram(
                            values, hist, control, T.min(tile, full - block * tile), digit
                        )
                    choose(hist, control, digit)
                if T.get_thread_binding() == 0:
                    control[3] = control[1]
                T.sync_threads()
            for i in T.Parallel(1024):
                Selected[row, part * 1024 + i] = -1
            for block in T.serial(tiles):
                for i in T.Parallel(tile):
                    flags[i] = 0
                    equals[i] = 0
                    if block * tile + i < full:
                        key = T.reinterpret(T.int32, Scores[start + block * tile + i, row])
                        values[i] = key
                        flags[i] = T.cast(key > control[3], T.int32)
                        equals[i] = T.cast(key == control[3], T.int32)
                    ranks[i] = flags[i]
                    equal_ranks[i] = equals[i]
                T.cumsum(ranks, dim=0)
                T.cumsum(equal_ranks, dim=0)
                for i in T.Parallel(tile):
                    if flags[i] != 0:
                        Selected[row, part * 1024 + control[4] + ranks[i] - 1] = values[i]
                        Selected[row, part * 1024 + 512 + control[4] + ranks[i] - 1] = (
                            start + block * tile + i
                        )
                    if equals[i] != 0 and control[5] + equal_ranks[i] <= 512:
                        equal_ids[control[5] + equal_ranks[i] - 1] = start + block * tile + i
                T.sync_threads()
                for i in T.Parallel(tile):
                    if i == tile - 1:
                        control[4] += ranks[i]
                        control[5] += equal_ranks[i]
                T.sync_threads()
            for i in T.Parallel(512):
                if i < T.min(full, 512) - control[4]:
                    Selected[row, part * 1024 + control[4] + i] = control[3]
                    Selected[row, part * 1024 + 512 + control[4] + i] = equal_ids[i]

    return main.with_attr("global_symbol", "qsa_select_partial")


@orin_jit
def merge(m: int, parts: int, dynamic_rows=False, threads: int = 128):
    """Merge bounded local winners, then sort IDs in the original emission order.

    No global score rescan or cluster synchronization is needed. Local top-512
    contains every possible global winner, including lower-ID score ties.
    """
    if parts not in (2, 4, 8) or threads not in (128, 256):
        raise ValueError("Invalid QSA candidate merge geometry")
    m = row_count(m, dynamic_rows)
    width = parts * 512

    @T.prim_func
    def main(
        Candidates: T.Tensor((m, parts * 1024), T.int32),
        Position: T.Tensor((1,), T.int32),
        Selected: T.Tensor((m, 2051), T.int32),
    ):
        with T.Kernel(m, threads=threads) as row:
            values = T.alloc_shared((width,), T.int32)
            keys = T.alloc_shared((width,), T.int32)
            hist = T.alloc_shared((256,), T.int32)
            control = T.alloc_shared((3,), T.int32)
            cumulative = T.alloc_fragment((256,), T.int32)
            next_keys = T.alloc_fragment((width,), T.int32)
            full = (Position[0] + row + 1) // 4
            tail = (Position[0] + row + 1) % 4
            for i in T.Parallel(width):
                values[i] = Candidates[row, i // 512 * 1024 + i % 512]
            T.clear(control)
            if T.get_thread_binding() == 0:
                control[1] = T.min(512, full)
            T.sync_threads()
            if full > 0:
                for digit in T.unroll(4):
                    T.clear(hist)
                    T.sync_threads()
                    for i in T.Parallel(width):
                        bucket = T.alloc_var(T.int32, 256)
                        key = values[i]
                        if key >= 0:
                            if digit == 0 or key >> 32 - digit * 8 == control[0] >> 32 - digit * 8:
                                bucket = key >> 24 - digit * 8 & 255
                        peers = T.call_pure_extern(
                            "uint32", "__match_any_sync", T.uint32(4294967295), bucket
                        )
                        if (
                            bucket < 256
                            and T.get_thread_binding() % 32
                            == T.call_pure_extern("int32", "__ffs", peers) - 1
                        ):
                            T.atomic_add(hist[bucket], T.call_pure_extern("int32", "__popc", peers))
                    T.sync_threads()
                    for i in T.Parallel(256):
                        cumulative[i] = hist[255 - i]
                    T.cumsum(cumulative, dim=0)
                    for i in T.Parallel(256):
                        before = cumulative[i] - hist[255 - i]
                        if before < control[1] and cumulative[i] >= control[1]:
                            control[0] = control[0] | 255 - i << 24 - digit * 8
                            control[2] = control[1] - before
                    T.sync_threads()
                    if T.get_thread_binding() == 0:
                        control[1] = control[2]
                    T.sync_threads()
            for i in T.Parallel(width):
                keys[i] = 2147483647
                if values[i] >= control[0] and values[i] >= 0:
                    identifier = Candidates[row, i // 512 * 1024 + 512 + i % 512]
                    keys[i] = identifier | T.if_then_else(values[i] == control[0], 1073741824, 0)
            T.sync_threads()
            for level in T.unroll(width.bit_length() - 1):
                for step in T.unroll(level + 1):
                    for i in T.Parallel(width):
                        other = i ^ 1 << level - step
                        smaller = (i & 1 << level + 1 == 0) == (i & 1 << level - step == 0)
                        next_keys[i] = T.if_then_else(
                            smaller, T.min(keys[i], keys[other]), T.max(keys[i], keys[other])
                        )
                    T.sync_threads()
                    T.copy(next_keys, keys)
                    T.sync_threads()
            for i in T.Parallel(2051):
                Selected[row, i] = -1
                if i < T.min(full, 512) * 4:
                    Selected[row, i] = (keys[i // 4] & 1073741823) * 4 + i % 4
                elif i < T.min(full, 512) * 4 + tail:
                    Selected[row, i] = full * 4 + i - T.min(full, 512) * 4

    return main.with_attr("global_symbol", "qsa_select_merge")
