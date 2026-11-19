"""Deterministic GPU top-k -> nucleus sampling with request-local inputs.

Scores and processors use FP64. Integer radix histograms preserve reduction
order across batch compositions. Fixed-point mass drops at most vocab / scale
unnormalized mass (the maximum-score token always has unit mass). No RNG state
or probability law is shared between requests.
"""

import tilelang.language as T
from kernels.shared.rows import explicit_scalar
from orinfer_build.compiler.cuda.common import orin_jit

TILE = 1024
SHORTLIST = 256
SHORTLIST_TOP_K = 64
NEEDS_RADIX = 2


def _fixed_mass(score, maximum, temperature, scale):
    return T.cast(
        T.call_pure_extern("float64", "exp", (score - maximum) / temperature) * T.float64(scale),
        T.uint64,
    )


@orin_jit
def initialize(vocab: int):
    @T.prim_func
    def main(Counts: T.Tensor((vocab,), T.int32), Bias: T.Tensor((vocab,), T.float64)):
        with T.Kernel(T.ceildiv(vocab, 256), threads=256) as b:
            for i in T.Parallel(256):
                j = b * 256 + i
                if j < vocab:
                    Counts[j] = 0
                    Bias[j] = 0

    return main


@orin_jit
def history(vocab: int, context: int):
    @T.prim_func
    def main(
        History: T.Tensor((context,), T.int32),
        BiasIds: T.Tensor((vocab,), T.int32),
        BiasValues: T.Tensor((vocab,), T.float64),
        Control: T.Tensor((4,), T.int32),
        Counts: T.Tensor((vocab,), T.int32),
        Bias: T.Tensor((vocab,), T.float64),
    ):
        with T.Kernel(T.ceildiv(max(context, vocab), 256), threads=256) as b:
            for i in T.Parallel(256):
                j = b * 256 + i
                if j < Control[0]:
                    token = History[j]
                    if token >= 0 and token < vocab:
                        T.atomic_add(Counts[token], 1)
                if j < Control[1]:
                    token = BiasIds[j]
                    if token >= 0 and token < vocab:
                        Bias[token] = BiasValues[j]

    return main


@orin_jit
def scores(vocab: int):
    blocks = (vocab + TILE - 1) // TILE

    @T.prim_func
    def main(
        Logits: T.Tensor((vocab,), T.float32),
        Counts: T.Tensor((vocab,), T.int32),
        Bias: T.Tensor((vocab,), T.float64),
        Mask: T.Tensor(((vocab + 31) // 32,), T.uint32),
        Control: T.Tensor((4,), T.int32),
        Parameters: T.Tensor((6,), T.float64),
        Scores: T.Tensor((vocab,), T.float64),
        Keys: T.Tensor((vocab,), T.uint64),
        Maximum: T.Tensor((blocks,), T.float64),
        Invalid: T.Tensor((blocks,), T.int32),
    ):
        with T.Kernel(blocks, threads=256) as b:
            values = T.alloc_fragment((TILE,), T.float64)
            bad = T.alloc_fragment((TILE,), T.int32)
            maximum = T.alloc_fragment((1,), T.float64)
            invalid = T.alloc_fragment((1,), T.int32)
            for i in T.Parallel(TILE):
                j = b * TILE + i
                values[i] = -T.infinity(T.float64)
                bad[i] = 0
                if j < vocab:
                    raw = T.cast(Logits[j], T.float64)
                    value = T.alloc_var(T.float64, raw)
                    count = Counts[j]
                    if count > 0:
                        value = T.if_then_else(
                            value < 0, value * Parameters[4], value / Parameters[4]
                        )
                        value -= Parameters[2]
                    value -= Parameters[3] * T.cast(count, T.float64)
                    value += Bias[j]
                    bad[i] = T.cast(
                        not T.call_pure_extern("bool", "isfinite", raw)
                        or not T.call_pure_extern("bool", "isfinite", value),
                        T.int32,
                    )
                    allowed = Control[2] == 0 or ((Mask[j // 32] >> (j % 32)) & 1) != 0
                    Scores[j] = value
                    Keys[j] = T.uint64(0)
                    if allowed:
                        values[i] = value
                        bits = T.reinterpret(T.uint64, value)
                        Keys[j] = T.if_then_else((bits >> 63) != 0, ~bits, bits ^ T.uint64(1 << 63))
            T.reduce_max(values, maximum, dim=0)
            T.reduce_max(bad, invalid, dim=0)
            if T.get_thread_binding() == 0:
                Maximum[b] = maximum[0]
                Invalid[b] = invalid[0]

    return main


@orin_jit
def masses(vocab: int):
    blocks = (vocab + TILE - 1) // TILE
    reduction_width = 1 << (blocks - 1).bit_length()
    scale = 1 << min(52, 63 - (vocab - 1).bit_length())

    @T.prim_func
    def main(
        Scores: T.Tensor((vocab,), T.float64),
        Keys: T.Tensor((vocab,), T.uint64),
        Maximum: T.Tensor((blocks,), T.float64),
        Invalid: T.Tensor((blocks,), T.int32),
        Parameters: T.Tensor((6,), T.float64),
        Mass: T.Tensor((vocab,), T.uint64),
        Output: T.Tensor((2,), T.int32),
    ):
        with T.Kernel(blocks, threads=256) as b:
            maxima = T.alloc_fragment((reduction_width,), T.float64)
            invalids = T.alloc_fragment((reduction_width,), T.int32)
            maximum = T.alloc_fragment((1,), T.float64)
            invalid = T.alloc_fragment((1,), T.int32)
            for k in T.Parallel(reduction_width):
                maxima[k] = -T.infinity(T.float64)
                invalids[k] = 0
                if k < blocks:
                    maxima[k] = Maximum[k]
                    invalids[k] = Invalid[k]
            T.reduce_max(maxima, maximum, dim=0)
            T.reduce_max(invalids, invalid, dim=0)
            invalid[0] = T.max(
                invalid[0],
                T.cast(not T.call_pure_extern("bool", "isfinite", maximum[0]), T.int32),
            )
            for i in T.Parallel(TILE):
                j = b * TILE + i
                if j < vocab:
                    Mass[j] = 0
                    if Keys[j] != 0 and invalid[0] == 0:
                        Mass[j] = _fixed_mass(Scores[j], maximum[0], Parameters[0], scale)
            if b == 0 and T.get_thread_binding() == 0:
                Output[0] = 0
                Output[1] = invalid[0]

    return main


@orin_jit
def radix_histogram(vocab: int):
    blocks = (vocab + TILE - 1) // TILE
    digit, mode = T.dynamic("digit"), T.dynamic("mode")

    @T.prim_func
    def main(
        Keys: T.Tensor((vocab,), T.uint64),
        Mass: T.Tensor((vocab,), T.uint64),
        State: T.Tensor((7,), T.uint64),
        Control: T.Tensor((4,), T.int32),
        Parameters: T.Tensor((6,), T.float64),
        Invalid: T.Tensor((blocks,), T.int32),
        Hist: T.Tensor((blocks, 256), T.uint64),
    ):
        with T.Kernel(blocks, threads=256) as b:
            hist = T.alloc_shared((256,), T.uint64)
            if Invalid[0] == 0:
                T.clear(hist)
                if not ((mode == 0 and Control[3] == 0) or (mode == 1 and Parameters[1] == 1)):
                    for i in T.Parallel(TILE):
                        j = b * TILE + i
                        if j < vocab:
                            key = Keys[j]
                            index = T.cast(4294967295 - j, T.uint64)
                            valid = T.alloc_var(T.bool, key != 0)
                            if mode > 0:
                                valid = valid and (
                                    key > State[0] or (key == State[0] and index >= State[1])
                                )
                            if mode == 2:
                                valid = valid and (
                                    key > State[2] or (key == State[2] and index >= State[3])
                                )
                            shift = T.alloc_var(T.int32, T.max(0, (7 - digit) * 8))
                            byte = T.alloc_var(
                                T.int32, T.cast((key >> shift) & T.uint64(255), T.int32)
                            )
                            if digit < 8:
                                if digit > 0:
                                    valid = valid and ((key >> shift) >> 8) == (
                                        (State[5] >> shift) >> 8
                                    )
                            else:
                                shift = (11 - digit) * 8
                                byte = T.cast((index >> shift) & T.uint64(255), T.int32)
                                valid = valid and key == State[5]
                                if digit > 8:
                                    valid = valid and ((index >> shift) >> 8) == (
                                        (State[6] >> shift) >> 8
                                    )
                            if valid:
                                T.atomic_add(
                                    hist[byte], T.if_then_else(mode == 0, T.uint64(1), Mass[j])
                                )
                T.copy(hist, Hist[b, :])

    return explicit_scalar(explicit_scalar(main, digit), mode)


@orin_jit
def radix_choose(vocab: int):
    blocks = (vocab + TILE - 1) // TILE
    digit, mode = T.dynamic("digit"), T.dynamic("mode")

    @T.prim_func
    def main(
        Hist: T.Tensor((blocks, 256), T.uint64),
        State: T.Tensor((7,), T.uint64),
        Control: T.Tensor((4,), T.int32),
        Parameters: T.Tensor((6,), T.float64),
        Invalid: T.Tensor((blocks,), T.int32),
        Output: T.Tensor((2,), T.int32),
    ):
        with T.Kernel(1, threads=256):
            values = T.alloc_fragment((256,), T.uint64)
            hist = T.alloc_shared((256,), T.uint64)
            if Invalid[0] == 0:
                T.clear(values)
                for block in T.serial(blocks):
                    for i in T.Parallel(256):
                        values[i] += Hist[block, i]
                T.copy(values, hist)
                T.cumsum(hist, dim=0, reverse=True)
                if T.get_thread_binding() == 0:
                    skip = (mode == 0 and Control[3] == 0) or (mode == 1 and Parameters[1] == 1)
                    if digit == 0:
                        State[5] = 0
                        State[6] = 0
                        if mode == 0:
                            State[0] = 0
                            State[1] = 0
                        if mode == 1:
                            State[2] = 0
                            State[3] = 0
                        if not skip:
                            if hist[0] == 0:
                                Output[1] = 1
                                State[4] = 0
                            elif mode == 0:
                                State[4] = T.min(T.cast(Control[3], T.uint64), hist[0]) - 1
                            elif mode == 1:
                                State[4] = (
                                    T.cast(
                                        T.call_pure_extern(
                                            "float64",
                                            "ceil",
                                            Parameters[1] * T.cast(hist[0], T.float64),
                                        ),
                                        T.uint64,
                                    )
                                    - 1
                                )
                            else:
                                State[4] = T.min(
                                    T.cast(Parameters[5] * T.cast(hist[0], T.float64), T.uint64),
                                    hist[0] - 1,
                                )
                    if not skip and Output[1] == 0:
                        remaining = State[4]
                        if remaining >= hist[0]:
                            Output[1] = 1
                        else:
                            # The reverse cumulative histogram is nonincreasing.
                            # Find the last bucket covering the rank, skipping
                            # empty buckets without a serial scan of all 256.
                            found = T.alloc_var(T.int32, 0)
                            for bit in T.unroll(8):
                                next_bucket = found + (128 >> bit)
                                if remaining < hist[next_bucket]:
                                    found = next_bucket
                            preceding = T.if_then_else(
                                found == 255, T.uint64(0), hist[T.min(found + 1, 255)]
                            )
                            State[4] = remaining - preceding
                            if digit < 8:
                                State[5] = State[5] | (T.cast(found, T.uint64) << ((7 - digit) * 8))
                            else:
                                State[6] = State[6] | (
                                    T.cast(found, T.uint64) << ((11 - digit) * 8)
                                )
                            if digit == 11:
                                if mode == 0:
                                    State[0] = State[5]
                                    State[1] = State[6]
                                elif mode == 1:
                                    State[2] = State[5]
                                    State[3] = State[6]
                                else:
                                    Output[0] = T.cast(T.uint64(4294967295) - State[6], T.int32)

    return explicit_scalar(explicit_scalar(main, digit), mode)


@orin_jit
def shortlist_bound(vocab: int):
    # Reduce validity once. The radix threshold slot is free until fallback.
    blocks = (vocab + TILE - 1) // TILE
    extent = min(256, max(32, 1 << (blocks - 1).bit_length()))
    levels = (extent - 1).bit_length()
    reduction_width = 1 << (blocks - 1).bit_length()

    @T.prim_func
    def main(
        Maximum: T.Tensor((blocks,), T.float64),
        State: T.Tensor((7,), T.uint64),
        Control: T.Tensor((4,), T.int32),
        Invalid: T.Tensor((blocks,), T.int32),
        Output: T.Tensor((2,), T.int32),
    ):
        with T.Kernel(1, threads=extent):
            lane = T.get_thread_binding()
            cache = T.alloc_shared((extent,), T.uint64)
            maxima = T.alloc_fragment((reduction_width,), T.float64)
            invalids = T.alloc_fragment((reduction_width,), T.int32)
            maximum = T.alloc_fragment((1,), T.float64)
            invalid = T.alloc_fragment((1,), T.int32)
            for tile in T.Parallel(reduction_width):
                maxima[tile] = -T.infinity(T.float64)
                invalids[tile] = 0
                if tile < blocks:
                    maxima[tile] = Maximum[tile]
                    invalids[tile] = Invalid[tile]
            T.reduce_max(maxima, maximum, dim=0)
            T.reduce_max(invalids, invalid, dim=0)
            invalid[0] = T.max(
                invalid[0],
                T.cast(not T.call_pure_extern("bool", "isfinite", maximum[0]), T.int32),
            )
            T.sync_threads()
            if lane == 0:
                State[4] = T.reinterpret(T.uint64, maximum[0])
                State[6] = 0
                Invalid[0] = invalid[0]
                Output[0] = 0
                Output[1] = T.if_then_else(invalid[0] != 0, 1, NEEDS_RADIX)
            if Control[3] > 0 and Control[3] <= SHORTLIST_TOP_K and invalid[0] == 0:
                key = T.alloc_var(T.uint64, T.uint64(0))
                # Each sampled tile supplies one distinct witness. Its kth
                # maximum bounds the true kth key from below. Padded tiles
                # and tiles without allowed scores contribute a zero key.
                tile = lane if blocks <= extent else lane * blocks // extent
                if tile < blocks and Maximum[tile] != -T.infinity(T.float64):
                    bits = T.reinterpret(T.uint64, Maximum[tile])
                    key = T.if_then_else((bits >> 63) != 0, ~bits, bits ^ T.uint64(1 << 63))
                cache[lane] = key
                for group in T.unroll(1, levels + 1):
                    for step in T.unroll(group):
                        half = 1 << (group - step - 1)
                        T.sync_threads()
                        partner = cache[lane ^ half]
                        other = T.alloc_var(T.uint64, partner)
                        T.sync_threads()
                        if ((lane >> group) & 1) == ((lane >> (group - step - 1)) & 1):
                            cache[lane] = T.max(key, other)
                        else:
                            cache[lane] = T.min(key, other)
                        key = cache[lane]
                T.sync_threads()
                if lane == 0 and Control[3] <= extent:
                    State[6] = cache[Control[3] - 1]

    return main


@orin_jit
def shortlist_collect(vocab: int):
    @T.prim_func
    def main(
        Keys: T.Tensor((vocab,), T.uint64),
        State: T.Tensor((7,), T.uint64),
        Control: T.Tensor((4,), T.int32),
        Counts: T.Tensor((vocab,), T.int32),
        Output: T.Tensor((2,), T.int32),
    ):
        with T.Kernel(T.ceildiv(vocab, TILE), threads=256) as block:
            if Control[3] > 0 and Control[3] <= SHORTLIST_TOP_K and Output[1] == NEEDS_RADIX:
                for i in T.Parallel(TILE):
                    token = block * TILE + i
                    if token < vocab:
                        key = Keys[token]
                        if key != 0 and key >= State[6]:
                            slot = T.atomic_add(Output[0], 1, return_prev=True)
                            if slot < SHORTLIST:
                                Counts[slot] = token

    return main


@orin_jit
def shortlist_sample(vocab: int):
    blocks = (vocab + TILE - 1) // TILE
    rank_bits = (SHORTLIST - 1).bit_length()
    scale = 1 << min(52, 63 - (vocab - 1).bit_length())

    @T.prim_func
    def main(
        Keys: T.Tensor((vocab,), T.uint64),
        Scores: T.Tensor((vocab,), T.float64),
        State: T.Tensor((7,), T.uint64),
        Counts: T.Tensor((vocab,), T.int32),
        Control: T.Tensor((4,), T.int32),
        Parameters: T.Tensor((6,), T.float64),
        Invalid: T.Tensor((blocks,), T.int32),
        Output: T.Tensor((2,), T.int32),
    ):
        with T.Kernel(1, threads=SHORTLIST):
            key_order = T.alloc_shared((SHORTLIST,), T.uint64)
            id_order = T.alloc_shared((SHORTLIST,), T.int32)
            token_order = T.alloc_shared((SHORTLIST,), T.int32)
            prefix = T.alloc_shared((SHORTLIST,), T.uint64)
            if Control[3] > 0 and Control[3] <= SHORTLIST_TOP_K and Output[1] == NEEDS_RADIX:
                count = Output[0]
                if count > 0 and count <= SHORTLIST:
                    for i in T.Parallel(SHORTLIST):
                        prefix[i] = 0
                        key_order[i] = 0
                        id_order[i] = -1
                        if i < count:
                            id_order[i] = Counts[i]
                            key_order[i] = Keys[Counts[i]]
                    for i in T.Parallel(SHORTLIST):
                        if i < count:
                            rank = T.alloc_var(T.int32, 0)
                            key = key_order[i]
                            token = id_order[i]
                            for j in T.serial(count):
                                if key_order[j] > key or (
                                    key_order[j] == key and id_order[j] < token
                                ):
                                    rank += 1
                            if rank < Control[3]:
                                maximum = T.reinterpret(T.float64, State[4])
                                prefix[rank] = _fixed_mass(
                                    Scores[token], maximum, Parameters[0], scale
                                )
                                token_order[rank] = token
                    T.cumsum(prefix, dim=0)
                    if T.get_thread_binding() == 0:
                        Output[1] = 0
                        Invalid[0] = 1
                        active = T.min(count, Control[3])
                        total = prefix[active - 1]
                        Output[0] = 0
                        if total == 0:
                            Output[1] = 1
                        else:
                            cut = T.alloc_var(T.int32, active - 1)
                            if Parameters[1] < 1:
                                target = (
                                    T.cast(
                                        T.call_pure_extern(
                                            "float64",
                                            "ceil",
                                            Parameters[1] * T.cast(total, T.float64),
                                        ),
                                        T.uint64,
                                    )
                                    - 1
                                )
                                first = T.alloc_var(T.int32, 0)
                                for bit in T.unroll(rank_bits):
                                    next = first + ((SHORTLIST // 2) >> bit)
                                    if next < active and prefix[next - 1] <= target:
                                        first = next
                                cut = first
                            total = prefix[cut]
                            target = T.min(
                                T.cast(Parameters[5] * T.cast(total, T.float64), T.uint64),
                                total - 1,
                            )
                            first = T.alloc_var(T.int32, 0)
                            for bit in T.unroll(rank_bits):
                                next = first + ((SHORTLIST // 2) >> bit)
                                if next <= cut and prefix[next - 1] <= target:
                                    first = next
                            Output[0] = token_order[first]

                elif T.get_thread_binding() == 0:
                    # Leave the row pending for the exact radix fallback.
                    Output[0] = 0

    return main


@orin_jit
def fallback_masses(vocab: int):
    # Completed rows retain their sample and never enter the radix stages.
    blocks = (vocab + TILE - 1) // TILE
    scale = 1 << min(52, 63 - (vocab - 1).bit_length())

    @T.prim_func
    def main(
        Scores: T.Tensor((vocab,), T.float64),
        Keys: T.Tensor((vocab,), T.uint64),
        State: T.Tensor((7,), T.uint64),
        Invalid: T.Tensor((blocks,), T.int32),
        Parameters: T.Tensor((6,), T.float64),
        Mass: T.Tensor((vocab,), T.uint64),
        Output: T.Tensor((2,), T.int32),
    ):
        with T.Kernel(blocks, threads=256) as b:
            if Invalid[0] == 0:
                maximum = T.reinterpret(T.float64, State[4])
                for i in T.Parallel(TILE):
                    j = b * TILE + i
                    if j < vocab:
                        Mass[j] = 0
                        if Keys[j] != 0:
                            Mass[j] = _fixed_mass(Scores[j], maximum, Parameters[0], scale)
                if b == 0 and T.get_thread_binding() == 0:
                    Output[0] = 0
                    Output[1] = 0

    return main
