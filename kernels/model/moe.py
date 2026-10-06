"""Explicit routing and deterministic mixture reduction for SM87 MoE.

Router logits and probabilities remain FP32. Top-k uses ascending expert IDs
to break exact ties, and can normalize over the selected experts or all experts.
These kernels do not copy or expand weights. Dispatch owns the output-slot map.
"""
import tilelang.language as T

from tools.operators.common import orin_jit


@orin_jit
def expert_histogram(M: int, E: int = 512, top_k: int = 10):
    """Build (IDs, Counts[E], RelativeSlot[M,top_k]), all INT32.

    Stable assignment order is token first, route rank second. No atomics or
    host synchronization. IDs must be in [0,E); each token's IDs are unique.
    """
    if any(type(x) is not int or x <= 0 for x in (M, E, top_k)) or top_k > E:
        raise ValueError('Invalid MoE dispatch dimensions')
    width = 1 << (M * top_k - 1).bit_length()

    @T.prim_func
    def main(IDs: T.Tensor((M, top_k), T.int32),
             Counts: T.Tensor((E,), T.int32),
             RelativeSlot: T.Tensor((M, top_k), T.int32)):
        with T.Kernel(E, threads=128) as expert:
            matches = T.alloc_fragment((width,), T.int32)
            prefix = T.alloc_fragment((width,), T.int32)
            for j in T.Parallel(width):
                matches[j] = T.if_then_else(j < M * top_k,
                    T.cast(IDs[j // top_k, j % top_k] == expert, T.int32), 0)
                prefix[j] = matches[j]
            T.cumsum(prefix, dim=0)
            for j in T.Parallel(width):
                if j == width - 1:
                    Counts[expert] = prefix[j]
                if j < M * top_k and matches[j] != 0:
                    RelativeSlot[j // top_k, j % top_k] = prefix[j] - 1
    return main


@orin_jit
def expert_offsets(E: int = 512, block_m: int = 16):
    """Build (Counts, RowOffsets[E], TileOffsets[E], TileCount[1]), INT32."""
    if type(E) is not int or E <= 0 or block_m != 16:
        raise ValueError('Invalid MoE expert prefix dimensions')
    width = 1 << (E - 1).bit_length()

    @T.prim_func
    def main(Counts: T.Tensor((E,), T.int32),
             RowOffsets: T.Tensor((E,), T.int32),
             TileOffsets: T.Tensor((E,), T.int32),
             TileCount: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=128):
            counts = T.alloc_fragment((width,), T.int32)
            rows = T.alloc_fragment((width,), T.int32)
            tiles = T.alloc_fragment((width,), T.int32)
            for e in T.Parallel(width):
                counts[e] = T.if_then_else(e < E, Counts[e], 0)
                rows[e] = counts[e]
                tiles[e] = T.ceildiv(counts[e], block_m)
            T.cumsum(rows, dim=0)
            T.cumsum(tiles, dim=0)
            for e in T.Parallel(E):
                RowOffsets[e] = rows[e] - counts[e]
                TileOffsets[e] = tiles[e] - T.ceildiv(counts[e], block_m)
            for e in T.Parallel(width):
                if e == width - 1:
                    TileCount[0] = tiles[e]
    return main


@orin_jit
def expert_tiles(M: int, E: int = 512, top_k: int = 10, block_m: int = 16):
    """Build (Counts, TileOffsets, TileExpert, TileRow), all INT32.

    Output capacity ceil(M*top_k/16)+min(E,M*top_k) bounds all distributions.
    Only the prefix indicated by expert_offsets.TileCount is initialized/read.
    """
    if any(type(x) is not int or x <= 0 for x in (M, E, top_k)) or block_m != 16:
        raise ValueError('Invalid MoE tile dimensions')
    assignments = M * top_k
    capacity = (assignments + 15) // 16 + min(E, assignments)
    expert_capacity = (M + 15) // 16  # Each expert is selected at most once per token.

    @T.prim_func
    def main(Counts: T.Tensor((E,), T.int32),
             TileOffsets: T.Tensor((E,), T.int32),
             TileExpert: T.Tensor((capacity,), T.int32),
             TileRow: T.Tensor((capacity,), T.int32)):
        with T.Kernel(E, threads=128) as e:
            for tile in T.Parallel(expert_capacity):
                if tile * block_m < Counts[e]:
                    destination = TileOffsets[e] + tile
                    TileExpert[destination] = e
                    TileRow[destination] = tile * block_m
    return main


@orin_jit
def expert_dispatch(M: int, H: int, E: int = 512, top_k: int = 10, scale_group: int = 64):
    """Build (A, Scale, IDs, RelativeSlot, RowOffsets, D, DS, SlotMap).

    A[M,H]/D[M*top_k,H] are INT8; Scale/DS are FP16. Others INT32.
    scale_group defaults to64; H uses one per-token scale for Q2I8.
    Dispatch copies quantized activations only. Source/destination must be disjoint.
    Inputs come from router_topk/expert_histogram/expert_offsets, without invalid
    or duplicate per-token IDs. The SlotMap is consumed by moe_combine.
    """
    if any(type(x) is not int or x <= 0 for x in (M, H, E, top_k)) or H % 64 or top_k > E:
        raise ValueError('Invalid MoE activation dispatch dimensions')
    if type(scale_group) is not int or scale_group <= 0 or H % scale_group:
        raise ValueError('Invalid activation scale group')
    groups = H // scale_group
    @T.prim_func
    def main(A: T.Tensor((M, H), T.int8),
             Scale: T.Tensor((M, groups), T.float16),
             IDs: T.Tensor((M, top_k), T.int32),
             RelativeSlot: T.Tensor((M, top_k), T.int32),
             RowOffsets: T.Tensor((E,), T.int32),
             D: T.Tensor((M * top_k, H), T.int8),
             DS: T.Tensor((M * top_k, groups), T.float16),
             SlotMap: T.Tensor((M, top_k), T.int32)):
        with T.Kernel(M * top_k, threads=128) as assignment:
            row, rank = assignment // top_k, assignment % top_k
            slot = RowOffsets[IDs[row, rank]] + RelativeSlot[row, rank]
            SlotMap[row, rank] = slot
            for col in T.Parallel(H):
                D[slot, col] = A[row, col]
            for group in T.Parallel(groups):
                DS[slot, group] = Scale[row, group]
    return main


@orin_jit
def router_topk(M: int, E: int = 512, top_k: int = 10, renormalize: bool = True):
    """Build (Logits[M,E], IDs[M,top_k], Prob[M,top_k]), F32/I32/F32.

    Logits must be finite. The normalized route is equivalent to FP32 softmax,
    top-k and renormalization, while avoiding a full probability materialization.
    """
    if any(type(x) is not int or x <= 0 for x in (M, E, top_k)) or top_k > E:
        raise ValueError('Invalid MoE router dimensions')
    if type(renormalize) is not bool:
        raise ValueError('renormalize must be a boolean')
    width = 1 << (E - 1).bit_length()
    selected_width = 1 << (top_k - 1).bit_length()

    @T.prim_func
    def main(Logits: T.Tensor((M, E), T.float32),
             IDs: T.Tensor((M, top_k), T.int32),
             Prob: T.Tensor((M, top_k), T.float32)):
        with T.Kernel(M, threads=128) as row:
            values = T.alloc_fragment((width,), T.float32)
            candidates = T.alloc_fragment((width,), T.int32)
            maximum = T.alloc_fragment((1,), T.float32)
            selected_id = T.alloc_fragment((1,), T.int32)
            selected = T.alloc_shared((selected_width,), T.float32)
            numerator = T.alloc_fragment((selected_width,), T.float32)
            denominator = T.alloc_fragment((1,), T.float32)
            full_exp = T.alloc_fragment((width,), T.float32)
            global_max = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(width):
                values[j] = T.if_then_else(j < E, Logits[row, j], -T.infinity(T.float32))
            T.reduce_max(values, global_max, dim=0)
            if not renormalize:
                for j in T.Parallel(width):
                    full_exp[j] = T.exp(values[j] - global_max[0])
                T.reduce_sum(full_exp, denominator, dim=0)
            for rank in T.serial(top_k):
                T.reduce_max(values, maximum, dim=0)
                for j in T.Parallel(width):
                    candidates[j] = T.if_then_else(j < E and values[j] == maximum[0], j, 2147483647)
                T.reduce_min(candidates, selected_id, dim=0)
                for j in T.Parallel(width):
                    if j == selected_id[0]:
                        values[j] = -T.infinity(T.float32)
                IDs[row, rank] = selected_id[0]
                selected[rank] = maximum[0]
            for rank in T.Parallel(selected_width):
                numerator[rank] = T.if_then_else(rank < top_k,
                    T.exp(selected[rank] - global_max[0]), 0.0)
            if renormalize:
                T.reduce_sum(numerator, denominator, dim=0)
            for rank in T.Parallel(top_k):
                Prob[row, rank] = numerator[rank] / denominator[0]
    return main


@orin_jit
def moe_combine(M: int, H: int, slots: int, top_k: int = 10):
    """Build (Expert, SlotMap, Prob, Shared, SharedGate, Output).

    Expert[slots,H], Shared[M,H], SharedGate[M] and Output[M,H] are FP16.
    SlotMap[M,top_k] is INT32; Prob[M,top_k] is FP32. Invalid slots are ignored.
    SharedGate contains raw gate logits. Accumulation and sigmoid are FP32,
    with one final FP16 cast. Each output element has one owner; no atomics.
    """
    if any(type(x) is not int or x <= 0 for x in (M, H, slots, top_k)):
        raise ValueError('Invalid MoE mixture dimensions')
    width = 256

    @T.prim_func
    def main(Expert: T.Tensor((slots, H), T.float16),
             SlotMap: T.Tensor((M, top_k), T.int32),
             Prob: T.Tensor((M, top_k), T.float32),
             Shared: T.Tensor((M, H), T.float16),
             SharedGate: T.Tensor((M,), T.float16),
             Output: T.Tensor((M, H), T.float16)):
        with T.Kernel(M, T.ceildiv(H, width), threads=128) as (row, block):
            acc = T.alloc_fragment((width,), T.float32)
            T.clear(acc)
            for rank in T.serial(top_k):
                slot = SlotMap[row, rank]
                if slot >= 0 and slot < slots:
                    for j in T.Parallel(width):
                        col = block * width + j
                        if col < H:
                            acc[j] += T.cast(Expert[slot, col], T.float32) * Prob[row, rank]
            gate = 1.0 / (1.0 + T.exp(-T.cast(SharedGate[row], T.float32)))
            for j in T.Parallel(width):
                col = block * width + j
                if col < H:
                    Output[row, col] = acc[j] + gate * T.cast(Shared[row, col], T.float32)
    return main
