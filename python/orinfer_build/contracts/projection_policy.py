"""Build-time decode shapes; serving consumes only compiled row selections."""

from dataclasses import dataclass
import math

CAPACITY = 128
SPECIALIZATIONS = (1, 2, 3, 4, 8, 16, 32)


@dataclass(frozen=True)
class DecodeShape:
    rows: int
    dynamic: bool = False

    @property
    def key(self):
        return "dynamic" if self.dynamic else f"m{self.rows}"

    def ranges(self):
        if not self.dynamic:
            return [(self.rows, self.rows)]
        result, first = [], 1
        for row in (*SPECIALIZATIONS, CAPACITY + 1):
            if first < row:
                result.append((first, row - 1))
            first = row + 1
        return result


def decode_shapes():
    return [DecodeShape(row) for row in SPECIALIZATIONS] + [DecodeShape(CAPACITY, True)]


def selection_name(slot, minimum, maximum, *, skip=False):
    if not 1 <= minimum <= maximum <= CAPACITY:
        raise ValueError("Invalid decode selection range")
    return f"{slot}/{'skip' if skip else 'rows'}/{minimum}-{maximum}"


def q4_configs(rows):
    """Small, legal search space, shared by exact and runtime-row compilation."""
    bm = min(rows, 8)
    return (
        [dict(algorithm="dp4a", block_m=bm, block_n=n, block_k=128, stages=2) for n in (8, 16)]
        + [
            dict(algorithm="tensor_core", block_m=m, block_n=n, block_k=128, stages=2)
            for m in (8, 16, 32, 64)
            for n in (64, 128)
        ]
        + [
            dict(algorithm="tensor_core", block_m=128, block_n=n, block_k=k, stages=s)
            for n in (64, 128)
            for k, s in ((128, 1), (128, 2), (256, 1))
        ]
        + [
            # Measured narrow-N tiles reduce compressed-weight decode overhead.
            # Keep them in the common search for every codec and live row count.
            dict(algorithm="tensor_core", block_m=8, block_n=32, block_k=256, stages=2),
            dict(algorithm="tensor_core", block_m=32, block_n=32, block_k=256, stages=1),
        ]
    )


def e8p_q4_configs(rows, *, direct=False):
    """Exact INT8 and direct INT4 schedules share one measured selection."""
    configs = q4_configs(rows) + [
        # More activation rows reuse each decoded E8P slab. Selection still
        # measures every workload; larger live sets can make this tile slower.
        dict(algorithm="tensor_core", block_m=256, block_n=64, block_k=128, stages=1)
    ]
    # Traversal and a smaller final tile change scheduling, not quantization.
    # Both compete at every measured row count, including small decode shapes.
    large = configs[-1]
    configs += [
        large | options
        for options in (
            {"group_m": 1},
            {"group_m": 1, "cache_scales": True},
            {"tail_m": 16},
            {"group_m": 1, "tail_m": 16},
        )
    ]
    configs += [
        dict(
            algorithm="tensor_core",
            block_m=m,
            block_n=64,
            block_k=128,
            stages=1,
            group_m=1,
        )
        for m in (32, 96, 192)
    ]
    if direct:
        configs += [
            dict(algorithm="int4_direct", block_m=8, block_n=n, block_k=128, stages=2, split_k=s)
            for n in (32, 64, 128)
            for s in (1, 2, 4)
            if n * s <= 256
        ]
    return configs


def w8_configs(rows, outputs):
    if outputs <= 0:
        raise ValueError("Invalid W8 output width")
    return (
        [dict(block_m=m, block_n=n, block_k=128) for m in (16, 32, 64, 128) for n in (64, 128)]
        + [dict(block_m=128, block_n=128, block_k=128, group_m=4)]
        + [
            dict(block_m=m, block_n=n, block_k=128, group_m=1, cache_scales=True)
            for m, n in ((32, 64), (64, 64), (128, 128), (256, 64))
        ]
        + [dict(algorithm="dp4a", block_m=min(rows, 4), block_n=n, block_k=128) for n in (4, 8, 16)]
    )


def encoder_configs(kind):
    """Keep the encoder subset tied to the shared legal tile catalog."""
    if kind == "w8":
        tiles = {(32, 64), (64, 64), (128, 64), (128, 128), (256, 64)}
        return [
            c
            for c in w8_configs(CAPACITY, 1)
            if c.get("algorithm", "tensor_core") == "tensor_core"
            and (c["block_m"], c["block_n"]) in tiles
        ]
    if kind != "e8p_q4":
        raise ValueError("Unsupported encoder projection codec")
    tiles = {
        (32, 64, 1),
        (32, 64, 2),
        (64, 64, 2),
        (96, 64, 1),
        (128, 64, 1),
        (128, 128, 1),
        (128, 128, 2),
        (192, 64, 1),
        (256, 64, 1),
    }
    return [
        c
        for c in e8p_q4_configs(CAPACITY)
        if c["algorithm"] == "tensor_core"
        and c["block_k"] == 128
        and not c.get("tail_m")
        and (c["block_m"], c["block_n"], c["stages"]) in tiles
    ]


def swiglu_configs(rows):
    return [
        dict(algorithm="tensor_core", block_m=m, block_n=n, block_k=128)
        for m in (16, 32)
        for n in (64, 128)
    ] + [dict(algorithm="dp4a", block_m=min(rows, 4), block_n=n, block_k=128) for n in (4, 8, 16)]


def candidate_scores(profiles, invalid, objective="relative_geomean"):
    """Compare the same measured workload without imposing runtime row cutoffs.

    A dynamic prefill template minimizes total GPU time for its representative
    requests. Decode fallbacks retain balanced relative costs across row counts.
    """
    if objective not in ("relative_geomean", "total_latency"):
        raise ValueError("Unknown dynamic tuning objective")
    width = len(profiles[0]["times_ms"])
    legal = [i for i in range(width) if i not in invalid]
    if not legal:
        return [float("inf")] * width
    minima = [min(p["times_ms"][i] for i in legal) for p in profiles]
    scores = []
    for i in range(width):
        if i in invalid:
            scores.append(float("inf"))
        elif objective == "total_latency":
            scores.append(sum(p["times_ms"][i] for p in profiles) / sum(minima))
        else:
            scores.append(
                math.exp(
                    sum(math.log(p["times_ms"][i] / best) for p, best in zip(profiles, minima))
                    / len(profiles)
                )
            )
    return scores
