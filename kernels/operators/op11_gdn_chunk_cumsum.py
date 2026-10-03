"""SM87 FP32 GDN per-chunk inclusive scan, with tokenmajor gate packing.

No Torch math or allocations. Buffers and explicit caller stream are supplied.
Each [request, head, chunk] scan starts at zero; no recurrent state is owned.
"""
import operator

import tilelang
import tilelang.language as T

HEADS = 48


def validate_lengths(lengths, *, batch, tokens):
    """Validate a host sequence before upload/update, including graph updates.

    Returns immutable Python ints; no device readback or implicit synchronization.
    The Rust caller must perform the same 0 <= length <= T int32 validation.
    """
    if type(batch) is not int or type(tokens) is not int or not 0 < batch < 2**31 or not 0 < tokens < 2**31:
        raise ValueError("batch/tokens must be positive Python ints, tokens < 2**31")
    values = tuple(lengths)
    if len(values) != batch:
        raise ValueError("lengths must contain exactly B entries")
    result = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("lengths must contain integers, not bools")
        try:
            value = operator.index(value)
        except TypeError as exc:
            raise ValueError("lengths must contain integers") from exc
        if not 0 <= value <= tokens:
            raise ValueError("length must satisfy 0 <= length <= T")
        result.append(value)
    return tuple(result)


@tilelang.jit(out_idx=[], execution_backend="nvrtc",
              target={"kind": "cuda", "arch": "sm_87"})
def _compile_headmajor(bt: int = 64, heads_tile: int = 4, threads: int = 128):
    assert bt in (16, 32, 64)
    assert heads_tile in (4, 8, 16) and HEADS % heads_tile == 0
    assert threads == 128
    batch, chunks = T.dynamic("batch"), T.dynamic("chunks")

    @T.prim_func
    def main(G: T.Tensor((batch, HEADS, chunks, bt), "float32"),
             CumulativeG: T.Tensor((batch, HEADS, chunks, bt), "float32")):
        with T.Kernel(HEADS // heads_tile, chunks, batch, threads=threads) as (hh, c, b):
            scan = T.alloc_shared((heads_tile, bt), "float32")
            last_nonzero = T.alloc_shared((heads_tile, bt), "int32")
            for h, t in T.Parallel(heads_tile, bt):
                scan[h, t] = G[b, hh * heads_tile + h, c, t]
                last_nonzero[h, t] = T.if_then_else(scan[h, t] != 0.0, t, -1)
            T.cumsum(scan, dim=1)
            T.cummax(last_nonzero, dim=1)
            for h, t in T.Parallel(heads_tile, bt):
                if last_nonzero[h, t] >= 0:
                    CumulativeG[b, hh * heads_tile + h, c, t] = scan[h, last_nonzero[h, t]]
                else:
                    CumulativeG[b, hh * heads_tile + h, c, t] = 0.0
    return main


@tilelang.jit(out_idx=[], execution_backend="nvrtc",
              target={"kind": "cuda", "arch": "sm_87"})
def _compile_pack(bt: int = 64, heads_tile: int = 4, threads: int = 128,
                  beta_dtype: str = "float32"):
    assert bt in (16, 32, 64)
    assert heads_tile in (4, 8, 16) and HEADS % heads_tile == 0
    assert threads == 128 and beta_dtype in ("float16", "float32")
    batch, tokens = T.dynamic("batch"), T.dynamic("tokens")
    chunks = T.ceildiv(tokens, bt)

    @T.prim_func
    def main(G: T.Tensor((batch, tokens, HEADS), "float32"),
             Beta: T.Tensor((batch, tokens, HEADS), beta_dtype),
             Lengths: T.Tensor((batch,), "int32"),
             CumulativeG: T.Tensor((batch, HEADS, chunks, bt), "float32"),
             PaddedBeta: T.Tensor((batch, HEADS, chunks, bt), "float32")):
        with T.Kernel(HEADS // heads_tile, chunks, batch, threads=threads) as (hh, c, b):
            scan = T.alloc_shared((heads_tile, bt), "float32")
            length = T.min(T.max(Lengths[b], 0), tokens)
            for h, t in T.Parallel(heads_tile, bt):
                token, head = c * bt + t, hh * heads_tile + h
                # Branch before reading: NaN/Inf padding never enters the scan.
                if token < length:
                    scan[h, t] = G[b, token, head]
                    PaddedBeta[b, head, c, t] = T.cast(Beta[b, token, head], "float32")
                else:
                    scan[h, t] = 0.0
                    PaddedBeta[b, head, c, t] = 0.0
            T.cumsum(scan, dim=1)
            for h, t in T.Parallel(heads_tile, bt):
                if c * bt + t < length:
                    CumulativeG[b, hh * heads_tile + h, c, t] = scan[h, t]
                elif length > c * bt:
                    # Preserve the exact last valid FP32 prefix, even when the
                    # parallel addition tree for padded zeros would differ.
                    CumulativeG[b, hh * heads_tile + h, c, t] = scan[h, length - c * bt - 1]
                else:
                    CumulativeG[b, hh * heads_tile + h, c, t] = 0.0
    return main


def gdn_chunk_cumsum(bt=64, heads_tile=4):
    """Headmajor FP32 [B,48,C,BT] -> same layout; caller zero-pads g."""
    kernel = _compile_headmajor(bt, heads_tile)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


def gdn_pack_chunk_cumsum(bt=64, heads_tile=4, beta_dtype="float32"):
    """Tokenmajor -> cumulative G and padded FP32 beta, no Q/K/V copy.

    op09 FP32 beta is default; beta_dtype=float16 explicitly imports rounded
    native beta without recovering the lost precision. g always remains FP32.
    Caller validates host lengths on each update, then uploads int32[B]. All
    buffers are contiguous, disjoint, correctly sized and stable for graphs.
    Valid-token NaN/Inf propagate; this operation does not validate gate values.
    Device clamp is defensive, not acceptance of an invalid host request.
    """
    kernel = _compile_pack(bt, heads_tile, beta_dtype=beta_dtype)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


def launch(kernel, g, cumulative_g, *, stream):
    return kernel.adapter.func(g, cumulative_g, stream=stream)


def launch_pack(kernel, g, beta, lengths, cumulative_g, padded_beta, *, stream):
    return kernel.adapter.func(g, beta, lengths, cumulative_g, padded_beta, stream=stream)
