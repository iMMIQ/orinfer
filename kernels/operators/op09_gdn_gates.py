"""SM87 GDN gates. Explicit FP32 outputs; dynamic M, 48 value heads.

No Torch/reference calls occur here. ``parameter_mode='exp_a'`` consumes an
offline FP32 exp(A_log) buffer instead of A_log; its preparation identity and
rounding are caller responsibilities. Every launch uses the caller's stream.
"""
import tilelang
import tilelang.language as T

HEADS = 48


@T.macro
def stable_softplus(x):
    return T.max(x, 0.0) + T.call_extern("float32", "log1pf", T.exp(-T.abs(x)))


@T.macro
def stable_sigmoid(x):
    e = T.exp(-T.abs(x))
    return T.if_then_else(x >= 0.0, 1.0 / (1.0 + e), e / (1.0 + e))


@tilelang.jit(out_idx=[], execution_backend="nvrtc",
              target={"kind": "cuda", "arch": "sm_87"})
def _compile_gates(dtype: str = "float16", block: int = 256,
                   threads: int = 128, parameter_mode: str = "a_log",
                   packed_ab: bool = False, beta_round_fp16: bool = False):
    assert dtype in ("float16", "float32")
    assert parameter_mode in ("a_log", "exp_a")
    assert block >= threads and block % threads == 0
    rows = T.dynamic("rows")
    input_columns = HEADS * (2 if packed_ab else 1)
    b_offset = HEADS if packed_ab else 0

    @T.prim_func
    def main(A: T.Tensor((rows, input_columns), dtype),
             B: T.Tensor((rows, input_columns), dtype),
             Parameter: T.Tensor((HEADS,), "float32"),
             DtBias: T.Tensor((HEADS,), "float32"),
             G: T.Tensor((rows, HEADS), "float32"),
             Beta: T.Tensor((rows, HEADS), "float32")):
        with T.Kernel(T.ceildiv(rows * HEADS, block), threads=threads) as bx:
            g = T.alloc_fragment((block,), "float32")
            beta = T.alloc_fragment((block,), "float32")
            for j in T.Parallel(block):
                idx = bx * block + j
                if idx < rows * HEADS:
                    h = idx % HEADS
                    x = T.cast(A[idx // HEADS, h], "float32") + DtBias[h]
                    if parameter_mode == "a_log":
                        g[j] = -T.exp(Parameter[h]) * stable_softplus(x)
                    else:
                        g[j] = -Parameter[h] * stable_softplus(x)
                    beta[j] = stable_sigmoid(T.cast(B[idx // HEADS, h + b_offset], "float32"))
                    if beta_round_fp16:
                        beta[j] = T.cast(T.cast(beta[j], "float16"), "float32")
            for j in T.Parallel(block):
                idx = bx * block + j
                if idx < rows * HEADS:
                    G[idx // HEADS, idx % HEADS] = g[j]
                    Beta[idx // HEADS, idx % HEADS] = beta[j]

    return main


def gdn_gates(dtype="float16", block=256, threads=128, parameter_mode="a_log",
              packed_ab=False, beta_round_fp16=False):
    """Compile/load one dynamic-M cubin, isolating TileLang's entry mapping."""
    kernel = _compile_gates(dtype, block, threads, parameter_mode, packed_ab, beta_round_fp16)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


def launch(kernel, a, b, parameter, dt_bias, g, beta, *, stream):
    """Inputs contiguous; outputs disjoint FP32 [M,48].

    For packed_ab=True, pass the same op07 [M,96] allocation as both a and b.
    The compiled kernel reads a at offset 0 and b at offset 48 with stride 96;
    read-only input aliasing is intentional and requires no repack/copy.
    """
    return kernel.adapter.func(a, b, parameter, dt_bias, g, beta, stream=stream)
