"""SM87 SwiGLU: FP16 input/output, FP32 nonlinear arithmetic, no workspace.

``split`` is the canonical [gate | up] layout. ``interleaved`` is physically
[gate0, up0, gate1, up1, ...]; it requires an upstream layout change or packing.
The dynamic-M cubin supports arbitrary positive row counts. Use explicit output
buffers and ``launch(..., stream=<CUDA stream handle>)`` for stable graph ABI.
"""
import tilelang
import tilelang.language as T

HIDDEN = 17408


@T.macro
def swiglu_fp16(gate, up):
    """Fusion expression; the returned value has the required FP16 rounding.

    A downstream A8 quantizer must cast this FP16 value back to FP32 before its
    amax/scale/code calculation to match an unfused FP16 SwiGLU -> A8 path.
    """
    g = T.cast(gate, T.float32)
    u = T.cast(up, T.float32)
    e = T.exp(-T.abs(g))
    sigmoid = T.if_then_else(g >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return T.cast((g * sigmoid) * u, T.float16)


@tilelang.jit(out_idx=[], execution_backend="nvrtc",
              target={"kind": "cuda", "arch": "sm_87"})
def _compile_swiglu(block: int = 1024, threads: int = 256, layout: str = "split"):
    assert layout in ("split", "interleaved")
    assert block > 0 and block % threads == 0
    rows = T.dynamic("rows")

    @T.prim_func
    def main(X: T.Tensor((rows, 2 * HIDDEN), T.float16),
             Y: T.Tensor((rows, HIDDEN), T.float16)):
        with T.Kernel(T.ceildiv(rows * HIDDEN, block), threads=threads) as bx:
            gate = T.alloc_fragment((block,), T.float32)
            up = T.alloc_fragment((block,), T.float32)
            result = T.alloc_fragment((block,), T.float16)
            for j in T.Parallel(block):
                index = bx * block + j
                row = index // HIDDEN
                col = index % HIDDEN
                if index < rows * HIDDEN:
                    if layout == "split":
                        gate[j] = X[row, col]
                        up[j] = X[row, HIDDEN + col]
                    else:
                        gate[j] = X[row, 2 * col]
                        up[j] = X[row, 2 * col + 1]
                    result[j] = swiglu_fp16(gate[j], up[j])
            for j in T.Parallel(block):
                index = bx * block + j
                if index < rows * HIDDEN:
                    Y[index // HIDDEN, index % HIDDEN] = result[j]

    return main


def swiglu(block=1024, threads=256, layout="split"):
    """Compile/load dynamic-M code and isolate TileLang's entry mapping."""
    kernel = _compile_swiglu(block, threads, layout)
    kernel.adapter.kernels = dict(kernel.adapter.kernels)
    return kernel


def launch(kernel, input_tensor, output_tensor, *, stream):
    """Launch an already compiled kernel on the caller's explicit stream."""
    return kernel.adapter.func(input_tensor, output_tensor, stream=stream)
