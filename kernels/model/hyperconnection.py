"""Qwen4 gated residual streams with explicit low-precision boundaries.

Norm weights are zero-centered FP32. Linear projections retain BF16 weights;
state/output dtype is chosen by the caller. FP32 reductions do not change the
materialization boundaries of norm, SiLU, sigmoid, product or residual updates.

Elementwise loops use scalar coalescing: the pinned TileLang/NVRTC combination
can lower vectorized BF16 conversions to trap-only cubins on SM87. Keeping
coalesced_width as an explicit IntImm avoids that lowering; the projections
still use BF16 Tensor Cores.
"""
import tilelang
import tilelang.language as T

from tools.operators.common import orin_jit


def _dtype(value):
    if value not in ('float16', 'bfloat16'):
        raise ValueError('HC stream dtype must be FP16 or BF16')
    return value


@orin_jit
def hc_norm(M: int, H: int, streams: int = 4, eps: float = 1e-6, dtype: str = 'bfloat16'):
    """Build (Residual[M,streams,H], Weight[streams,H], Normed), dtype/F32/dtype."""
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, H, streams)) or not 0 < eps < 1:
        raise ValueError('Invalid HC normalization dimensions/epsilon')
    width = 1 << (H - 1).bit_length()
    @T.prim_func
    def main(Residual: T.Tensor((M, streams, H), dtype),
             Weight: T.Tensor((streams, H), T.float32),
             Normed: T.Tensor((M, streams, H), dtype)):
        with T.Kernel(M, streams, threads=256) as (row, branch):
            values = T.alloc_fragment((width,), T.float32)
            square = T.alloc_fragment((width,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(width, coalesced_width=T.int32(1)):
                values[j] = T.if_then_else(j < H, T.cast(Residual[row, branch, j], T.float32), 0.0)
                square[j] = values[j] * values[j]
            T.reduce_sum(square, total, dim=0)
            for j in T.Parallel(H, coalesced_width=T.int32(1)):
                Normed[row, branch, j] = values[j] * T.rsqrt(total[0] / H + eps) * (1.0 + Weight[branch, j])
    return main


@orin_jit
def hc_silu(M: int, rank: int, streams: int = 4, dtype: str = 'bfloat16'):
    """Build (Down[M,rank], Activated), dividing by stream count before SiLU."""
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, rank, streams)):
        raise ValueError('Invalid HC low-rank dimensions')
    @T.prim_func
    def main(Down: T.Tensor((M, rank), dtype), Activated: T.Tensor((M, rank), dtype)):
        with T.Kernel(M, threads=128) as row:
            for j in T.Parallel(rank, coalesced_width=T.int32(1)):
                v = T.cast(T.cast(T.cast(Down[row, j], T.float32) / streams, dtype), T.float32)
                Activated[row, j] = v / (1.0 + T.exp(-v))
    return main


@orin_jit
def hc_mix(M: int, H: int, streams: int = 4, dtype: str = 'bfloat16'):
    """Build (Normed[M,streams,H], Up[M,streams,H], Mixed[M,H])."""
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, H, streams)):
        raise ValueError('Invalid HC mixing dimensions')
    @T.prim_func
    def main(Normed: T.Tensor((M, streams, H), dtype),
             Up: T.Tensor((M, streams, H), dtype),
             Mixed: T.Tensor((M, H), dtype)):
        with T.Kernel(M, T.ceildiv(H, 256), threads=128) as (row, block):
            total = T.alloc_fragment((256,), T.float32)
            T.clear(total)
            for branch in T.serial(streams):
                for j in T.Parallel(256, coalesced_width=T.int32(1)):
                    col = block * 256 + j
                    if col < H:
                        gate = T.cast(1.0 / (1.0 + T.exp(-T.cast(Up[row, branch, col], T.float32))), dtype)
                        product = T.cast(T.cast(gate, T.float32) * T.cast(Normed[row, branch, col], T.float32), dtype)
                        total[j] += T.cast(product, T.float32)
            for j in T.Parallel(256, coalesced_width=T.int32(1)):
                if block * 256 + j < H:
                    Mixed[row, block * 256 + j] = total[j] / streams
    return main


@orin_jit
def hc_combine(M: int, H: int, streams: int = 4, dtype: str = 'bfloat16'):
    """Build (Block[M,H], Residual[M,streams,H], Inject[M,streams], Output).

    Inject is the linear projection of normed residuals before division/sigmoid.
    Output may alias Residual: each element has exactly one owner and is read
    before its write. The Block and Inject inputs must remain disjoint.
    """
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, H, streams)):
        raise ValueError('Invalid HC combination dimensions')
    @T.prim_func
    def main(Block: T.Tensor((M, H), dtype),
             Residual: T.Tensor((M, streams, H), dtype),
             Inject: T.Tensor((M, streams), dtype),
             Output: T.Tensor((M, streams, H), dtype)):
        with T.Kernel(M, streams, T.ceildiv(H, 256), threads=128) as (row, branch, block):
            raw = T.cast(T.cast(T.cast(Inject[row, branch], T.float32) / streams, dtype), T.float32)
            gate = T.cast(2.0 * T.cast(T.cast(1.0 / (1.0 + T.exp(-raw)), dtype), T.float32), dtype)
            for j in T.Parallel(256, coalesced_width=T.int32(1)):
                col = block * 256 + j
                if col < H:
                    injection = T.cast(T.cast(Block[row, col], T.float32) * T.cast(gate, T.float32), dtype)
                    Output[row, branch, col] = T.cast(Residual[row, branch, col], T.float32) + T.cast(injection, T.float32)
    return main


@orin_jit
def hc_combine_norm(M: int, H: int, streams: int = 4, eps: float = 1e-6,
                    dtype: str = 'bfloat16'):
    """Combine residuals and normalize, retaining the rounded residual output.

    Output may alias Residual. Normed must remain disjoint from all inputs and
    Output. Each CTA owns one whole row/stream, including the FP32 reduction.
    The intervening stream-dtype rounding matches hc_combine then hc_norm.
    """
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, H, streams)) or not 0 < eps < 1:
        raise ValueError('Invalid HC combination/normalization dimensions')
    width = 1 << (H - 1).bit_length()
    @T.prim_func
    def main(Block: T.Tensor((M, H), dtype),
             Residual: T.Tensor((M, streams, H), dtype),
             Inject: T.Tensor((M, streams), dtype),
             Weight: T.Tensor((streams, H), T.float32),
             Output: T.Tensor((M, streams, H), dtype),
             Normed: T.Tensor((M, streams, H), dtype)):
        with T.Kernel(M, streams, threads=256) as (row, branch):
            values = T.alloc_fragment((width,), T.float32)
            square = T.alloc_fragment((width,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            raw = T.cast(T.cast(T.cast(Inject[row, branch], T.float32) / streams, dtype), T.float32)
            gate = T.cast(2.0 * T.cast(T.cast(1.0 / (1.0 + T.exp(-raw)), dtype), T.float32), dtype)
            for j in T.Parallel(width, coalesced_width=T.int32(1)):
                values[j] = 0.0
                if j < H:
                    injection = T.cast(T.cast(Block[row, j], T.float32) * T.cast(gate, T.float32), dtype)
                    rounded = T.cast(T.cast(Residual[row, branch, j], T.float32) + T.cast(injection, T.float32), dtype)
                    values[j] = T.cast(rounded, T.float32)
                    Output[row, branch, j] = rounded
                square[j] = values[j] * values[j]
            T.reduce_sum(square, total, dim=0)
            for j in T.Parallel(H, coalesced_width=T.int32(1)):
                Normed[row, branch, j] = values[j] * T.rsqrt(total[0] / H + eps) * (1.0 + Weight[branch, j])
    return main


@orin_jit
def hc_projection(M: int, N: int, K: int, block_m: int = 16, dtype: str = 'bfloat16', block_n: int = 64,
                  silu: bool = False, streams: int = 4, group_m: int = 0):
    """Build (A[M,K], W[N,K], C[M,N]), stream dtype/BF16/stream dtype.

    FP32 accumulation, BF16 Tensor Core arithmetic, explicit output rounding.
    This preserves the original BF16 HC weights. N tails are supported for
    the four-stream injection projection; K must be a multiple of 64.
    """
    _dtype(dtype)
    if any(type(x) is not int or x <= 0 for x in (M, N, K)) or K % 64 or block_m not in (16, 32, 64, 128):
        raise ValueError('Invalid HC projection dimensions')
    if block_n not in (16,32,64,128):raise ValueError('Invalid HC output tile')
    if type(group_m) is not int or group_m not in (0,1,2,4,8):
        raise ValueError('Invalid HC row grouping')
    threads=64 if block_n==16 else 128
    @T.macro
    def project(A,W,C,by,bx):
        a = T.alloc_shared((block_m, 64), T.bfloat16)
        w = T.alloc_shared((block_n, 64), T.bfloat16)
        acc = T.alloc_fragment((block_m, block_n), T.float32)
        T.clear(acc)
        for kg in T.Pipelined(K // 64, num_stages=2):
            for i, j in T.Parallel(block_m, 64, coalesced_width=T.int32(1)):
                a[i, j] = T.if_then_else(by * block_m + i < M, A[by * block_m + i, kg * 64 + j], 0.0)
            T.copy(W[bx * block_n, kg * 64], w)
            T.gemm(a, w, acc, transpose_B=True)
        if silu:
            for i,j in T.Parallel(block_m,block_n):
                if by*block_m+i<M and bx*block_n+j<N:
                    rounded=T.cast(acc[i,j],dtype)
                    v=T.cast(T.cast(T.cast(rounded,T.float32)/streams,dtype),T.float32)
                    C[by*block_m+i,bx*block_n+j]=v/(1+T.exp(-v))
        else:T.copy(acc, C[by * block_m, bx * block_n])
    if group_m:
        mt,nt=(M+block_m-1)//block_m,(N+block_n-1)//block_n
        @T.prim_func
        def grouped(A:T.Tensor((M,K),dtype),W:T.Tensor((N,K),T.bfloat16),C:T.Tensor((M,N),dtype)):
            with T.Kernel(mt*nt,threads=threads)as pid:
                first=(pid//(group_m*nt))*group_m
                actual=T.min(mt-first,group_m)
                by=first+(pid%(group_m*nt))%actual
                bx=(pid%(group_m*nt))//actual
                project(A,W,C,by,bx)
        return grouped
    else:
        @T.prim_func
        def main(A:T.Tensor((M,K),dtype),W:T.Tensor((N,K),T.bfloat16),C:T.Tensor((M,N),dtype)):
            with T.Kernel(T.ceildiv(M,block_m),T.ceildiv(N,block_n),threads=threads)as (by,bx):
                project(A,W,C,by,bx)
        return main


@orin_jit
def hc_up_mix(M: int, H: int, rank: int, streams: int = 4):
    """BF16 up projection and branch mix, preserving every FP16 boundary."""
    assert M>=1 and H%32==0 and rank%64==0 and streams==4
    bm,bn=16,32
    @T.prim_func
    def main(A:T.Tensor((M,rank),T.float16), W:T.Tensor((streams*H,rank),T.bfloat16),
             Normed:T.Tensor((M,streams,H),T.float16), Out:T.Tensor((M,H),T.float16)):
        with T.Kernel(T.ceildiv(M,bm),T.ceildiv(H,bn),threads=128) as (by,bx):
            a=T.alloc_shared((bm,64),T.bfloat16)
            w=T.alloc_shared((streams*bn,64),T.bfloat16)
            acc=T.alloc_fragment((bm,streams*bn),T.float32)
            rounded_up=T.alloc_shared((bm,streams*bn),T.float16)
            total=T.alloc_fragment((bm,bn),T.float32)
            T.annotate_layout({total:tilelang.Fragment((bm,bn),
                forward_thread_fn=lambda i,j:(i%4)*32+j,
                forward_index_fn=lambda i,j:i//4)})
            T.clear(acc);T.clear(total)
            for kg in T.Pipelined(rank//64,num_stages=2):
                for i,j in T.Parallel(bm,64):
                    a[i,j]=0
                    if by*bm+i<M:a[i,j]=A[by*bm+i,kg*64+j]
                for n,j in T.Parallel(streams*bn,64):
                    w[n,j]=W[(n//bn)*H+bx*bn+n%bn,kg*64+j]
                T.gemm(a,w,acc,transpose_B=True)
            T.copy(acc,rounded_up)
            for branch in T.serial(streams):
                for i,j in T.Parallel(bm,bn):
                    if by*bm+i<M:
                        up=rounded_up[i,branch*bn+j]
                        gate=T.cast(1/(1+T.exp(-T.cast(up,T.float32))),T.float16)
                        product=T.cast(T.cast(gate,T.float32)*T.cast(Normed[by*bm+i,branch,bx*bn+j],T.float32),T.float16)
                        total[i,j]+=T.cast(product,T.float32)
            for i,j in T.Parallel(bm,bn):
                if by*bm+i<M:Out[by*bm+i,bx*bn+j]=total[i,j]/streams
    return main


@orin_jit
def hc_injection(M: int, N: int, K: int, dtype: str = 'float16', threads: int = 256):
    """Small-output HC projection with one CTA per token/output channel.

    Round activation operands to BF16, retain BF16 weights and reduce in FP32.
    The reduction association differs from Tensor Core projection; all stream
    rounding boundaries remain explicit. Output must be disjoint from inputs.
    """
    _dtype(dtype)
    if (any(type(x) is not int or x <= 0 for x in (M,N,K)) or N > 16
            or threads not in (128,256)):
        raise ValueError('Invalid small HC projection dimensions')
    width = (K+threads-1)//threads*threads
    @T.prim_func
    def main(A:T.Tensor((M,K),dtype),W:T.Tensor((N,K),T.bfloat16),Out:T.Tensor((M,N),dtype)):
        with T.Kernel(M,N,threads=threads) as (row,col):
            product=T.alloc_fragment((width,),T.float32)
            total=T.alloc_fragment((1,),T.float32)
            for d in T.Parallel(width,coalesced_width=T.int32(1)):
                product[d]=0
                if d<K:
                    activation=T.cast(T.cast(A[row,d],T.bfloat16),T.float32)
                    product[d]=activation*T.cast(W[col,d],T.float32)
            T.reduce_sum(product,total,dim=0)
            Out[row,col]=total[0]
    return main


@orin_jit
def hc_down_partial(M: int, N: int, K: int, splits: int = 4):
    """BF16 Tensor Core HC Down slices with FP32 partial sums.

    Partials are transient plan workspace, overwritten completely on every
    call. Do not round partials or apply SiLU before the final reduction.
    """
    if (any(type(x) is not int or x <= 0 for x in (M,N,K))
            or splits not in (2,4,8,16) or K%(64*splits)):
        raise ValueError('Invalid HC split-K dimensions')
    bm,bn=16,32
    @T.prim_func
    def main(A:T.Tensor((M,K),T.float16),W:T.Tensor((N,K),T.bfloat16),
             Partials:T.Tensor((splits,M,N),T.float32)):
        with T.Kernel(T.ceildiv(M,bm),T.ceildiv(N,bn),splits,threads=128) as (by,bx,part):
            a=T.alloc_shared((bm,64),T.bfloat16)
            w=T.alloc_shared((bn,64),T.bfloat16)
            acc=T.alloc_fragment((bm,bn),T.float32)
            T.clear(acc)
            for kg in T.Pipelined(K//(64*splits),num_stages=2):
                for i,j in T.Parallel(bm,64,coalesced_width=T.int32(1)):
                    a[i,j]=0
                    if by*bm+i<M:a[i,j]=A[by*bm+i,part*(K//splits)+kg*64+j]
                T.copy(W[bx*bn,part*(K//splits)+kg*64],w)
                T.gemm(a,w,acc,transpose_B=True)
            for i,j in T.Parallel(bm,bn):
                if by*bm+i<M and bx*bn+j<N:
                    Partials[part,by*bm+i,bx*bn+j]=acc[i,j]
    return main


@orin_jit
def hc_down_finish(M: int, N: int, splits: int = 4, streams: int = 4):
    """Reduce FP32 HC Down slices, then round/divide/SiLU in original order."""
    if (any(type(x) is not int or x <= 0 for x in (M,N,streams))
            or splits not in (2,4,8,16)):
        raise ValueError('Invalid HC split-K epilogue dimensions')
    @T.prim_func
    def main(Partials:T.Tensor((splits,M,N),T.float32),Out:T.Tensor((M,N),T.float16)):
        with T.Kernel(T.ceildiv(M*N,128),threads=128) as block:
            acc=T.alloc_fragment((128,),T.float32)
            T.clear(acc)
            for part in T.serial(splits):
                for i in T.Parallel(128):
                    index=block*128+i
                    if index<M*N:acc[i]+=Partials[part,index//N,index%N]
            for i in T.Parallel(128):
                index=block*128+i
                if index<M*N:
                    rounded=T.cast(acc[i],T.float16)
                    value=T.cast(T.cast(T.cast(rounded,T.float32)/streams,T.float16),T.float32)
                    Out[index//N,index%N]=value/(1+T.exp(-value))
    return main
