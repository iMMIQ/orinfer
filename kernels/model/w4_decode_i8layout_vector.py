"""M1 W4A16: direct vector loads from the I8-fragment U4 layout.

Only lossless storage permutation. No shared B, no full dequantized weights.
Physical words [N/64,K/128,128 lanes,8 words] hold one eight-nibble bundle
per m16n8k16 K-step. Each bundle supplies two adjacent N8 fragments.
Fragment convention: NVIDIA PTX m16n8k16 floating-point matrix fragments.
"""
import tilelang.language as T
from kernels.operators.op03_ffn_gate_up import _orin_jit, _PAIR_SOURCE


@_orin_jit
def w4_decode_i8layout_vector(N: int, K: int, SPLIT=1, output_dtype='float16', TILE_N=64,
                           output_layout='flat', byte_permute=False, vector_words=2):
    assert TILE_N in (64, 128, 256)
    assert N % TILE_N == 0 and K % (128*SPLIT) == 0
    assert output_dtype in ('float16', 'float32')
    assert output_layout in ('flat','qkvz')
    assert vector_words in (2,4)
    if output_layout=='qkvz':
        assert N==16384 and K==5120 and SPLIT==1 and output_dtype=='float16'
    blocks = TILE_N//64

    @T.macro
    def multiply(A, PP, S, Z, bx, sk, acc):
        T.import_source(_PAIR_SOURCE)
        tx = T.get_thread_binding()
        lane = tx % 32
        warp = tx // 32
        group = lane // 4
        tid = lane % 4
        packed = T.alloc_local((blocks, 8), T.uint32)
        native = T.alloc_local((2, 8 if vector_words==4 else 2), T.uint32)
        pair = T.alloc_local((2,), T.uint32)
        scales = T.alloc_local((blocks*2,), T.float16)
        zeros = T.alloc_local((blocks*2,), T.int8)
        ar = T.alloc_local((4,), T.uint32)
        br = T.alloc_local((2,), T.uint32)
        for i in T.unroll(blocks*8):
            acc[i] = 0.0
        for i in T.unroll(4):
            ar[i] = 0
        for kg in T.serial(K//128//SPLIT):
            gk = sk * (K//128//SPLIT) + kg
            for block in T.unroll(blocks):
                if vector_words==4:
                    for vector in T.unroll(2):
                        for ni in T.vectorized(4):
                            native[0,vector*4+ni] = PP[bx*blocks+block,gk,(tx//4)*4+tid//2,vector*4+ni]
                            native[1,vector*4+ni] = PP[bx*blocks+block,gk,(tx//4)*4+tid//2+2,vector*4+ni]
                for ki32 in T.unroll(4):
                    if vector_words==2:
                        for ni in T.vectorized(2):
                            native[0, ni] = PP[bx*blocks+block, gk, (tx//4)*4+tid//2, ki32*2+ni]
                            native[1, ni] = PP[bx*blocks+block, gk, (tx//4)*4+tid//2+2, ki32*2+ni]
                    if byte_permute:
                        for ni in T.unroll(2):
                            pair[ni] = T.call_pure_extern('uint32','__byte_perm',
                                native[0,ki32*2+ni if vector_words==4 else ni],
                                native[1,ki32*2+ni if vector_words==4 else ni],T.cast(0x6240+(tid%2)*0x1111,T.uint32))
                        packed[block,ki32*2] = T.call_pure_extern('uint32','__byte_perm',pair[0],pair[1],T.cast(0x5410,T.uint32))
                        packed[block,ki32*2+1] = T.call_pure_extern('uint32','__byte_perm',pair[0],pair[1],T.cast(0x7632,T.uint32))
                    else:
                        for half in T.unroll(2):
                            shift = half*16+(tid%2)*8
                            packed[block, ki32*2+half] = 0
                            for ni in T.unroll(2):
                                pair0 = (native[0,ki32*2+ni if vector_words==4 else ni]>>shift)&255
                                pair1 = (native[1,ki32*2+ni if vector_words==4 else ni]>>shift)&255
                                packed[block, ki32*2+half] = packed[block, ki32*2+half] | ((pair0 | (pair1<<8))<<(ni*16))
            for part in T.unroll(blocks*2):
                scales[part] = S[bx*TILE_N+warp*16+(part//2)*64+(part%2)*8+group, gk]
                zeros[part] = Z[bx*TILE_N+warp*16+(part//2)*64+(part%2)*8+group, gk]
            for ki in T.unroll(8):
                if group == 0:
                    col = gk*128+ki*16+tid*2
                    ar[0] = (T.cast(T.reinterpret(T.uint16, A[0, col]), T.uint32)
                             | (T.cast(T.reinterpret(T.uint16, A[0, col+1]), T.uint32) << 16))
                    ar[2] = (T.cast(T.reinterpret(T.uint16, A[0, col+8]), T.uint32)
                             | (T.cast(T.reinterpret(T.uint16, A[0, col+9]), T.uint32) << 16))
                for part in T.unroll(blocks*2):
                    br[0] = T.call_pure_extern('uint32', 'op03_deq_pair',
                        T.cast((packed[part//2, ki] >> ((part%2)*16)) & 255, T.uint8), scales[part], zeros[part])
                    br[1] = T.call_pure_extern('uint32', 'op03_deq_pair',
                        T.cast((packed[part//2, ki] >> ((part%2)*16+8)) & 255, T.uint8), scales[part], zeros[part])
                    T.ptx_mma('float32', 'm16n8k16', 'row', 'col', 'fp16', 'fp16', 'fp32',
                              ar.data, 0, br.data, 0, acc.data, part*4, False)

    if output_layout=='qkvz':
        @T.prim_func
        def decode_i8_vector(A: T.Tensor((1, K), T.float16),
                 PP: T.Tensor((N//64, K//128, 128, 8), T.uint32),
                 S: T.Tensor((N, K//128), T.float16),
                 Z: T.Tensor((N, K//128), T.int8),
                 QKV: T.Tensor((1,10240), T.float16),
                 ZOUT: T.Tensor((1,6144), T.float16)):
            with T.Kernel(N//TILE_N, threads=128) as bx:
                acc=T.alloc_local((blocks*8,),T.float32)
                multiply(A,PP,S,Z,bx,0,acc)
                tx=T.get_thread_binding()
                if (tx%32)//4==0:
                    for part in T.unroll(blocks*2):
                        for j in T.unroll(2):
                            index=bx*TILE_N+(tx//32)*16+(part//2)*64+(part%2)*8+(tx%4)*2+j
                            if index<10240:
                                QKV[0,index]=acc[part*4+j]
                            else:
                                ZOUT[0,index-10240]=acc[part*4+j]
    else:
        @T.prim_func
        def decode_i8_vector(A: T.Tensor((1, K), T.float16),
                 PP: T.Tensor((N//64, K//128, 128, 8), T.uint32),
                 S: T.Tensor((N, K//128), T.float16),
                 Z: T.Tensor((N, K//128), T.int8),
                 O: T.Tensor((SPLIT, N), output_dtype)):
            with T.Kernel(N//TILE_N, SPLIT, threads=128) as (bx, sk):
                acc=T.alloc_local((blocks*8,),T.float32)
                multiply(A,PP,S,Z,bx,sk,acc)
                tx=T.get_thread_binding()
                if (tx%32)//4==0:
                    for part in T.unroll(blocks*2):
                        for j in T.unroll(2):
                            O[sk,bx*TILE_N+(tx//32)*16+(part//2)*64+(part%2)*8+(tx%4)*2+j]=acc[part*4+j]
    return decode_i8_vector
