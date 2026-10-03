"""Experimental W4A8: four-entry LUT plus integer coarse step.

I8-fragment W4 with four biased codebook bytes and one coarse step. A slots have
explicit wait/barrier ownership. No production registration.
"""
import tilelang.language as T
from kernels.operators.op03_ffn_gate_up import _orin_jit
from kernels.model.w4a8_lut4_helpers import LUT4_SOURCE

EPILOGUE_SOURCE = r'''
__device__ __forceinline__ int orin_stage_half(unsigned int* p, int index, float value) {
    reinterpret_cast<__half*>(p)[index] = __float2half_rn(value);
    return 0;
}
__device__ __forceinline__ unsigned int orin_read_staged_half(const unsigned int* p, int index) {
    return reinterpret_cast<const unsigned short*>(p)[index];
}
'''


@_orin_jit
def w4a8_lut4(M, N: int, K: int, BM=256, BN=128, stages=2, fixedpoint=False,
                         ldmatrix=True, clamped=True, packed_metadata=True, coalesced_epilogue=False,
                         group_major_metadata=False, vector_major_packed=False):
    assert BM in (64, 128, 256) and BN in (64, 128)
    assert N % BN == 0 and K % 128 == 0 and stages in (1, 2)
    assert packed_metadata and clamped and ldmatrix and K%256==0
    assert not coalesced_epilogue or stages==2
    threads = BN // 16 * 32
    parts_m = BM // 16
    source = LUT4_SOURCE + (EPILOGUE_SOURCE if coalesced_epilogue else '')
    metadata_shape = (K//128,N) if group_major_metadata else (N,K//128)
    packed_shape = (N//64,K//128,2,128,4) if vector_major_packed else (N//64,K//128,128,8)
    @T.macro
    def copy_a(A, shared, stage, kg, by, tx):
        for vector in T.unroll(BM*8//threads):
            index = vector*threads+tx
            row, col = index//8, (index%8)*4
            physical_col = col ^ ((row%8)*4)
            T.ptx_cp_async(
                T.tvm_access_ptr(T.type_annotation(T.uint32), shared.data,
                    stage*BM*32+row*32+physical_col, 4, 2),
                T.tvm_access_ptr(T.type_annotation(T.uint32), A.data,
                    (by*BM+row)*(K//4)+kg*32+col, 4, 1),
                4, predicate=by*BM+row<M)
        T.ptx_commit_group()

    @T.macro
    def copy_b(PP, Z, Coef, packed, coefficient, zero, kg, slot, bx, warp, lane, group):
        for vector in T.unroll(2):
            for j in T.vectorized(4):
                if vector_major_packed:
                    packed[slot,vector*4+j]=PP[bx*(BN//64)+warp//4,kg,vector,(warp%4)*32+lane,j]
                else:
                    packed[slot,vector*4+j]=PP[bx*(BN//64)+warp//4,kg,(warp%4)*32+lane,vector*4+j]
        for ni in T.unroll(2):
            if group_major_metadata:
                coefficient[slot,ni]=Coef[kg,bx*BN+warp*16+ni*8+group]
                zero[slot,ni]=Z[kg,bx*BN+warp*16+ni*8+group]
            else:
                coefficient[slot,ni]=Coef[bx*BN+warp*16+ni*8+group,kg]
                zero[slot,ni]=Z[bx*BN+warp*16+ni*8+group,kg]

    @T.prim_func
    def lut4_projection(A: T.Tensor((M, K//4), T.uint32),
               PP: T.Tensor(packed_shape, T.uint32),
               S: T.Tensor((N, K//128), T.float16),
               Z: T.Tensor(metadata_shape, T.int8),
               Coef: T.Tensor(metadata_shape, T.uint32),
               AS: T.Tensor((M,), T.float16), WS: T.Tensor((N,), T.float16),
               C: T.Tensor((M, N), T.float16)):
        with T.Kernel(N//BN, T.ceildiv(M, BM), threads=threads) as (bx, by):
            T.annotate_min_blocks_per_sm(1)
            T.import_source(source)
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            group = lane // 4
            tid = lane % 4
            # Raw cp.async pointers and MMA reads use the same physical XOR.
            # It preserves contiguous four-word vectors on each row.
            a = T.alloc_shared((stages, BM, 32), T.uint32)
            packed = T.alloc_local((2,8), T.uint32)
            scale = T.alloc_local((2,), T.float16)
            zero = T.alloc_local((2,2), T.int8)
            coefficient = T.alloc_local((2,2), T.uint32)
            inverse = T.alloc_local((2,), T.float32)
            ar = T.alloc_local((4,), T.uint32)
            br = T.alloc_local((4,), T.uint32)
            accum = T.alloc_local((parts_m*8,), T.int32)
            for j in T.unroll(parts_m*8):
                accum[j] = 0
            for ni in T.unroll(2):
                if clamped:
                    inverse[ni]=1.0
                else:
                    inverse[ni] = T.call_extern('float32', '__fdiv_rn', 1.0,
                        T.cast(WS[bx*BN+warp*16+ni*8+group], T.float32))
            if stages==2:
                copy_a(A, a, 0, 0, by, tx)
            copy_b(PP,Z,Coef,packed,coefficient,zero,0,0,bx,warp,lane,group)
            for kg_pair in T.serial(K//256):
                for slot in T.unroll(2):
                    kg=kg_pair*2+slot
                    if stages==2:
                        if kg+1<K//128:
                            copy_a(A,a,(kg+1)%2,kg+1,by,tx)
                    else:
                        copy_a(A,a,0,kg,by,tx)
                    if kg+1<K//128:
                        copy_b(PP,Z,Coef,packed,coefficient,zero,kg+1,1-slot,bx,warp,lane,group)
                    for ni in T.unroll(2):
                        scale[ni]=1.0
                    if stages==2 and kg+1<K//128:
                        T.ptx_wait_group(1)
                    else:
                        T.ptx_wait_group(0)
                    T.sync_threads()
                    for ki in T.unroll(4):
                        for ni in T.unroll(2):
                            for half in T.unroll(2):
                                quartet = (packed[slot,ki*2+ni]>>(half*16))&65535
                                br[ni*2+half] = T.call_pure_extern('uint32','orin_w4_i8_lut4',
                                    T.cast(quartet,T.uint16),coefficient[slot,ni],T.cast(zero[slot,ni],T.uint8))
                        for mi in T.unroll(parts_m):
                            if ldmatrix:
                                load_row=mi*16+lane%8+((lane//8)%2)*8
                                load_col=(ki*8+(lane//16)*4)^((load_row%8)*4)
                                T.ptx_ldmatrix(False, 4,
                                    T.access_ptr(a[kg%stages,load_row,load_col],'r',extent=4),
                                    T.access_ptr(ar[0],'w',extent=4))
                            else:
                                for ai in T.unroll(4):
                                    ar[ai] = a[kg%stages, mi*16+group+(ai%2)*8,
                                        (ki*8+tid+(ai//2)*4) ^ ((group%8)*4)]
                            for ni in T.unroll(2):
                                T.ptx_mma('int32', 'm16n8k32', 'row', 'col', 'int8', 'int8', 'int32',
                                    ar.data, 0, br.data, ni*2, accum.data, mi*8+ni*4, False)
                    # All warps finish current reads before the slot is reused.
                    T.sync_threads()
            for mi in T.unroll(parts_m):
                for ni in T.unroll(2):
                    for ci in T.unroll(4):
                        row = by*BM+mi*16+group+(ci//2)*8
                        col = bx*BN+warp*16+ni*8+tid*2+ci%2
                        if row < M:
                            value = (T.cast(accum[mi*8+ni*4+ci], T.float32)
                                * T.cast(AS[row], T.float32) * T.cast(WS[col], T.float32))
                            if coalesced_epilogue:
                                # The last K iteration drains cp.async and ends
                                # with a full barrier. Reuse its dead A slots;
                                # capacity 2*BM*128 bytes >= BM*BN*2 bytes.
                                local_row=mi*16+group+(ci//2)*8
                                local_col=warp*16+ni*8+tid*2+ci%2
                                physical=local_row*BN+(local_col^((local_row%8)*8))
                                T.evaluate(T.call_extern('int32','orin_stage_half',
                                    T.access_ptr(a[0,0,0],'w',extent=stages*BM*32),physical,value))
                            else:
                                C[row, col] = value
            if coalesced_epilogue:
                T.sync_threads()
                for i,j in T.Parallel(BM,BN):
                    if by*BM+i<M:
                        physical=i*BN+(j^((i%8)*8))
                        bits=T.call_pure_extern('uint32','orin_read_staged_half',
                            T.access_ptr(a[0,0,0],'r',extent=stages*BM*32),physical)
                        C[by*BM+i,bx*BN+j]=T.reinterpret(T.float16,T.cast(bits,T.uint16))
    return lut4_projection
