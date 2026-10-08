"""Offline fused E8P nearest-neighbor search: one warp per vector."""

import tilelang.language as T

from tools.operators.common import orin_jit


SOURCE = r"""
__device__ unsigned int orin_e8p_nearest(const float* x, const signed char* book, int lane) {
    const int bits[8] = {0,4,1,5,2,6,3,7};
    float best = 3.402823466e38f;
    unsigned int chosen = 0;
    for (int parity=0; parity<2; ++parity) {
        float v[8];
        unsigned int neg=0;
        #pragma unroll
        for (int d=0; d<8; ++d) {
            float shifted=x[d]-(1-2*parity);
            neg |= (shifted < 0) << bits[d];
            v[d]=fabsf(shifted);
        }
        #pragma unroll
        for (int b=0; b<8; ++b) {
            int index=lane+32*b;
            unsigned int basis_sign=0;
            float distance=0, flip_cost=3.402823466e38f;
            int flip_dim=0;
            #pragma unroll
            for (int d=0; d<8; ++d) {
                int coord=book[index*8+d];
                float a=abs(coord), delta=v[d]-a;
                distance += delta*delta;
                basis_sign |= (coord < 0) << bits[d];
                float cost=4*v[d]*a;
                if (cost < flip_cost) { flip_cost=cost; flip_dim=d; }
            }
            bool correct=(__popc(neg ^ basis_sign)&1) != 0;
            if (correct) distance += flip_cost;
            unsigned int low=neg ^ basis_sign;
            if (correct) low ^= 1u << bits[flip_dim];
            unsigned int code=(index << 8) | (low ^ parity);
            if (distance < best || (distance == best && code < chosen)) {
                best=distance; chosen=code;
            }
        }
    }
    #pragma unroll
    for (int offset=16; offset>0; offset/=2) {
        float other=__shfl_down_sync(0xffffffffu,best,offset);
        unsigned int code=__shfl_down_sync(0xffffffffu,chosen,offset);
        if (lane+offset<32 && (other<best || (other==best && code<chosen))) {
            best=other; chosen=code;
        }
    }
    return chosen;
}
"""


@orin_jit
def encode(vectors):
    if type(vectors) is not int or vectors <= 0:
        raise ValueError("Invalid vector count")

    @T.prim_func
    def main(
        X: T.Tensor((vectors, 8), T.float32),
        Book: T.Tensor((256, 8), T.int8),
        Codes: T.Tensor((vectors,), T.uint16),
    ):
        with T.Kernel(T.ceildiv(vectors, 4), threads=128) as block:
            T.import_source(SOURCE)
            tx = T.get_thread_binding()
            row = block * 4 + tx // 32
            if row < vectors:
                code = T.call_extern(
                    "uint32", "orin_e8p_nearest", T.address_of(X[row, 0]), Book.data, tx % 32
                )
                if tx % 32 == 0:
                    Codes[row] = T.cast(code, T.uint16)

    return main
