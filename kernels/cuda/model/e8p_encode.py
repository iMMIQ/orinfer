"""Offline fused E8P nearest-neighbor search: one warp per vector."""

import tilelang.language as T

from orinfer_build.compiler.cuda.common import orin_jit


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


# Keep only groups whose nearest member can reach the global eighth neighbor.
# Each group fixes a basis and parity, leaving 128 sign patterns. This retains
# the full codebook, including groups tied at the boundary; it is not ANN.
TOP8_SOURCE = r"""
__device__ __forceinline__ double e8p_score64(
    const float* x, const signed char* book, unsigned code) {
    const int bits[8] = {0,4,1,5,2,6,3,7};
    unsigned low=code & 255u, parity=__popc(low)&1u;
    double score=0;
    #pragma unroll
    for (int d=0; d<8; ++d) {
        int c=book[(code>>8)*8+d]*(1-2*((int)((low^parity)>>bits[d])&1))
              +1-2*(int)parity;
        score += double(c*c)-2.0*double(x[d])*double(c);
    }
    return score;
}

__device__ __forceinline__ bool e8p_before(
    float a, unsigned ca, float b, unsigned cb, float margin,
    const float* x, const signed char* book) {
    if (ca==cb) return false;
    if (cb==0xffffffffu) return ca!=0xffffffffu;
    if (ca==0xffffffffu) return false;
    if (fabsf(a-b)>margin) return a<b;
    // FP32 is sufficient away from a boundary. Re-evaluate close comparisons
    // in FP64 and use code order for ties, including zero/symmetric vectors.
    double da=e8p_score64(x,book,ca), db=e8p_score64(x,book,cb);
    return da<db || (da==db && ca<cb);
}

__device__ void orin_e8p_top8(
    const float* x, const signed char* book, void* output, int lane) {
    const int bits[8] = {0,4,1,5,2,6,3,7};
    float bounds[16], sum_abs=0;
    #pragma unroll
    for (int d=0; d<8; ++d) sum_abs+=fabsf(x[d]);
    // With sum(abs(x)) <= 2^-64, every term c*c-2*x*c rounds
    // to c*c in e8p_score64 (odd integer |c| <= 11). The existing
    // FP64 comparator therefore orders all codes by norm then ID.
    // Avoid expanding hundreds of tied groups for underflow-sized rows.
    if (sum_abs<=0x1p-64f) {
        const unsigned short zero_codes[8] = {ORINFER_ZERO_TOP8};
        if (lane==0) {
            #pragma unroll
            for (int i=0; i<8; ++i) ((unsigned short*)output)[i]=zero_codes[i];
        }
        return;
    }
    // Conservative FP32 roundoff allowance for group bounds and code scores.
    // Reduced distances omit the common ||x||^2 to avoid cancellation there.
    float margin=64.0f*1.1920928955078125e-7f*(1024.0f+22.0f*sum_abs);
    #pragma unroll
    for (int slot=0; slot<16; ++slot) {
        int group=lane+32*slot, index=group&255, p=group>>8;
        float score=0, flip=3.402823466e38f;
        unsigned negative=0, basis_sign=0;
        #pragma unroll
        for (int d=0; d<8; ++d) {
            int c=book[index*8+d], a=abs(c), delta=1-2*p;
            float y=x[d]-float(delta), v=fabsf(y);
            negative|=(y<0)<<d;
            basis_sign|=(c<0)<<d;
            score += float(a*a+1)-2.0f*float(delta)*x[d]-2.0f*float(a)*v;
            flip=fminf(flip,4.0f*v*float(a));
        }
        if (__popc(negative^basis_sign)&1) score+=flip;
        bounds[slot]=isfinite(score) ? score : -__int_as_float(0x7f800000);
    }
    // Eight distinct group minima give a valid upper bound on the eighth code.
    unsigned removed=0;
    float cutoff=__int_as_float(0x7f800000);
    #pragma unroll
    for (int rank=0; rank<8; ++rank) {
        float best=__int_as_float(0x7f800000);
        int which=-1;
        #pragma unroll
        for (int slot=0; slot<16; ++slot) {
            if (!(removed&(1u<<slot)) && (which<0 || bounds[slot]<best)) {
                best=bounds[slot]; which=slot;
            }
        }
        float value=best;
        #pragma unroll
        for (int offset=16; offset>0; offset/=2)
            value=fminf(value,__shfl_xor_sync(0xffffffffu,value,offset));
        unsigned owners=__ballot_sync(0xffffffffu,best==value);
        int owner=__ffs(owners)-1;
        if (lane==owner) removed|=1u<<which;
        cutoff=value;
    }
    cutoff=isfinite(cutoff) ? cutoff+2.0f*margin : __int_as_float(0x7f800000);
    float scores[8];
    unsigned codes[8];
    #pragma unroll
    for (int i=0; i<8; ++i) { scores[i]=__int_as_float(0x7f800000); codes[i]=0xffffffffu; }
    #pragma unroll
    for (int slot=0; slot<16; ++slot) {
        unsigned active=__ballot_sync(0xffffffffu,bounds[slot]<=cutoff);
        while (active) {
            int owner=__ffs(active)-1;
            active&=active-1;
            int group=owner+32*slot, p=group>>8, index=group&255;
            #pragma unroll
            for (int part=0; part<4; ++part) {
                unsigned low=lane+32*part;
                low|=((__popc(low)^p)&1u)<<7;
                unsigned code=(index<<8)|low;
                float score=0;
                #pragma unroll
                for (int d=0; d<8; ++d) {
                    int c=book[index*8+d]*(1-2*((int)((low^p)>>bits[d])&1))+1-2*p;
                    score += float(c*c)-2.0f*x[d]*float(c);
                }
                if (!isfinite(score) || score<=cutoff) {
                    if (e8p_before(score,code,scores[7],codes[7],margin,x,book)) {
                        #pragma unroll
                        for (int i=0; i<8; ++i) {
                            if (e8p_before(score,code,scores[i],codes[i],margin,x,book)) {
                                float old_score=scores[i]; unsigned old_code=codes[i];
                                scores[i]=score; codes[i]=code;
                                score=old_score; code=old_code;
                            }
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int rank=0; rank<8; ++rank) {
        float score=scores[0]; unsigned code=codes[0];
        #pragma unroll
        for (int offset=16; offset>0; offset/=2) {
            float other=__shfl_xor_sync(0xffffffffu,score,offset);
            unsigned candidate=__shfl_xor_sync(0xffffffffu,code,offset);
            if (e8p_before(other,candidate,score,code,margin,x,book)) {
                score=other; code=candidate;
            }
        }
        if (lane==0) ((unsigned short*)output)[rank]=(unsigned short)code;
        if (codes[0]==code) {
            #pragma unroll
            for (int i=0; i<7; ++i) { scores[i]=scores[i+1]; codes[i]=codes[i+1]; }
            scores[7]=__int_as_float(0x7f800000); codes[7]=0xffffffffu;
        }
    }
}
"""


@orin_jit
def encode_top8(vectors):
    if type(vectors) is not int or vectors <= 0:
        raise ValueError("Invalid vector count")
    # Derive the constants from the same canonical basis as Encoder.book.
    # Code ordering is part of the representation, never a handwritten list.
    import numpy as np
    from orinfer_build.quantization.e8p import basis
    from orinfer_build.quantization.vq import e8p_decode

    grid = e8p_decode(np.arange(65536, dtype=np.uint16), basis()).astype(np.int32)
    zero_codes = np.argsort((grid * grid).sum(1), kind="stable")[:8]
    source = TOP8_SOURCE.replace("ORINFER_ZERO_TOP8", ",".join(map(str, zero_codes)))

    @T.prim_func
    def main(
        X: T.Tensor((vectors, 8), T.float32),
        Book: T.Tensor((256, 8), T.int8),
        Codes: T.Tensor((vectors, 8), T.uint16),
    ):
        with T.Kernel(T.ceildiv(vectors, 4), threads=128) as block:
            T.import_source(source)
            tx = T.get_thread_binding()
            row = block * 4 + tx // 32
            if row < vectors:
                T.call_extern(
                    "void",
                    "orin_e8p_top8",
                    T.address_of(X[row, 0]),
                    Book.data,
                    T.access_ptr(Codes[row, 0], "w", 8),
                    tx % 32,
                )

    return main
