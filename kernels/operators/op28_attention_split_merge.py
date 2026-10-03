"""Stable split-context statistics merge and native attention sigmoid gate.

TileLang production API: M,L[B,24,S]FP32, O[B,24,S,256]FP32,
RawGate[B,24,256]FP16, Y[B,24,256]FP16 (or explicit fused FP32).
O is the UNNORMALIZED local weighted V. No Torch production arithmetic.
"""
import math
import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit


def validate_host_statistics(m, l, o):
    """Validate scheduler/state-owned CPU lists after statistics mutations.

    Each partition is finite m, positive finite l, finite O; or exactly
    (-inf,0,all-zero O). A zero-l/nonzero-O partition is illegal, not a
    valid partition. This offline/state boundary check is outside capture;
    the trusted producer may instead guarantee the same invariant.
    """
    if not m or len(m) != len(l) or len(m) != len(o):
        raise ValueError('empty or mismatched batch')
    splits = len(m[0][0]) if len(m[0]) == 24 else 0
    if splits <= 0:
        raise ValueError('invalid head/split dimensions')
    for mb, lb, ob in zip(m, l, o):
        if len(mb) != 24 or len(lb) != 24 or len(ob) != 24:
            raise ValueError('head count must be 24')
        for mh, lh, oh in zip(mb, lb, ob):
            if len(mh) != splits or len(lh) != splits or len(oh) != splits:
                raise ValueError('nonrectangular split dimensions')
            for maximum, total, vector in zip(mh, lh, oh):
                if len(vector) != 256 or not all(math.isfinite(v) for v in vector):
                    raise ValueError('invalid/nonfinite weighted-V')
                if not math.isfinite(total) or total < 0:
                    raise ValueError('invalid exp sum')
                if total == 0:
                    if maximum != -math.inf or any(v != 0 for v in vector):
                        raise ValueError('empty partition must be -inf,0,zero O')
                elif not math.isfinite(maximum):
                    raise ValueError('active maximum must be finite')
    return True


@orin_jit
def _compile(splits: int, gate_mode: str, output_dtype: str):
    batch = T.dynamic('batch')

    @T.prim_func
    def kernel(M: T.Tensor((batch,24,splits),T.float32),
               L: T.Tensor((batch,24,splits),T.float32),
               O: T.Tensor((batch,24,splits,256),T.float32),
               RawGate: T.Tensor((batch,24,256),T.float16),
               Y: T.Tensor((batch,24,256),output_dtype)):
        with T.Kernel(24,batch,threads=128) as (h,b):
            maxima = T.alloc_fragment((splits,),T.float32)
            weights = T.alloc_fragment((splits,),T.float32)
            sums = T.alloc_fragment((splits,),T.float32)
            maximum = T.alloc_fragment((1,),T.float32)
            denominator = T.alloc_fragment((1,),T.float32)
            products = T.alloc_fragment((splits,256),T.float32)
            acc = T.alloc_fragment((256,),T.float32)
            T.annotate_layout({
                maxima:tilelang.Fragment((splits,),forward_thread_fn=lambda s,rep:rep,replicate=128),
                weights:tilelang.Fragment((splits,),forward_thread_fn=lambda s,rep:rep,replicate=128),
                sums:tilelang.Fragment((splits,),forward_thread_fn=lambda s,rep:rep,replicate=128),
                products:tilelang.Fragment((splits,256),forward_thread_fn=lambda s,d:d%128),
                acc:tilelang.Fragment((256,),forward_thread_fn=lambda d:d%128),
                maximum:tilelang.Fragment((1,),forward_thread_fn=lambda j,rep:rep,replicate=128),
                denominator:tilelang.Fragment((1,),forward_thread_fn=lambda j,rep:rep,replicate=128)})
            for s in T.Parallel(splits):
                maxima[s] = -T.infinity(T.float32)
                if L[b,h,s] > 0:
                    maxima[s] = M[b,h,s]
            T.reduce_max(maxima,maximum,dim=0)
            for s in T.Parallel(splits):
                weights[s] = 0.0
                sums[s] = 0.0
                if L[b,h,s] > 0:
                    weights[s] = T.exp(M[b,h,s]-maximum[0])
                    sums[s] = weights[s]*L[b,h,s]
            T.reduce_sum(sums,denominator,dim=0)
            for s,d in T.Parallel(splits,256):
                products[s,d] = 0.0
                if L[b,h,s] > 0:
                    products[s,d] = weights[s]*O[b,h,s,d]
            T.reduce_sum(products,acc,dim=0)
            for d in T.Parallel(256):
                Y[b,h,d] = 0.0
                if denominator[0] > 0:
                    attention = acc[d]/denominator[0]
                    gate = 1.0/(1.0+T.exp(-T.cast(RawGate[b,h,d],T.float32)))
                    if gate_mode == 'native_fp16':
                        rounded_attention = T.cast(attention,T.float16)
                        rounded_gate = T.cast(gate,T.float16)
                        Y[b,h,d] = T.cast(rounded_attention,T.float32)*T.cast(rounded_gate,T.float32)
                    else:
                        Y[b,h,d] = attention*gate
    return kernel


def attention_split_merge(splits=4, gate_mode='native_fp16', output_dtype='float16'):
    """Build dynamic B, specialized S. Launch M,L,O,RawGate,Y,stream=... .

    Native Y=half(half(merged attention)*half(sigmoid(raw gate))).
    Explicit fp32_fused candidate permits float16 or float32 output.
    All-empty head writes exact zero. Caller guarantees validated statistics,
    finite raw gates and disjoint contiguous inputs/output. No workspace.
    """
    if type(splits) is not int or not 1 <= splits <= 16:
        raise ValueError('splits must be an integer in 1..16')
    if gate_mode not in ('native_fp16','fp32_fused'):
        raise ValueError('unsupported gate rounding')
    if output_dtype not in ('float16','float32'):
        raise ValueError('unsupported output dtype')
    if gate_mode == 'native_fp16' and output_dtype != 'float16':
        raise ValueError('native gate has FP16 output; FP32 is an explicit fused candidate')
    return _compile(splits,gate_mode,output_dtype)
