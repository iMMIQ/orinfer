"""Finite-logit greedy selection, matching the lowest-index argmax tie rule."""
import tilelang
import tilelang.language as T

from tools.operators.common import orin_jit


@orin_jit
def greedy_partials(vocab: int):
    if type(vocab) is not int or vocab < 1:
        raise ValueError('Positive vocabulary required')
    blocks = (vocab + 1023) // 1024
    @T.prim_func
    def main(Logits: T.Tensor((1, vocab), T.float32),
             Values: T.Tensor((blocks,), T.float32),
             Indices: T.Tensor((blocks,), T.int32),
             Invalid: T.Tensor((blocks,), T.int32)):
        with T.Kernel(blocks, threads=256) as block:
            values = T.alloc_fragment((1024,), T.float32)
            indices = T.alloc_fragment((1024,), T.int32)
            invalid = T.alloc_fragment((1024,), T.int32)
            maximum = T.alloc_fragment((1,), T.float32)
            minimum = T.alloc_fragment((1,), T.int32)
            bad = T.alloc_fragment((1,), T.int32)
            T.annotate_layout({
                values:tilelang.Fragment((1024,),forward_thread_fn=lambda i:i%256,forward_index_fn=lambda i:i//256),
                indices:tilelang.Fragment((1024,),forward_thread_fn=lambda i:i%256,forward_index_fn=lambda i:i//256),
                invalid:tilelang.Fragment((1024,),forward_thread_fn=lambda i:i%256,forward_index_fn=lambda i:i//256),
                maximum:tilelang.Fragment((1,),forward_thread_fn=lambda i,rep:rep,replicate=256),
                minimum:tilelang.Fragment((1,),forward_thread_fn=lambda i,rep:rep,replicate=256),
                bad:tilelang.Fragment((1,),forward_thread_fn=lambda i,rep:rep,replicate=256)})
            for i in T.Parallel(1024):
                values[i] = -T.infinity(T.float32)
                invalid[i] = 0
                if block * 1024 + i < vocab:
                    values[i] = Logits[0, block * 1024 + i]
                    invalid[i] = T.cast(not T.call_pure_extern('bool', 'isfinite', values[i]), T.int32)
            T.reduce_max(values, maximum, dim=0)
            T.reduce_max(invalid, bad, dim=0)
            for i in T.Parallel(1024):
                indices[i] = 2147483647
                if block * 1024 + i < vocab and values[i] == maximum[0]:
                    indices[i] = block * 1024 + i
            T.reduce_min(indices, minimum, dim=0)
            if T.get_thread_binding() == 0:
                Values[block] = maximum[0]
                Indices[block] = minimum[0]
                Invalid[block] = bad[0]
    return main


@orin_jit
def greedy_merge(vocab: int):
    if type(vocab) is not int or vocab < 1:
        raise ValueError('Positive vocabulary required')
    blocks = (vocab + 1023) // 1024
    width = max(32,1 << (blocks - 1).bit_length())
    threads=min(256,width)
    @T.prim_func
    def main(Values: T.Tensor((blocks,), T.float32),
             Indices: T.Tensor((blocks,), T.int32),
             Invalid: T.Tensor((blocks,), T.int32),
             Output: T.Tensor((2,), T.int32)):
        with T.Kernel(1, threads=threads):
            values = T.alloc_fragment((width,), T.float32)
            indices = T.alloc_fragment((width,), T.int32)
            invalid = T.alloc_fragment((width,), T.int32)
            maximum = T.alloc_fragment((1,), T.float32)
            minimum = T.alloc_fragment((1,), T.int32)
            bad = T.alloc_fragment((1,), T.int32)
            T.annotate_layout({
                values:tilelang.Fragment((width,),forward_thread_fn=lambda i:i%threads,forward_index_fn=lambda i:i//threads),
                indices:tilelang.Fragment((width,),forward_thread_fn=lambda i:i%threads,forward_index_fn=lambda i:i//threads),
                invalid:tilelang.Fragment((width,),forward_thread_fn=lambda i:i%threads,forward_index_fn=lambda i:i//threads),
                maximum:tilelang.Fragment((1,),forward_thread_fn=lambda i,rep:rep,replicate=threads),
                minimum:tilelang.Fragment((1,),forward_thread_fn=lambda i,rep:rep,replicate=threads),
                bad:tilelang.Fragment((1,),forward_thread_fn=lambda i,rep:rep,replicate=threads)})
            for i in T.Parallel(width):
                values[i] = -T.infinity(T.float32)
                invalid[i] = 0
                if i < blocks:
                    values[i] = Values[i]
                    invalid[i] = Invalid[i]
            T.reduce_max(values, maximum, dim=0)
            T.reduce_max(invalid, bad, dim=0)
            for i in T.Parallel(width):
                indices[i] = 2147483647
                if i < blocks and values[i] == maximum[0]:
                    indices[i] = Indices[i]
            T.reduce_min(indices, minimum, dim=0)
            if T.get_thread_binding() == 0:
                Output[0] = minimum[0]
                Output[1] = bad[0]
    return main
