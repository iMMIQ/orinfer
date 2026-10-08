"""Small deterministic stateful kernel for native-package integration tests."""

import tilelang.language as T
from tools.operators.common import orin_jit


@orin_jit
def transition(prefill: bool):
    @T.prim_func
    def main(
        Input: T.Tensor((1,), T.int32),
        Weight: T.Tensor((4, 4), T.float32),
        Token: T.Tensor((1,), T.int32),
        Status: T.Tensor((1,), T.int32),
        Position: T.Tensor((1,), T.int32),
        Logits: T.Tensor((4,), T.float32),
    ):
        with T.Kernel(1, threads=32):
            selected = T.alloc_shared((1,), T.int32)
            for i in T.Parallel(1):
                selected[0] = Input[0] if prefill else Token[0]
            T.sync_threads()
            for i in T.Parallel(4):
                Logits[i] = Weight[selected[0], i]
            T.sync_threads()
            for i in T.Parallel(1):
                Token[0] = (selected[0] + 1) % 4
                Status[0] = 0
                Position[0] = Position[0] + 1

    return main
