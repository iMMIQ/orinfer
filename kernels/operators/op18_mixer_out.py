"""GDN/full-attention W4A16 output projection 6144->5120 on SM87.

Explicit-output TileLang API; caller owns workspace/stable addresses and stream.
Weights: adjacent low/high U4 P, FP16 S, numeric int8 Z, group128.
W=half((q-z)*s), FP32 accumulation, FP16 Y. Residual is never added here.
The implementation reuses validated dimension-generic op03/op05 kernels.
"""

from dataclasses import dataclass

from kernels.operators.op05_ffn_down import ffn_down_partial, ffn_down_merge
from kernels.operators.op03_ffn_gate_up import ffn_gate_up

K_MIXER = 6144
N_HIDDEN = 5120


def mixer_out_partial(M, SPLIT=8, implementation="register"):
    """Build (X,P,S,Z,partial[SPLIT,M,5120]F32), no merge or residual."""
    return ffn_down_partial(M, N=N_HIDDEN, K=K_MIXER, SPLIT=SPLIT, implementation=implementation)


def mixer_out_merge(M, SPLIT=8):
    """Build (partialF32,YF16) using op32's ordered FP32 final reduction."""
    return ffn_down_merge(M, N=N_HIDDEN, SPLIT=SPLIT)


def mixer_out_full(M, implementation="register"):
    """Build (X,P,S,Z,Y), full-K FP32 MMA, no global workspace."""
    if isinstance(M, int) and M <= 0:
        raise ValueError("M must be positive")
    return ffn_gate_up(M, N=N_HIDDEN, K=K_MIXER, implementation=implementation, BM=64)


@dataclass(frozen=True)
class MixerOut:
    route: str
    projection: object
    merge: object = None
    splits: int = 0

    def workspace_shape(self, M):
        if M <= 0:
            raise ValueError("M must be positive")
        return (self.splits, M, N_HIDDEN) if self.merge is not None else None

    def __call__(self, X, P, S, Z, Y, partial=None, *, stream=None):
        if self.merge is None:
            if partial is not None:
                raise ValueError("full route takes no partial workspace")
            self.projection(X, P, S, Z, Y, stream=stream)
        else:
            if partial is None:
                raise ValueError("split-K requires caller-owned FP32 partial")
            self.projection(X, P, S, Z, partial, stream=stream)
            self.merge(partial, Y, stream=stream)


def build_mixer_out(M, route="splitk_register", SPLIT=8):
    if route not in ("splitk_shared", "splitk_register", "full_shared", "full_register"):
        raise ValueError("unknown mixer_out route")
    implementation = route.rsplit("_", 1)[1]
    if route.startswith("splitk"):
        return MixerOut(
            route, mixer_out_partial(M, SPLIT, implementation), mixer_out_merge(M, SPLIT), SPLIT
        )
    return MixerOut(route, mixer_out_full(M, implementation))
