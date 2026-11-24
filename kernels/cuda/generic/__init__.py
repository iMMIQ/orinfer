"""Portable, architecture-agnostic TileLang kernels for non-Orin discrete CUDA GPUs.

All operators in this package avoid SM87-specific inline PTX (`mma.s4`, manual
`ldmatrix`, explicit shared-memory bank swizzles) and rely solely on portable
high-level TileLang primitives (`T.copy`, `T.gemm`, `T.Pipelined`, `T.Parallel`,
`T.reduce_max`) so TileLang lowers natively across Ampere (SM80/86), Ada (SM89),
Hopper (SM90), and Blackwell (SM100/120) while preserving exact numerical
compatibility with Orinfer's `orinfer_e8p_int8` weight representation.
"""

from kernels.cuda.generic.e8p_gemm import e8p_dense_gemm, e8p_grouped_gemm
from kernels.cuda.generic.gdn_recurrent import gdn_recurrent
from kernels.cuda.generic.rotation_a8 import plain_activation_a8, rotate_activation_a8
from kernels.cuda.generic.w8a8_gemm import w8a8_gemm

__all__ = [
    "e8p_dense_gemm",
    "e8p_grouped_gemm",
    "gdn_recurrent",
    "plain_activation_a8",
    "rotate_activation_a8",
    "w8a8_gemm",
]
