"""Operand and portable reference contracts; published tiles are autotuned."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionPolicy:
    """Selected quantized operands, independent of execution row count."""

    packed: str
    scale: str
    signs: str
    outputs: int
    inputs: int


def projection_policies(buffers):
    """Restore selected Q4 roles, including QKV's shared Z rotation.

    Keep W8 sources available for offline reference, but their presence must
    never override the selected Q4 representation in a generated body.
    """
    policies = {}
    for name, b in buffers.items():
        if not name.startswith("Q4_") or not name.endswith("_packed"):
            continue
        base = name.removesuffix("_packed")
        signs = base + "_signs"
        if base.endswith(".attn_qkv"):
            signs = base.removesuffix(".attn_qkv") + ".attn_gate_signs"
        scale = base + "_scale"
        if b["dtype"] != "u16" or len(b["shape"]) != 2:
            raise ValueError(f"Invalid Q4 matrix: {name}")
        n, packed_k = b["shape"]
        k = packed_k * 4
        for role, dtype, shape in ((scale, "f16", [n]), (signs, "i8", [k])):
            operand = buffers.get(role)
            if operand is None or operand["dtype"] != dtype or operand["shape"] != shape:
                raise ValueError(f"Invalid Q4 operand: {role}")
        policies["W_" + base.removeprefix("Q4_") + ".weight_weight"] = ProjectionPolicy(
            name, scale, signs, n, k
        )
    return policies


def expert_implementation(rows, packed_k, *, nibble_book):
    expert_tile_config(rows)
    if packed_k not in (64, 128) or type(nibble_book) is not bool:
        raise ValueError("Invalid expert representation")
    # Representation chooses the numerical codec. Legal route/GEMM geometry
    # is selected separately, with the same compressed bank for every length.
    return "exact_a8" if packed_k == 64 and nibble_book else "int8"


def expert_tile_config(rows):
    if type(rows) is not int or not 1 <= rows <= 4096:
        raise ValueError("Unsupported expert row count")
    # One legal compiler/reference configuration. Final row/expert portfolios
    # are measured together; neither model rows nor matrix width selects a tile.
    return dict(
        block_m=64,
        block_n=64,
        shortbook=True,
        num_stages=2,
        byte_permute=True,
        selectorbook=True,
        group_n=False,
    )


def expert_projection_config(rows, inputs, *, parity_xor=False):
    if type(inputs) is not int or inputs <= 0 or inputs % 128:
        raise ValueError("Unsupported expert input width")
    if type(parity_xor) is not bool:
        raise ValueError("Invalid E8P parity strategy")
    config = expert_tile_config(rows)
    if parity_xor:
        # Explicit codec diagnostic, independent of token count or matrix size.
        config.update(warp_decode=True, block_m=8, block_n=64, split_k=1, parity_xor=True)
    return config
