"""Integer E8P representation and numerical reference fixtures.

One U16 index selects eight lattice coordinates, represented exactly on an integer grid. Rotation
is an input-side normalized block128 Hadamard with fixed signs, not a transform
that can be moved through SwiGLU. Codebook data is supplied by the caller.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


SIGN_BITS = np.array([0, 4, 1, 5, 2, 6, 3, 7], np.uint16)


def rotate(x, signs):
    x = np.asarray(x, dtype=np.float32)
    signs = np.asarray(signs)
    if x.shape[-1] % 128 or signs.shape != (x.shape[-1],) or not np.isin(signs, [-1, 1]).all():
        raise ValueError("Block128 rotation needs one +/-1 sign per input channel")
    y = (x * signs).copy()
    for stage in range(7):
        h = 1 << stage
        block = y.reshape(*y.shape[:-1], -1, 2 * h)
        left, right = block[..., :h].copy(), block[..., h:].copy()
        block[..., :h], block[..., h:] = left + right, left - right
    return y * np.float32(128**-0.5)


def e8p_decode(codes, table):
    c = np.asarray(codes, dtype=np.uint16)
    signs = c & 255
    parity = np.zeros(c.shape, np.uint16)
    for i in range(8):
        parity ^= (signs >> i) & 1
    adjusted = signs ^ parity
    negative = ((adjusted[..., None] >> SIGN_BITS) & 1).astype(np.int16)
    shift = (1 - 2 * parity.astype(np.int16))[..., None]
    return (table[c >> 8].astype(np.int16) * (1 - 2 * negative) + shift).astype(np.int8)


def e8p_sign_table(table):
    """8 KiB runtime table of +/- basis coordinates with both parity shifts.

    This is an exact decoder table, not a weight representation. Columns are
    the positive/negative uint32 words for the low/high halves of a vector.
    """
    if (
        table.dtype != np.int8
        or table.shape != (256, 8)
        or (table.astype(np.int16) % 2).any()
        or np.abs(table.astype(np.int16)).max() > 126
    ):
        raise ValueError("Expected an even signed INT8 E8P basis")
    values = np.empty((512, 8, 2), np.int8)
    for parity in (0, 1):
        shift = 1 - 2 * parity
        values[parity * 256 : (parity + 1) * 256, :, 0] = table.astype(np.int16) + shift
        values[parity * 256 : (parity + 1) * 256, :, 1] = -table.astype(np.int16) + shift
    return np.stack(
        [
            np.ascontiguousarray(values[:, half * 4 : half * 4 + 4, sign]).view("<u4").reshape(512)
            for half in range(2)
            for sign in range(2)
        ],
        axis=-1,
    )


def e8p_selector_table(table):
    """Encode the spread29 integer basis as exact byte-permute selectors.

    Two plus/minus constants supply the three nonzero magnitudes and zero.
    This replaces the decoder LUT only; packed weight codes and row scales do
    not change. Reject other integer bases rather than rounding their values.
    """
    if (
        table.dtype != np.int8
        or table.shape != (256, 8)
        or not np.isin(table, [-10, -6, -2, 0, 2, 6, 10]).all()
    ):
        raise ValueError("E8P selector decoder requires the spread29 integer basis")
    values = table.astype(np.int16)
    selectors = np.where(values == 0, 3, (np.abs(values) - 2) // 4) | (values < 0) * 4
    return np.sum(
        selectors.astype(np.uint32) << np.arange(0, 32, 4, dtype=np.uint32),
        axis=-1,
        dtype=np.uint32,
    ).reshape(256, 1)


def e8p_selector_supports_parity_xor(selectors):
    """Check the nonzero-basis contract for the exact XOR parity decoder."""
    if selectors.dtype != np.uint32 or selectors.shape != (256, 1):
        raise ValueError("Invalid E8P selector table")
    if np.any(selectors & np.uint32(0x88888888)):
        raise ValueError("Invalid E8P selector nibble")
    return bool(np.all(((selectors >> np.arange(0, 32, 4, dtype=np.uint32)) & 3) != 3))


def e8p_nibble_table(selectors):
    """Losslessly encode (w-1)/2 for the one-PRMT signed-nibble decoder.

    Zero basis coordinates need a different parity rule and are rejected.
    The paired activation split uses the same within-eight K permutation.
    """
    if not e8p_selector_supports_parity_xor(selectors):
        raise ValueError("Exact E8P nibble decoding requires a nonzero spread29 basis")
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    ids = (selectors >> shifts) & 7
    half = np.array([1, 3, 5, 0, -1, -3, -5, 0], np.int32)[ids]
    return np.bitwise_or.reduce(
        (half[:, [0, 2, 1, 3, 4, 6, 5, 7]] & 15).astype(np.uint32) << shifts, axis=1
    ).reshape(256, 1)


def e8p_repack_k64(packed):
    """Reorder K128 codes into contiguous K64 rows without changing any code."""
    if packed.dtype != np.uint16 or packed.ndim != 4 or packed.shape[-1] != 16:
        raise ValueError("Expected K128 E8P codes")
    e, g, n, _ = packed.shape
    return np.ascontiguousarray(packed.reshape(e, g, n, 2, 8).transpose(0, 1, 3, 2, 4)).reshape(
        e, g * 2, n, 8
    )


@dataclass(frozen=True)
class Weights:
    kind: str
    indices: np.ndarray
    table: np.ndarray
    scales: np.ndarray
    signs: np.ndarray
    patches: np.ndarray | None = None  # [N,K/128] U16: valid:1, signed value:8, position:7

    def validate(self):
        d, dtype = 8, np.uint16
        if (
            self.kind != "e8p"
            or self.indices.ndim != 2
            or min(self.indices.shape) <= 0
            or self.indices.dtype != dtype
        ):
            raise ValueError("Invalid vector indices")
        n, k = self.indices.shape[0], self.indices.shape[1] * d
        if k % 128 or self.table.shape != (256, d) or self.table.dtype != np.int8:
            raise ValueError("Invalid vector table/geometry")
        if (
            np.abs(self.table.astype(np.int16)).max() > 126
            or (self.table.astype(np.int16) % 2).any()
        ):
            raise ValueError("E8P basis must be even signed integers without shift overflow")
        if (
            self.scales.dtype != np.float16
            or self.scales.shape != (n,)
            or not np.isfinite(self.scales).all()
            or (self.scales <= 0).any()
        ):
            raise ValueError("Invalid vector row scales")
        if (
            self.signs.dtype != np.int8
            or self.signs.shape not in ((0,), (k,))
            or (len(self.signs) and not np.isin(self.signs, [-1, 1]).all())
        ):
            raise ValueError("Invalid vector rotation signs")
        if self.patches is not None and (
            self.patches.dtype != np.uint16 or self.patches.shape != (n, k // 128)
        ):
            raise ValueError("Invalid integer patch slots")
        return n, k

    @property
    def nbytes(self):
        return sum(
            a.nbytes
            for a in (self.indices, self.table, self.scales, self.signs, self.patches)
            if a is not None
        )

    def integer_weights(self):
        n, k = self.validate()
        q = e8p_decode(self.indices, self.table)
        q = q.reshape(n, k).copy()
        if self.patches is not None:
            p = self.patches
            rows, groups = np.where((p & 32768) != 0)
            positions = groups * 128 + (p[rows, groups] & 127)
            values = ((p[rows, groups] >> 7) & 255).astype(np.uint8).view(np.int8)
            q[rows, positions] = values
        return q

    def dequantize(self):
        return self.integer_weights().astype(np.float32) * self.scales.astype(np.float32)[:, None]

    def gpu_layout(self):
        n, k = self.validate()
        d = 8
        p = np.ascontiguousarray(self.indices.reshape(n, k // 128, 128 // d).transpose(1, 0, 2))
        table = np.ascontiguousarray(self.table).view("<u4").reshape(256, d // 4)
        return p, table, self.scales.copy()


def save(path, w, provenance=None):
    shape = w.validate()
    path = Path(path)
    with path.open("xb"):
        pass
    try:
        tensors = {name: getattr(w, name) for name in ("indices", "table", "scales", "signs")}
        if w.patches is not None:
            tensors["patches"] = w.patches
        save_file(
            tensors,
            str(path),
            metadata={
                "format": "orinfer.integer_vq.v1",
                "kind": w.kind,
                "shape": json.dumps(shape),
                "provenance": json.dumps(provenance or {}, sort_keys=True),
            },
        )
    except BaseException:
        path.unlink()
        raise


def load(path):
    with safe_open(path, framework="np") as f:
        m = f.metadata() or {}
        required = {"indices", "table", "scales", "signs"}
        if m.get("format") != "orinfer.integer_vq.v1" or set(f.keys()) not in (
            required,
            required | {"patches"},
        ):
            raise ValueError("Unsupported vector fixture")
        w = Weights(
            m.get("kind", ""),
            *[f.get_tensor(n) for n in ("indices", "table", "scales", "signs")],
            f.get_tensor("patches") if "patches" in f.keys() else None,
        )
        if list(w.validate()) != json.loads(m.get("shape", "null")):
            raise ValueError("Vector fixture shape mismatch")
        return w
