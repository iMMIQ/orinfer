"""Offline GGUF metadata reader and lossless safetensors container conversion.

Quantized tensors remain raw U8 blocks. Their source type/dimensions are
recorded in standard safetensors string metadata; this module never requantizes
weights. Online execution does not parse GGUF.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct


# GGML block elements/bytes. Q2_0 is the 64-element scalar format, type 42;
# older gguf Python releases omit it. Unsupported formats fail explicitly.
BLOCKS = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    20: (32, 18),
    23: (256, 136),
    30: (1, 2),
    42: (64, 18),
}
SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


@dataclass(frozen=True)
class Tensor:
    name: str
    dimensions: tuple[int, ...]  # GGML order: innermost dimension first.
    kind: int
    offset: int
    bytes: int

    @property
    def dtype(self):
        return {0: "F32", 1: "F16", 30: "BF16"}.get(self.kind, "U8")

    @property
    def storage_shape(self):
        shape = list(reversed(self.dimensions))
        if self.dtype == "U8":
            block, size = BLOCKS[self.kind]
            shape[-1] = shape[-1] // block * size
        return shape


@dataclass(frozen=True)
class Header:
    metadata: dict
    tensors: tuple[Tensor, ...]
    data_start: int
    file_size: int

    def safetensors_prefix(self):
        """Preserve payload offsets, representing alignment gaps as U8 tensors.

        The source's entire [data_start,file_size) byte sequence can be copied
        directly after this prefix, including padding. No full tensor allocation
        or reordering is needed, including for SSD-resident embedding tables.
        """
        result = {}
        metadata = {"orin.gguf_conversion": "1"}
        cursor = 0
        for tensor in sorted(self.tensors, key=lambda t: t.offset):
            if tensor.offset > cursor:
                result[f"__gguf_padding_{cursor}"] = {
                    "dtype": "U8",
                    "shape": [tensor.offset - cursor],
                    "data_offsets": [cursor, tensor.offset],
                }
            end = tensor.offset + tensor.bytes
            result[tensor.name] = {
                "dtype": tensor.dtype,
                "shape": tensor.storage_shape,
                "data_offsets": [tensor.offset, end],
            }
            metadata[f"ggml.type.{tensor.name}"] = str(tensor.kind)
            metadata[f"ggml.dimensions.{tensor.name}"] = json.dumps(tensor.dimensions)
            metadata[f"orin.layout.{tensor.name}"] = f"ggml_{tensor.kind}"
            cursor = end
        tail = self.file_size - self.data_start
        if cursor < tail:
            result[f"__gguf_padding_{cursor}"] = {
                "dtype": "U8",
                "shape": [tail - cursor],
                "data_offsets": [cursor, tail],
            }
        result["__metadata__"] = metadata
        raw = json.dumps(result, separators=(",", ":"), ensure_ascii=True).encode()
        raw += b" " * (-len(raw) % 8)
        return struct.pack("<Q", len(raw)) + raw


def read_header(handle, *, file_size):
    """Read little-endian GGUF v2/v3 metadata, rejecting malformed extents.

    handle may contain only the downloaded prefix. file_size is the full source
    size; an incomplete prefix raises EOFError, not a partially valid header.
    """

    def take(count):
        if count < 0 or handle.tell() + count > min(file_size, 64 * 1024**2):
            raise ValueError("Invalid GGUF metadata extent")
        value = handle.read(count)
        if len(value) != count:
            raise EOFError("Incomplete GGUF metadata")
        return value

    def scalar(fmt):
        return struct.unpack("<" + fmt, take(struct.calcsize(fmt)))[0]

    def string():
        return take(scalar("Q")).decode("utf-8")

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            inner, count = scalar("I"), scalar("Q")
            if inner == 9 or count > 1_000_000:
                raise ValueError("Invalid GGUF metadata array")
            return [value(inner) for _ in range(count)]
        if kind not in SCALARS:
            raise ValueError(f"Unsupported GGUF metadata type: {kind}")
        return scalar(SCALARS[kind])

    if take(4) != b"GGUF" or scalar("I") not in (2, 3):
        raise ValueError("Expected little-endian GGUF v2/v3")
    count, fields = scalar("Q"), scalar("Q")
    if not 0 < count <= 100_000 or fields > 100_000:
        raise ValueError("Invalid GGUF tensor/metadata count")
    metadata = {}
    for _ in range(fields):
        name = string()
        if name in metadata:
            raise ValueError("Duplicate GGUF metadata key")
        metadata[name] = value(scalar("I"))
    alignment = metadata.get("general.alignment", 32)
    if (
        type(alignment) is not int
        or alignment <= 0
        or alignment > 65536
        or alignment & (alignment - 1)
    ):
        raise ValueError("Invalid GGUF alignment")
    tensors, seen = [], set()
    for _ in range(count):
        name, rank = string(), scalar("I")
        if not name or name in seen or name == "__metadata__" or name.startswith("__gguf_padding_"):
            raise ValueError("Invalid or duplicate GGUF tensor name")
        if not 1 <= rank <= 4:
            raise ValueError("Invalid GGUF tensor rank")
        seen.add(name)
        dimensions = tuple(scalar("Q") for _ in range(rank))
        kind, offset = scalar("I"), scalar("Q")
        if kind not in BLOCKS:
            raise ValueError(f"Unsupported GGML weight type: {kind}")
        block, size = BLOCKS[kind]
        if not all(dimensions) or dimensions[0] % block or offset % alignment:
            raise ValueError("Invalid GGUF tensor dimensions/alignment")
        length = math.prod(dimensions) // block * size
        tensors.append(Tensor(name, dimensions, kind, offset, length))
    start = (handle.tell() + alignment - 1) // alignment * alignment
    cursor = 0
    for tensor in sorted(tensors, key=lambda t: t.offset):
        if tensor.offset < cursor or start + tensor.offset + tensor.bytes > file_size:
            raise ValueError("Overlapping or out-of-bounds GGUF tensor")
        cursor = tensor.offset + tensor.bytes
    return Header(metadata, tuple(tensors), start, file_size)


def convert(source, output):
    """Stream a local GGUF into a new safetensors file, preserving all payload."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    created = False
    try:
        with source.open("rb") as src:
            header = read_header(src, file_size=source.stat().st_size)
            prefix = header.safetensors_prefix()
            src.seek(header.data_start)
            with output.open("xb") as dst:
                created = True
                dst.write(prefix)
                remaining = header.file_size - header.data_start
                while remaining:
                    data = src.read(min(16 * 1024**2, remaining))
                    if not data:
                        raise EOFError("Truncated GGUF payload")
                    dst.write(data)
                    remaining -= len(data)
        return header
    except Exception:
        # Exclusive creation above means only this attempt's partial is removed.
        if created:
            output.unlink(missing_ok=True)
        raise
