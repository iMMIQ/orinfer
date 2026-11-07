"""Lossless INT4-element transpose and NZ block packing for A2 grouped GEMM."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def transposed_nz_blocks(packed):
    """ND bytes [E,K,N/2] -> NZ bytes [E,K/64,N/16,16,32]."""
    if packed.dtype != np.uint8 or packed.ndim != 3:
        raise ValueError("Expected unsigned packed INT4 byte planes")
    experts, k, half_n = packed.shape
    n = half_n * 2
    if experts < 1 or k < 1 or k % 64 or n < 1 or n % 16:
        raise ValueError("Transposed NZ requires K%64=0 and N%16=0")
    for expert in range(experts):
        for first in range(0, k, 64):
            source = packed[expert, first : first + 64]
            codes = np.empty((64, n), dtype=np.uint8)
            codes[:, 0::2] = source & 15
            codes[:, 1::2] = source >> 4
            transposed = codes.T
            yield np.ascontiguousarray(transposed[:, 0::2] | (transposed[:, 1::2] << 4)).reshape(
                n // 16, 16, 32
            )


def restore_nd(packed_nz):
    if packed_nz.dtype != np.uint8 or packed_nz.ndim != 5 or packed_nz.shape[-2:] != (16, 32):
        raise ValueError("Expected packed INT4 transposed NZ storage")
    e, k1, n1, _, _ = packed_nz.shape
    codes = np.empty((e, k1, n1, 16, 64), dtype=np.uint8)
    codes[..., 0::2] = packed_nz & 15
    codes[..., 1::2] = packed_nz >> 4
    logical = codes.reshape(e, k1, n1 * 16, 64).transpose(0, 1, 3, 2).reshape(e, k1 * 64, n1 * 16)
    return np.ascontiguousarray(logical[..., 0::2] | (logical[..., 1::2] << 4))


def transposed_nz_file_chunks(path, offset, experts, k, n):
    """Read aligned K blocks positionally without mapping the full weight bank."""
    if experts < 1 or k < 1 or k % 64 or n < 1 or n % 16 or offset < 0:
        raise ValueError("Transposed NZ requires positive aligned geometry and nonnegative offset")
    with Path(path).open("rb", buffering=8 << 20) as source:
        source.seek(offset)
        for _expert in range(experts):
            for _first in range(0, k, 64):
                data = source.read(64 * (n // 2))
                if len(data) != 64 * (n // 2):
                    raise EOFError("Truncated packed INT4 bank")
                block = np.frombuffer(data, dtype=np.uint8).reshape(1, 64, n // 2)
                for packed in transposed_nz_blocks(block):
                    yield memoryview(packed).cast("B")


def convert(source, output, experts, k, n):
    if source.stat().st_size != experts * k * n // 2:
        raise ValueError("Packed source extent differs")
    if output.exists():
        raise ValueError("Preserve existing immutable banks")
    stage = output.with_suffix(".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with stage.open("wb", buffering=8 << 20) as f:
        for block in transposed_nz_file_chunks(source, 0, experts, k, n):
            f.write(block)
        f.flush()
        os.fsync(f.fileno())
    stage.replace(output)
    if output.stat().st_size != source.stat().st_size:
        raise ValueError("Lossless NZ byte extent differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for fixture in json.loads(args.fixtures.read_text()):
        k, n, e = fixture["input"], fixture["output"], fixture["experts"]
        target = args.output / f"w4-{k}-{n}.bin"
        convert(Path(fixture["files"][0]), target, e, k, n)
        print(
            json.dumps(
                dict(
                    file=str(target),
                    bytes=target.stat().st_size,
                    shape=[e, k // 64, n // 16, 16, 32],
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
