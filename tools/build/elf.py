"""Strip offline native libraries while preserving every runtime section byte."""

from pathlib import Path
import struct
import subprocess


def runtime_sections(path):
    with Path(path).open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:6] != b"\x7fELF\x02\x01":
            raise ValueError("Expected a little-endian ELF64 library")
        offset = struct.unpack_from("<Q", header, 40)[0]
        size, count, names_index = struct.unpack_from("<HHH", header, 58)
        if size != 64 or not 0 < names_index < count:
            raise ValueError("Invalid ELF section table")
        stream.seek(offset)
        sections = [struct.unpack("<IIQQQQIIQQ", stream.read(size)) for _ in range(count)]
        names = sections[names_index]
        stream.seek(names[4])
        strings = stream.read(names[5])
        result = {}
        for section in sections:
            name = strings[section[0] :].split(b"\0", 1)[0]
            if section[2] & 2 or name.startswith((b".aicore", b"__aicore")):
                stream.seek(section[4])
                data = b"" if section[1] == 8 else stream.read(section[5])
                result[name] = (section[1:4], section[5], section[8], data)
        return result


def strip_library(path):
    before = runtime_sections(path)
    subprocess.run(["strip", "--strip-unneeded", str(path)], check=True)
    if runtime_sections(path) != before:
        raise ValueError(f"Stripping changed runtime sections: {path}")
