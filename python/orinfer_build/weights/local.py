"""Persistent pread sources sharing the bounded safetensors format contract."""

import json
import math
import os
from pathlib import Path

from orinfer_build.weights.safetensors_source import SIZES, read_header, relative_name


class LocalSource:
    def __init__(self, directory, index="model.safetensors.index.json"):
        self.directory = Path(directory).resolve()
        self.index = json.loads((self.directory / index).read_text())["weight_map"]
        self.files, self.headers, self.signatures = {}, {}, {}

    def tensor(self, name):
        filename = self.index[name]
        if filename not in self.files:
            relative_name(filename)
            path = (self.directory / filename).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError("Checkpoint path leaves model directory")
            fd = os.open(path, os.O_RDONLY)
            try:
                stat = os.fstat(fd)
                self.headers[filename] = read_header(
                    lambda offset, count: os.pread(fd, count, offset),
                    file_size=stat.st_size,
                    max_header_bytes=256 << 20,
                )
            except BaseException:
                os.close(fd)
                raise
            self.files[filename] = fd
            self.signatures[filename] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
        begin, header = self.headers[filename]
        return filename, begin, header[name]

    def metadata(self, name):
        return self.tensor(name)[2]

    def read_rows(self, name, first=0, count=None, *, max_bytes=None):
        filename, begin, info = self.tensor(name)
        shape = info["shape"]
        rows = shape[0] if shape else 1
        if type(first) is not int:
            raise ValueError("Invalid tensor row range")
        count = rows - first if count is None else count
        if type(count) is not int or not 0 <= first <= rows or not 0 <= count <= rows - first:
            raise ValueError("Invalid tensor row range")
        stride = math.prod(shape[1:]) * SIZES[info["dtype"]]
        size = count * stride
        if max_bytes is not None and size > max_bytes:
            raise ValueError("Tensor read exceeds bounded byte budget")
        data = os.pread(
            self.files[filename], size, begin + info["data_offsets"][0] + first * stride
        )
        if len(data) != size:
            raise ValueError("Truncated tensor payload")
        return data

    def close(self):
        for fd in self.files.values():
            os.close(fd)
        self.files.clear()
        self.headers.clear()
