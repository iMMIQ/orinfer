"""Bounded CPU cache of immutable decoded embedding rows, owned by a model."""

from collections import OrderedDict

import numpy as np


class RowCache:
    def __init__(self, source, *, capacity_bytes, dtype=np.float32):
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("Nonnegative row cache capacity required")
        self.source = source
        self.capacity = capacity_bytes
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("Row cache dtype must be FP16 or FP32")
        self.rows = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def read(self, name, row):
        key = (name, row)
        if key in self.rows:
            self.hits += 1
            self.rows.move_to_end(key)
            return self.rows[key]
        self.misses += 1
        value = np.array(self.source.rows(name, row, 1)[0], dtype=self.dtype, copy=True)
        value.setflags(write=False)
        # Reserve space for array/key/LRU objects as well as decoded payload.
        size = value.nbytes + 512
        if size <= self.capacity:
            while self.bytes + size > self.capacity:
                _, old = self.rows.popitem(last=False)
                self.bytes -= old.nbytes + 512
            self.rows[key] = value
            self.bytes += size
        return value

    def clear(self):
        self.rows.clear()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
