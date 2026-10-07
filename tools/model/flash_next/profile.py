"""Optional CUDA-event diagnostic phases; excluded from throughput trials."""
from collections import defaultdict
from contextlib import contextmanager
import time

import torch


class PhaseTrace:
    def __init__(self):self.records=[]

    @contextmanager
    def phase(self, name):
        start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin=time.perf_counter();start.record()
        yield
        end.record();self.records.append((name,start,end,(time.perf_counter()-begin)*1000))

    def summary(self):
        torch.cuda.synchronize();result=defaultdict(lambda:dict(calls=0,wall_ms=0.,cuda_span_ms=0.))
        for name,start,end,wall in self.records:
            row=result[name];row['calls']+=1;row['wall_ms']+=wall;row['cuda_span_ms']+=start.elapsed_time(end)
        return dict(result)
