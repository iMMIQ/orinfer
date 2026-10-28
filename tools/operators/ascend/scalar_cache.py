"""Maintain A2 Scalar GM cache coherence across native kernel boundaries."""

import re


def lower_scalar_cache(source):
    tensors = {
        name: dtype
        for dtype, name in re.findall(r"AscendC::GlobalTensor<([^>]+)>\s+(\w+)\s*;", source)
    }
    readers = {name for name in tensors if re.search(rf"\b{re.escape(name)}\.GetValue\(", source)}
    writers = {name for name in tensors if re.search(rf"\b{re.escape(name)}\.SetValue\(", source)}
    if not readers and not writers:
        return source
    kernel = re.search(r'extern "C" __global__ __aicore__ void \w+\([^)]*\)\s*\{', source)
    if kernel is None:
        raise ValueError("Scalar GM access has no recognized device kernel")
    opening = kernel.end() - 1
    stack, closing, regions = [], None, []
    for offset in range(opening, len(source)):
        if source[offset] == "{":
            stack.append(offset)
        elif source[offset] == "}":
            start = stack.pop()
            if source[max(0, start - 24) : start].strip().endswith("if ASCEND_IS_AIC"):
                regions.append((start, offset))
            if not stack:
                closing = offset
                break
    if closing is None:
        raise ValueError("Unbalanced AOT kernel body")
    if writers:
        if not re.search(r"\w+<<<1, nullptr, stream>>>", source):
            raise ValueError("Scalar GM writes require one Cube core; use DMA for parallel output")
        for name in writers:
            for match in re.finditer(rf"\b{re.escape(name)}\.SetValue\(", source):
                if not any(start < match.start() < end for start, end in regions):
                    raise ValueError("Scalar GM writes must belong to the single Cube scope")
    bindings = list(re.finditer(r"\w+\.SetGlobalBuffer\([^;]+;", source[opening:closing]))
    if not bindings:
        raise ValueError("Scalar GM access has no bound GlobalTensor")
    entry = opening + bindings[-1].end()
    anchor = sorted(readers | writers)[0]
    flush = (
        f"AscendC::DataCacheCleanAndInvalid<{tensors[anchor]}, "
        f"AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>({anchor});"
    )
    # Entry also covers inputs updated by DMA or another kernel between replays.
    # Every current GM Scalar writer has one Cube owner. Its dirty lines must
    # reach GM before subsequent DMA consumers on other cores can observe them.
    ending = f"\n  if ASCEND_IS_AIC {{ {flush} }}\n" if writers else ""
    return source[:entry] + "\n  " + flush + source[entry:closing] + ending + source[closing:]
