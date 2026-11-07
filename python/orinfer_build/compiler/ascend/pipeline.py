"""Keep independent gather-slot DMAs in flight through the pinned compiler.

The TileLang producer marks a loop containing only immutable GM reads and
disjoint UB slot writes. Explicit full fences surround that loop. The pinned
auto-sync pass adds full fences around scalar control inside it; those fences
serialize independent slots and can be removed within this declared region.
"""

import re


def lower_gather_pipeline(source, *, required=False):
    globals_ = re.findall(r"AscendC::GlobalTensor<[^>]+>\s+(\w+)\s*;", source)
    marker = re.compile(r"AscendC::(SetFlag|WaitFlag)<AscendC::HardEvent::S_MTE2>\(7\);")
    matches = list(marker.finditer(source))
    if not matches:
        if required:
            raise ValueError("Compiler discarded the declared gather pipeline region")
        return source
    if len(matches) % 2 or any(
        v.group(1) != ("SetFlag" if i % 2 == 0 else "WaitFlag") for i, v in enumerate(matches)
    ):
        raise ValueError("Gather pipeline regions must be paired and unnested")
    for begin, end in reversed(list(zip(matches[::2], matches[1::2], strict=True))):
        body = source[begin.end() : end.start()]
        # Vector arithmetic, stores to GM and cross-core communication require
        # their own dependence analysis and cannot enter this narrow schedule.
        if (
            any(re.search(rf"\b{re.escape(name)}\.SetValue\(", body) for name in globals_)
            or any(
                call not in {"PipeBarrier", "SetFlag", "WaitFlag"}
                for call in re.findall(r"AscendC::(\w+)\s*[<(]", body)
            )
            or any(
                call != "copy_gm_to_ub" for call in re.findall(r"tl::ascend::(\w+)\s*[<(]", body)
            )
        ):
            raise ValueError("Gather pipeline contains an unsupported operation")
        body = re.sub(r"AscendC::PipeBarrier<PIPE_ALL>\(\);", "", body)
        source = source[: begin.start()] + body + source[end.end() :]
    return source
