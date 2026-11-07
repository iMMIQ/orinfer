"""Qwen model declarations for the shared packet operand-table compiler."""

import math
import re
from orinfer_build.compiler.packet_graph import TABLE as TABLE, build as shared_build


def build(b):
    columns = sorted(n for n, scope in b.weights.scopes.items() if scope == "sequence")
    copy_specs = (
        [
            ("Token", "Input", 4),
            ("MtpToken", "MtpInput", 4),
            ("MtpLastHidden", "MtpCondition", b.h * 2),
            ("MtpCondition", "HeadHidden", b.h * 2),
            ("HeadLogits", "MtpLogits", b.v * 4),
            ("HeadLogits", "SequenceLogits", b.v * 4),
            ("HeadLogits", "Logits", b.v * 4),
        ]
        if b.mtp
        else []
    )
    # Mixed packets also commit each ordinary GDN segment's convolution tail.
    for destination in columns:
        if re.fullmatch(r"L\d+_History", destination):
            buffer = b.weights.buffers[destination]
            if buffer["dtype"] != "f16":
                raise ValueError("Expected FP16 Qwen convolution history")
            copy_specs.append(("HistoryOut", destination, math.prod(buffer["shape"]) * 2))
    b.packet_copies = copy_specs
    return shared_build(b)
