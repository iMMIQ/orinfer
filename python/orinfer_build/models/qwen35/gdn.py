"""Emit the shared chunkwise prefill recurrence around request-owned FP32 state."""

from kernels.cuda.model import gdn_chunk
from orinfer_build.compiler.compile import primitive
from orinfer_build.tuning.gdn_tuning import chunked


def prefill(b, section, layer):
    from orinfer_build.common.gpu import gpu_lock

    w, hv = b.weights, b.hv
    if "GdnChunkW" not in w.buffers:
        for name, dtype, shape in [
            ("GdnChunkW", "f16", [hv, 4096, 128]),
            ("GdnChunkU", "f16", [hv, 4096, 128]),
            ("GdnChunkD", "f16", [hv, 4096, 32]),
            ("GdnChunkG", "f32", [hv, 4096, 2]),
        ]:
            w.fixed(name, dtype, shape)
    with gpu_lock(b.lock):
        selection = chunked(b.row_tuner)
    config = selection["configs"][int(selection["default"])]
    b.gdn_execution["prefill"] = "chunked"
    common = dict(Q="Q", K="K", W="GdnChunkW", U="GdnChunkU", D="GdnChunkD", GP="GdnChunkG")
    identity = f"{b.hk}-{hv}-{config}"
    b.emit(
        section + "/recurrent_prepare",
        "gdn-wy-prepare-" + identity,
        lambda: primitive(
            gdn_chunk.prepare, 4096, HK=b.hk, HV=hv, C=config["C"], dynamic_rows=True
        ),
        dict(common, V="V", G="g", Beta="Beta"),
    )
    b.emit(
        section + "/recurrent_chunk",
        "gdn-wy-recurrent-" + identity,
        lambda: primitive(gdn_chunk.recurrent, 4096, HK=b.hk, HV=hv, **config, dynamic_rows=True),
        dict(common, State=f"L{layer}_State", Out="Y"),
    )
