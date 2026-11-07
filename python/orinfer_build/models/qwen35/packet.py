"""Lift unchanged Qwen packet boundaries into address-table dynamic kernels."""

from orinfer_build.compiler.compile import compile_function, instantiate
from orinfer_build.compiler.private_rows import wrap_private
from orinfer_build.models.qwen35.parallel import LANES, TABLE

# Shared with qwen3_5::rows::PACKET_COLUMNS; appended after the layer table.
COLUMNS = (
    "Input",
    "Token",
    "TokenStatus",
    "Logits",
    "Step",
    "Positions",
    "SeqLength",
    "SegmentLength",
    "LastIndex",
    "Requests",
    "MRopePositions",
    "Features",
    "FeatureIndex",
    "AudioFeatures",
    "AudioFeatureIndex",
    "MtpTargetHidden",
)
ROW_BUFFERS = {"Hidden", "R0", "HeadHidden", "HeadLogits", "Pairs"}


def build(b):
    from kernels.cuda.model import qwen35, control, e8p_q4, packet_io, speculation
    from kernels.shared.rows import static_rows
    from kernels.cuda.operators.op23_final_norm import final_norm
    from kernels.cuda.vision.features import overlay

    w = b.weights
    base = LANES * len(b.text["layer_types"]) * len(b.state_columns)
    w.buffers[TABLE]["shape"] = [base + LANES * len(COLUMNS)]
    kernels = {k["name"]: k for k in b.aot.kernels}

    def invoke(factory, *args, **kwargs):
        return static_rows(instantiate(factory, *args, **kwargs))

    def emit(role, function, pointers):
        pointers = {
            key: dict(kind="buffer", name=value) if isinstance(value, str) else value
            for key, value in pointers.items()
        }
        modes = {}
        for buffer in function.buffer_map.values():
            if buffer.name not in pointers:
                continue
            name = pointers[buffer.name]["name"]
            modes[buffer.name] = (
                "weight"
                if w.buffers[name].get("data") is not None
                else "row"
                if name in ROW_BUFFERS
                else "private",
                name,
            )
        wrapped, bindings = wrap_private(
            function, LANES, modes, COLUMNS, pointers, TABLE, table_offset=base * 8
        )
        b.emit(
            "packet/" + role,
            f"packet-{role}-{b.h}-{b.v}-{b.context}-{b.standard}",
            lambda: compile_function(wrapped),
            bindings,
            LANES,
        )

    def existing(role, factory, *args, original=None, **kwargs):
        old = kernels["qwen_rows/" + (original or role)]
        emit(role, invoke(factory, *args, **kwargs), b.aot.compiler.pointers(old))

    emit("tokens", invoke(packet_io.token_input), dict(Token="Token", Input="Input"))
    existing(
        "prepare", qwen35.prepare, 1, b.h, b.context, mrope_input=True, original="begin/prepare"
    )
    existing(
        "embedding",
        e8p_q4.embedding if b.embedding_q4 else qwen35.embedding,
        1,
        b.v,
        b.h,
        original="begin/embedding",
    )
    media = b.audio or b.vision
    if media:
        existing(
            "overlay", overlay, 1, b.h, b.context, media["max_features"], original="begin/overlay"
        )
    existing("advance", control.advance, 1, original="end/advance")
    if b.mtp:
        existing(
            "capture",
            speculation.capture_target_hidden,
            b.h,
            4096,
            ring=True,
            original="capture/hidden",
        )
    existing(
        "norm", final_norm, 1, 1, hidden=b.h, zero_centered=not b.standard, original="gather/norm"
    )
    emit(
        "outputs",
        invoke(packet_io.outputs, b.v),
        dict(
            Source="HeadLogits", Pair="Pairs", Logits="Logits", Token="Token", Status="TokenStatus"
        ),
    )
