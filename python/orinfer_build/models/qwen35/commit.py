"""Batch exact-prefix restoration with the shared independent-owner wrapper."""

import hashlib
from kernels.cuda.model.gdn_sequence import gdn_commit_controlled
from kernels.cuda.model.speculation import select_prefix
from orinfer_build.compiler.compile import compile_function, instantiate
from orinfer_build.compiler.private_rows import wrap_private
from orinfer_build.models.qwen35.parallel import LANES

TABLE = "QwenCommitPointers"
COLUMNS = (
    "VerifyK",
    "VerifyG",
    "VerifyUpdate",
    "AcceptedInputs",
    "SegmentLength",
    "State",
    "HistoryPrefixes",
    "History",
)


def factory(role, hk, hv, channels, capacity):
    if role == "state":
        if capacity:
            from kernels.cuda.model.gdn_deferred import commit

            fn = commit(capacity, key_heads=hk, value_heads=hv)
        else:
            fn = instantiate(gdn_commit_controlled, 8, key_heads=hk, value_heads=hv)
        mapping = dict(
            K="VerifyK",
            G="VerifyG",
            Update="VerifyUpdate",
            Accepted="AcceptedInputs",
            Length="SegmentLength",
            State="State",
        )
        if capacity:
            mapping.update(
                LogK="GdnLogKey", LogD="GdnLogDelta", LogG="GdnLogGate", Count="GdnLogCount"
            )
    elif role == "history":
        fn = instantiate(select_prefix, 8, 3 * channels, "float16")
        mapping = dict(Prefix="HistoryPrefixes", Count="AcceptedInputs", State="History")
    else:
        raise ValueError("Unknown commit role")
    roles = {name: ("private", identity) for name, identity in mapping.items()}
    return fn, roles


def build(b):
    layers = len(b.text["layer_types"])
    table = b.mtp.get("verification_table")
    columns = table["columns"] if table else COLUMNS
    target = table["buffer"] if table else TABLE
    capacity = 0
    decode = b.gdn_execution["decode"]
    if decode["kind"] == "deferred":
        capacity = decode["capacity"]
    if not table:
        b.weights.fixed(target, "u64", [LANES * layers * len(columns)])
    for layer, kind in enumerate(b.text["layer_types"]):
        if kind != "linear_attention":
            continue
        for role in ("state", "history"):
            fn, roles = factory(role, b.hk, b.hv, b.qkv, capacity)
            fn, bindings = wrap_private(
                fn,
                LANES,
                roles,
                columns,
                {},
                target,
                table_offset=layer * len(columns) * 8,
                row_stride=layers * len(columns),
            )
            b.emit(
                f"commit/layer{layer}/{role}",
                "commit-rows-" + hashlib.sha256(str(fn).encode()).hexdigest(),
                lambda fn=fn: compile_function(fn),
                bindings,
                LANES,
            )
