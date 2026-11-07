"""Owner-parallel causal kernels with device lengths and shared row arenas."""

import hashlib
import json

from kernels.shared.rows import explicit_scalar
from orinfer_build.models.qwen35.parallel import (
    LANES,
    ROW_BUFFERS,
    state_role,
    attention_pointers,
    geometry,
)
from orinfer_build.compiler.private_rows import PrivateRows
from orinfer_build.compiler.compile import compile_function, instantiate as invoke

TABLE = "QwenVerifyPointers"
OFFSET = "QwenVerifyRowOffset"
SCRATCH = "QwenRowScratch"


def symbolic_rows(function):
    from tvm import tirx as tir

    variables = set()
    for buffer in function.buffer_map.values():
        for expression in [*buffer.shape, *buffer.strides]:
            tir.stmt_functor.post_order_visit(
                expression,
                lambda node: (
                    variables.add(node)
                    if isinstance(node, tir.Var) and node.name in ("m", "rows", "tokens")
                    else None
                ),
            )
    for variable in variables:
        function = explicit_scalar(function, variable)
    return function, variables


def factories(b, attention, conv):
    from kernels.cuda.model import flash_mtp, speculative_factors, gdn_deferred
    from kernels.cuda.model.gdn_sequence import _gdn_sequence
    from kernels.cuda.model.kv_int8 import full_prepare_mrope_int8
    from kernels.cuda.model.attention_rows import causal_attention_int8, merge_attention_int8
    from kernels.cuda.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.cuda.operators.op09_gdn_gates import _compile_gates
    from kernels.cuda.operators.op17_gdn_gated_norm import _compile_gated_norm

    gqa = dict(
        block_m=attention["block_m"],
        target_tiles=16,
        max_splits=attention["max_splits"],
        query_heads=b.qh,
        kv_heads=b.kvh,
        head_group=attention["head_group"],
        independent_queries=attention.get("independent_queries", False),
    )
    common = dict(key_heads=b.hk, value_heads=b.hv)
    out = {
        "gates": lambda: invoke(_compile_gates, packed_ab=True, heads=b.hv),
        "history": lambda: invoke(flash_mtp.history_prefix, 8, b.qkv, 3, dynamic_rows=True),
        "conv": lambda: invoke(gdn_conv_prep, 1, 8, dynamic_tokens=True, **conv, **common),
        "recurrent": lambda: invoke(
            _gdn_sequence, 8, value_tile=16, threads=32, compact=True, dynamic_rows=True, **common
        ),
        "factors": lambda: invoke(
            speculative_factors.save_gdn_factors, 8, dynamic_rows=True, **common
        ),
        "gated_norm": lambda: invoke(_compile_gated_norm, None, heads=b.hv),
        "prepare": lambda: invoke(
            full_prepare_mrope_int8,
            (b.context + 127) // 128,
            b.context,
            (11, 11, 10),
            max_position=b.context,
            query_heads=b.qh,
            kv_heads=b.kvh,
        ),
        "attention": lambda: invoke(
            causal_attention_int8,
            8,
            context=b.context,
            block_n=attention["block_n"],
            threads=attention.get("threads", 128),
            stages=attention.get("stages", 0),
            **gqa,
        ),
        "attention_merge": lambda: invoke(merge_attention_int8, 8, **gqa),
    }
    decode = b.gdn_execution["decode"]
    if decode["kind"] == "deferred":
        out["recurrent"] = lambda: gdn_deferred.verification(decode["capacity"], **common)
    return out


def build(b, *, attention=None, conv=None):
    selection = attention
    canonical = (
        geometry(selection) if selection is not None else b.row_tuner.causal_attention(b.context)
    )
    canonical = dict(canonical, independent_queries=True)
    configs, choices = {}, []
    for width in range(2, 9):
        config = canonical
        key = json.dumps(config, sort_keys=True)
        name = f"attention{len(configs)}" if key not in configs else configs[key][0]
        configs[key] = (name, config)
        if choices and choices[-1]["program"] == name:
            choices[-1]["max_tokens"] = width
        else:
            choices.append(dict(min_tokens=width, max_tokens=width, program=name))
    attention = canonical
    conv = conv or geometry(b.row_tuner.gdn_convolution())
    functions = factories(b, attention, conv)
    b.weights.fixed(OFFSET, "i32", [1], "sequence")
    original = {k["name"]: k for k in b.aot.kernels}
    entries = []
    private = PrivateRows(
        b.weights,
        ROW_BUFFERS,
        SCRATCH,
        state_role,
        LANES,
        len(b.text["layer_types"]),
        columns=(OFFSET, "SegmentLength", "AcceptedInputs"),
    )
    for layer, kind in enumerate(b.text["layer_types"]):
        roles = (
            [
                "gates",
                "history",
                "conv",
                *(["recurrent_materialize"] if "recurrent_materialize" in functions else []),
                "recurrent",
                "factors",
                "gated_norm",
            ]
            if kind == "linear_attention"
            else ["prepare"]
        )
        work = [(role, role, functions[role]) for role in roles]
        if kind != "linear_attention":
            for name, config in configs.values():
                selected = factories(b, config, conv)
                work.extend(
                    (f"{name}/{role}", role, selected[role])
                    for role in ("attention", "attention_merge")
                )
        for emitted, role, factory in work:
            verify = role in ("history", "recurrent", "factors")
            name = f"qwen_rows/{'verify/' if verify else ''}layer{layer}/{'' if verify else 'private/'}{role}"
            fn, variables = symbolic_rows(factory())
            pointers = b.aot.compiler.pointers(original[name])
            if role == "attention":
                prepared = original[f"qwen_rows/layer{layer}/private/prepare"]
                pointers = attention_pointers(pointers, b.aot.compiler.pointers(prepared))
            if role == "recurrent":
                pointers.update(
                    {
                        arg: dict(kind="buffer", name=f"L{layer}_{target}")
                        for arg, target in dict(
                            LogK="GdnLogKey",
                            LogD="GdnLogDelta",
                            LogG="GdnLogGate",
                            Count="GdnLogCount",
                        ).items()
                    }
                )
            modes, pointers = private.bind(fn, pointers, extents={v: 8 for v in variables})
            entries.append((layer, emitted, fn, variables, modes, pointers))
    offsets, cursor, columns = private.layout()
    b.weights.buffers[SCRATCH]["shape"][0] = max(b.weights.buffers[SCRATCH]["shape"][0], cursor)
    layers = len(b.text["layer_types"])
    b.weights.fixed(TABLE, "u64", [LANES * layers * len(columns)])
    b.mtp["verification_table"] = dict(
        buffer=TABLE,
        columns=columns,
        row_offset=OFFSET,
        deferred=b.gdn_execution["decode"]["kind"] == "deferred",
        attention=choices,
    )
    for layer, role, fn, variables, modes, pointers in entries:
        fn, bindings = private.wrap(
            fn,
            modes,
            pointers,
            layer,
            TABLE,
            scalar_columns={v.name: ("SegmentLength", 8) for v in variables},
            row_offset=OFFSET,
            row_capacity=4096,
            batch_parameters=("independent_instances",)
            if role in ("attention", "attention_merge")
            else (),
        )

        key = "causal-rows-" + role + "-" + hashlib.sha256(str(fn).encode()).hexdigest()
        b.emit(
            f"verify_state/layer{layer}/{role}",
            key,
            lambda fn=fn: compile_function(fn),
            bindings,
            LANES,
        )
