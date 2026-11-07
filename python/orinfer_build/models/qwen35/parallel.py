"""Request-parallel M1 state, wrapping the original TileLang arithmetic."""

import re

from orinfer_build.compiler.private_rows import PrivateRows
from orinfer_build.compiler.compile import compile_function, instantiate as invoke

TABLE = "QwenRowPointers"
SCRATCH = "QwenRowScratch"
LANES = 128
ROW_BUFFERS = {"QKV", "Zout", "AB", "FullX", "FullQ", "FullGate", "MixerIn"}


def state_role(name):
    return re.sub(r"^L\d+_", "", name)


def attention_pointers(pointers, prepared):
    """Bind owner KV from its writer, independent of prefill's selected ABI."""
    result = dict(pointers)
    for name in ("K", "V", "KS", "VS"):
        alias = "live_" + name.lower()
        pointer = prepared.get(name) or prepared.get(alias)
        if pointer is None:
            raise ValueError(f"Missing prepared KV pointer {name}")
        result[name] = pointer
        if alias in result:
            result[alias] = pointer
    return result


def geometry(selection, row=1):
    key = selection["default"]
    for choice in selection["selections"]:
        if (
            choice["min_rows"] <= row <= choice["max_rows"]
            and row % choice.get("multiple_of", 1) == 0
        ):
            key = choice["implementation"]
    return selection["configs"][int(key)]


def factory(role, context, attention, text=None, recurrent=None):
    from kernels.cuda.model import kv_int8
    from kernels.cuda.model.attention_rows import causal_attention_int8, merge_attention_int8
    from kernels.cuda.model.gdn_sequence import _gdn_sequence
    from kernels.shared.rows import static_rows
    from kernels.cuda.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.cuda.operators.op09_gdn_gates import _compile_gates
    from kernels.cuda.operators.op17_gdn_gated_norm import _compile_gated_norm
    from orinfer_build.tuning.row_tuning import ATTENTION_TARGET

    text = text or {}
    hk, hv = text.get("linear_num_key_heads", 16), text.get("linear_num_value_heads", 48)
    qh, kvh = text.get("num_attention_heads", 24), text.get("num_key_value_heads", 4)
    dim = text.get("head_dim", 256)
    gated = text.get("attn_output_gate", True)

    if role == "gates":
        fn = invoke(_compile_gates, packed_ab=True, heads=hv)
    elif role == "conv":
        # A request-owned decode CTA has exactly one time row. The compact
        # layout preserves the prefill reduction tree and caches the old raw
        # tail before updating this head's history in place.
        fn = invoke(
            gdn_conv_prep,
            1,
            1,
            compact_decode=True,
            inplace_history=True,
            key_heads=hk,
            value_heads=hv,
        )
    elif role == "recurrent" and (recurrent or {}).get("deferred"):
        from kernels.cuda.model.gdn_deferred import recurrent as logged_recurrent

        fn = logged_recurrent(
            recurrent["capacity"],
            key_heads=hk,
            value_heads=hv,
            value_tile=recurrent["value_tile"],
            threads=recurrent["threads"],
        )
    elif role == "recurrent":
        fn = invoke(
            _gdn_sequence,
            1,
            in_place=True,
            key_heads=hk,
            value_heads=hv,
            **(recurrent or dict(value_tile=16, threads=32, prefetch_qk=True)),
        )
    elif role == "gated_norm":
        fn = invoke(_compile_gated_norm, 1, heads=hv)
    elif role == "prepare":
        fn = invoke(
            kv_int8.full_prepare_mrope_int8,
            (context + 127) // 128,
            context,
            (11, 11, 10),
            max_position=context,
            query_heads=qh,
            kv_heads=kvh,
            dimension=dim,
            rotary_dim=64 if gated else dim,
            gated=gated,
            zero_centered=gated,
        )
    else:
        options = dict(
            block_m=attention["block_m"],
            query_heads=qh,
            kv_heads=kvh,
            target_tiles=ATTENTION_TARGET,
            max_splits=attention["max_splits"],
            head_group=attention["head_group"],
            parallel_instances=True,
            independent_queries=attention.get("independent_queries", False),
            dimension=dim,
            gated=gated,
        )
        if role == "attention":
            options.update(
                context=context,
                block_n=attention["block_n"],
                threads=attention.get("threads", 128),
                stages=attention.get("stages", 0),
            )
            fn = invoke(causal_attention_int8, 1, **options)
        elif role == "attention_merge":
            fn = invoke(merge_attention_int8, 1, **options)
        else:
            raise ValueError(f"Unknown Qwen private role: {role}")
    return static_rows(fn)


def build(b, attention=None, recurrent=None):
    """Bind dynamic private slots, including an optional whole log pipeline.

    Changing log capacity while a request is live requires materialization;
    one package therefore uses one log geometry for its complete row domain.
    """
    from orinfer_build.common.gpu import gpu_lock

    if attention is None:
        with gpu_lock(b.lock):
            attention = b.row_tuner.causal_attention(b.context)
    if recurrent is None:
        if any(kind == "linear_attention" for kind in b.text["layer_types"]):
            with gpu_lock(b.lock):
                recurrent = b.row_tuner.private_gdn()
        else:
            recurrent = dict(configs=[{}], default="0", selections=[], used=["0"])
    w = b.weights
    recurrent_choices = list(b.row_choices(recurrent, LANES))
    deferred_configs = [c for _, c in recurrent_choices if c.get("deferred")]
    deferred = bool(deferred_configs)
    if deferred:
        if len(recurrent_choices) != 1:
            raise ValueError("Deferred GDN requires one request-state geometry")
        from kernels.cuda.model import gdn_deferred
        from orinfer_build.compiler.compile import primitive

        config = deferred_configs[0]
        capacity = config["capacity"]
        b.gdn_execution["decode"] = dict(kind="deferred", capacity=capacity)
        hk, hv = b.text["linear_num_key_heads"], b.text["linear_num_value_heads"]
        gdn_deferred.validate_geometry(capacity, hk, hv, config["value_tile"], config["threads"])
        for layer, kind in enumerate(b.text["layer_types"]):
            if kind != "linear_attention":
                continue
            bindings = dict(State=f"L{layer}_State")
            for argument, suffix, dtype, shape in [
                ("LogK", "GdnLogKey", "f16", [capacity, hk, 128]),
                ("LogD", "GdnLogDelta", "f32", [capacity, hv, 128]),
                ("LogG", "GdnLogGate", "f32", [capacity, hv]),
                ("Count", "GdnLogCount", "i32", [hv, 128]),
            ]:
                name = f"L{layer}_{suffix}"
                w.fixed(name, dtype, shape, "sequence")
                b.reset.append(name)
                bindings[argument] = name
            b.emit(
                f"layer{layer}/private/recurrent_materialize",
                f"gdn-materialize-{hk}-{hv}-{capacity}",
                lambda: primitive(gdn_deferred.materialize, capacity, key_heads=hk, value_heads=hv),
                bindings,
                dynamic=False,
            )
    kernels = {k["name"]: k for k in b.aot.kernels}
    layers = []
    private = PrivateRows(w, ROW_BUFFERS, SCRATCH, state_role, LANES, len(b.text["layer_types"]))
    for layer, kind in enumerate(b.text["layer_types"]):
        entries = []
        names = (
            ["gates", "conv", "recurrent", "gated_norm"]
            if kind == "linear_attention"
            else ["prepare", "attention", "attention_merge"]
        )
        for role, suffix, config in [
            (role, suffix, config)
            for role in names
            for suffix, config in (
                b.row_choices(recurrent, LANES) if role == "recurrent" else [("", None)]
            )
        ]:
            old = kernels[f"qwen_rows/layer{layer}/private/{role}"]
            pointers = b.aot.compiler.pointers(old)
            if role == "attention":
                prepared = kernels[f"qwen_rows/layer{layer}/private/prepare"]
                pointers = attention_pointers(pointers, b.aot.compiler.pointers(prepared))
            if config and config.get("deferred"):
                pointers.update(
                    {
                        key: dict(kind="buffer", name=f"L{layer}_{value}")
                        for key, value in dict(
                            LogK="GdnLogKey",
                            LogD="GdnLogDelta",
                            LogG="GdnLogGate",
                            Count="GdnLogCount",
                        ).items()
                    }
                )
            if role == "conv":
                pointers["History"] = pointers["HI"]
            fn = factory(role, b.context, attention, b.text, config)
            modes, normalized = private.bind(fn, pointers)
            entries.append((role, suffix, config, fn, modes, normalized))
        layers.append(entries)
    offsets, cursor, columns = private.layout(first=("HistoryOut",))
    w.fixed(SCRATCH, "u8", [cursor], "sequence")
    b.state_columns = columns
    w.fixed(TABLE, "u64", [LANES * len(layers) * len(columns)])
    print("PRIVATE ROWS", len(columns), "columns", cursor, "scratch bytes/request", flush=True)
    for layer, entries in enumerate(layers):
        for role, suffix, config, fn, modes, pointers in entries:
            wrapped, bindings = private.wrap(
                fn,
                modes,
                pointers,
                layer,
                TABLE,
                batch_parameters=("independent_instances",)
                if role in ("attention", "attention_merge")
                else (),
            )
            key = f"private-rows-{role}-{b.h}-{b.hv}-{b.qh}-{attention if role.startswith('attention') else config}"

            def compile_wrapped(wrapped=wrapped, role=role, config=config):
                return compile_function(
                    wrapped,
                    pass_configs={
                        "tl.disable_thread_storage_sync": True,
                        "tl.disable_shared_memory_reuse": True,
                    }
                    if role == "recurrent" and config.get("prefetch_qk")
                    else {},
                )

            b.emit(f"state/layer{layer}/{role}{suffix}", key, compile_wrapped, bindings, LANES)
