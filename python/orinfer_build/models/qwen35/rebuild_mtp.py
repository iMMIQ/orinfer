"""Add fixed MTP operand tables while retaining a native Qwen package's measured compute."""

from orinfer_build.common.paths import work_path


import argparse
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import re
import types

from orinfer_build.common.files import write_json
from orinfer_build.packaging.publication import atomic_model, clone_model, load_model, AssetReader


def attention_geometry(kernel, source, dimension, head_group=None):
    call = re.search(
        r"= attn_split_count\([^,]+,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,[^;]*)?\);", source
    )
    independent = call is None
    if independent:
        split = re.search(r"split_count\[0\] = (\d+);", source)
        tile = re.search(r"void\* k = [^;]+buf_dyn_shmem \+ (\d+)", source)
        if split is None or tile is None or head_group is None:
            raise ValueError("Expected the registered Qwen attention split geometry")
        group, block_m, target, splits = (
            head_group,
            int(tile[1]) // (dimension * 2),
            16,
            int(split[1]),
        )
    else:
        group, block_m, target, splits = map(int, call.groups())
    if target != 16:
        raise ValueError("Unsupported attention split target")
    result = dict(
        block_m=block_m,
        target_tiles=target,
        max_splits=splits,
        head_group=group,
        independent_queries=independent,
    )
    k = re.search(r"void\* k = [^;]+buf_dyn_shmem \+ (\d+)", source)
    v = re.search(r"void\* v = [^;]+buf_dyn_shmem \+ (\d+)", source)
    if k is not None and v is not None:
        result.update(
            block_n=abs(int(v[1]) - int(k[1])) // (dimension * 2),
            threads=kernel["block"][0],
            stages=int("packed_k" in source),
        )
    return result


class ExistingWeights:
    def __init__(self, metadata, scopes):
        self.buffers = {b["name"]: copy.deepcopy(b) for b in metadata["buffers"]}
        self.scopes = dict(scopes)

    def fixed(self, name, dtype, shape, scope="workspace"):
        if name not in self.buffers:
            self.buffers[name] = dict(
                name=name,
                dtype=dtype,
                shape=shape,
                layout="native_contiguous",
                alignment=256,
                access="read_write",
            )
            self.scopes[name] = scope
        elif self.buffers[name]["dtype"] != dtype or self.buffers[name]["shape"] != shape:
            raise ValueError(f"MTP reconstruction changes the physical buffer {name}")
        return name

    def immutable(self, name, *, cast=None, shape=None):
        target = name if cast is None else f"Runtime.{name}.{cast}"
        if target not in self.buffers:
            raise ValueError(f"Missing registered immutable tensor {target}")
        return target

    def combined(self, name, inputs):
        if name not in self.buffers:
            raise ValueError(f"Missing registered joined tensor {name}")
        return name

    def constant(self, name, value):
        if name not in self.buffers:
            raise ValueError(f"Missing registered constant {name}")
        return name


def recipes(b, package, origin):
    """Reconstruct private arithmetic only; matrix implementations stay unchanged."""
    from orinfer_build.compiler.compile import primitive
    from kernels.cuda.model.gdn_sequence import _gdn_sequence
    from kernels.cuda.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.cuda.model.attention_rows import causal_attention_int8, merge_attention_int8
    from orinfer_build.models.qwen35 import mtp, gdn
    from orinfer_build.contracts.abi import evaluate

    originals = {k["name"]: k for k in package["kernels"]}
    dynamic = {d["name"]: d for d in package["dynamic_batch_kernels"]}
    reader = AssetReader(origin)
    aot = b.aot
    actual_emit = aot.emit

    def record(
        name, key, make, bindings, rows=4096, *, dynamic=True, row_divisor=1, row_multiplier=1
    ):
        if name in originals:
            aot.recipes[name] = (key, make, bindings, rows, dynamic, row_divisor, row_multiplier)
        return dict(kind="kernel", name=name)

    aot.emit = record
    b.projection = lambda *args, **kwargs: None
    b.rotation = lambda *args, **kwargs: None
    b.norm_rotation = lambda *args, **kwargs: None
    b.row_tuner = types.SimpleNamespace(
        gdn_sequence=lambda: dict(configs=[dict(token_unroll=1)], default="0", selections=[]),
        gdn_ab=lambda: dict(configs=[{}], default="0", selections=[]),
    )

    def convolution(self, name, bindings):
        prefix = "qwen_rows/" + name
        for actual, kernel in originals.items():
            if actual != prefix and not actual.startswith(prefix + "/rows/"):
                continue
            host = aot.compiler.host(kernel)
            expression = host["launch_expressions"]["gridDimX"]

            def grid(rows):
                return evaluate(expression, dict(m=rows, M=rows, rows=rows, batch=rows))

            first = grid(1)
            tile = next((n - 1 for n in range(2, 129) if grid(n) != first), None)
            if tile is None:
                raise ValueError("Unrecognized causal convolution token tile")

            def make(tile=tile):
                return primitive(
                    gdn_conv_prep,
                    1,
                    4096,
                    tile_tokens=tile,
                    dynamic_tokens=True,
                    key_heads=self.hk,
                    value_heads=self.hv,
                )

            record(
                actual,
                "retained-conv-" + kernel["module"]["sha256"],
                make,
                bindings,
                dynamic.get(actual, {}).get("capacity", 4096),
            )

    b.convolution = types.MethodType(convolution, b)

    def attention(self, name, **kwargs):
        prefix = "qwen_rows/" + name
        for actual, kernel in originals.items():
            if actual not in (prefix, prefix + "_merge") and not actual.startswith(
                (prefix + "/rows/", prefix + "_merge/rows/")
            ):
                continue
            merge = actual.startswith(prefix + "_merge")
            reference = originals[actual.replace(prefix + "_merge", prefix, 1)] if merge else kernel
            config = attention_geometry(
                reference,
                reader.read(reference["source"]).decode(),
                self.dimension,
                self.qh // self.kvh,
            )
            if merge:
                config = {
                    k: v for k, v in config.items() if k not in ("block_n", "threads", "stages")
                }
            fn = merge_attention_int8 if merge else causal_attention_int8
            config.update(
                query_heads=self.qh, kv_heads=self.kvh, dimension=self.dimension, gated=True
            )
            if not merge:
                config["context"] = self.context
            capacity = dynamic.get(actual, {}).get("capacity", kwargs.get("rows", 4096))

            def make(fn=fn, config=config, capacity=capacity):
                return primitive(fn, capacity, **config)

            record(actual, "retained-attention-" + kernel["module"]["sha256"], make, {}, capacity)

    b.attention = types.MethodType(attention, b)
    saved_prefill = gdn.prefill
    try:
        # The published recurrent construction is retained, including its arithmetic.
        # Source-only chunked-prefill candidates do not participate in this rebuild.
        gdn.prefill = lambda *args, **kwargs: None
        b.forward()
        b.head()
        mtp.build(b)
        if b.vision:
            from kernels.cuda.vision.features import overlay

            b.emit(
                "begin/overlay",
                f"vision-overlay-{b.context}",
                lambda: primitive(
                    overlay, 4096, b.h, b.context, b.vision["max_features"], dynamic_rows=True
                ),
                dict(
                    Embedding="Hidden", Features="Features", Index="FeatureIndex", Position="Step"
                ),
            )
    finally:
        gdn.prefill = saved_prefill
        aot.emit = actual_emit
    # Add selected sequential recurrence aliases with their original grid/ABI.
    for name, kernel in originals.items():
        if re.fullmatch(r"qwen_rows/layer\d+/private/recurrent(?:/rows/\d+(?:-\d+)?)?", name):

            def make():
                return primitive(
                    _gdn_sequence,
                    4096,
                    in_place=True,
                    value_tile=16,
                    threads=32,
                    prefetch_qk=True,
                    dynamic_rows=True,
                    key_heads=b.hk,
                    value_heads=b.hv,
                    token_unroll=1,
                )

            aot.recipes[name] = (
                "retained-recurrent-" + kernel["module"]["sha256"],
                make,
                {},
                dynamic.get(name, {}).get("capacity", 4096),
                True,
                1,
                1,
            )


def rebuild(args):
    from orinfer_build.models.qwen35.build import Builder, configuration
    from orinfer_build.models.qwen35.compiler import Aot
    from orinfer_build.compiler.host_compiler import Compiler
    from orinfer_build.models.qwen35 import mtp_packets
    from orinfer_build.packaging.compact import compact_package

    manifest = {}
    data, origin, package = load_model(args.model, plan=manifest)
    if data["architecture"] != "qwen3_5" or manifest.get("mtp", {}).get("packet_table"):
        raise ValueError("Expected a native Qwen MTP package without an operand table")
    if manifest.get("gdn_execution") != {"prefill": "sequential", "decode": {"kind": "dense"}}:
        raise ValueError("Incremental rebuild requires the registered dense sequential GDN body")
    config = json.loads((args.model / "config.json").read_text())
    text = configuration(config, manifest["max_context"])
    if config["model_type"] != "qwen3_5":
        raise ValueError("Incremental MTP rebuild currently supports dense Qwen")
    with atomic_model(args.target) as stage:
        operator = clone_model(args.model, stage, origin)
        b = Builder.__new__(Builder)
        b.config, b.text = config, text
        b.standard, b.moe, b.lock = False, False, None
        b.context = manifest["max_context"]
        b.h, b.f, b.v = text["hidden_size"], text["intermediate_size"], text["vocab_size"]
        b.hk, b.hv = text["linear_num_key_heads"], text["linear_num_value_heads"]
        b.qh, b.kvh, b.dimension = (
            text["num_attention_heads"],
            text["num_key_value_heads"],
            text["head_dim"],
        )
        b.qkv, b.mixdim = (2 * b.hk + b.hv) * 128, b.hv * 128
        b.qproj, b.kv_width = 2 * b.qh * b.dimension, b.kvh * b.dimension
        b.fullx = b.qproj + 2 * b.kv_width
        b.embedding_q4 = config["quantization_config"].get("embedding") == "e8p_q4"
        b.tied = text.get("tie_word_embeddings", False)
        b.head_weight = "model.language_model.embed_tokens.weight" if b.tied else "lm_head.weight"
        b.weights = ExistingWeights(manifest, data["buffer_scopes"])
        b.gdn_execution = copy.deepcopy(manifest["gdn_execution"])
        b.vision, b.audio = manifest.get("vision"), None
        b.reset = []
        b.kv = dict(manifest["kv_cache"]["buffers"])
        b.signs = {
            int(name.split(".")[-1]): name
            for name in b.weights.buffers
            if name.startswith("RotationSigns.")
        }
        b.book = "orinfer.e8p_q4.selectors"
        b.aot = Aot(operator, args.output, args.cache, b.weights.buffers, compile_lock=nullcontext)
        b.aot.identity["source_execution_package"] = data["execution_package"]
        b.aot.compiler = Compiler(origin, operator, args.output, b.weights.buffers)
        b.aot.kernels = copy.deepcopy(package["kernels"])
        b.aot.dynamic = copy.deepcopy(package["dynamic_batch_kernels"])
        recipes(b, package, origin)
        b.mtp = copy.deepcopy(data["metadata"]["mtp"])
        mtp_packets.build(b)
        data["metadata"]["mtp"] = b.mtp
        data["metadata"]["buffers"] = list(b.weights.buffers.values())
        data["buffer_scopes"] = b.weights.scopes
        package["kernels"], package["dynamic_batch_kernels"] = b.aot.kernels, b.aot.dynamic
        package["buffer_contracts"] = [
            {k: v for k, v in buffer.items() if k != "data"}
            for buffer in b.weights.buffers.values()
        ]
        package["toolchain"]["mtp_packet_source"] = b.aot.identity["mtp_packet_source"]
        package["toolchain"]["mtp_packet_reference"] = b.aot.identity["source_execution_package"]
        data["execution_package"] = compact_package(operator, package, stage / "cache/packages")
        write_json(stage / "cache/model.json", data)
    write_json(
        args.output / "result.json",
        dict(model=str(args.target), execution_package=data["execution_package"], complete=True),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=work_path("cache", "build", "qwen35-mtp"))
    rebuild(p.parse_args())


if __name__ == "__main__":
    main()
