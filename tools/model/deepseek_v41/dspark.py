"""Build DSpark on the shared DSv4.1 A2 projections, attention and owner rows.

Target weights remain shared. Draft W4 banks are prepared offline; the target
captures HC means and seeds a bounded INT8 draft window in the same packet.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from contextlib import nullcontext
import json
import multiprocessing
from pathlib import Path

import numpy as np

from tools.build.files import atomic_json

BLOCK = 5
LAYERS = (37, 38, 39)


def builder_class():
    # TileLang/Torch belong to the offline Ascend compiler environment.
    from tools.model.deepseek_v41.build import Builder, Strategy
    from tools.operators.ascend import attention_cube, kv_int8, memory, model_ops, topk, vector

    class Draft(Builder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.expert_count, self.experts_per_token = 128, 3
            self.layer_namespace, self.layer_offset = "mtp", 40
            self.layer_ids = tuple(range(40, 43))
            self.final_norm_name = "mtp.2.norm.weight"
            self.all_head_rows = True
            self.compressed_coordinates = False

        def matrix(self, name, dtype, *args, **kwargs):
            # The target is in the QuaRot basis; the draft retains its own
            # unrotated vocabulary projection and embedding.
            if name == "head.weight":
                name, dtype, kwargs["float32"] = "mtp.2.head.weight", "bf16", True
            # The Ascend DSpark checkpoint retains BF16 norm parameters;
            # the shared norm arithmetic consumes their exact FP32 expansion.
            if name.startswith("mtp.") and name.endswith("norm.weight") and dtype == "f32":
                dtype, kwargs["float32"] = "bf16", True
            return super().matrix(name, dtype, *args, **kwargs)

        def store(self, hidden, layer, cosine, sine, slots, *, noise=False):
            prefix = self.layer_name(layer) + ".attn"
            a8, scales = self.quant(hidden, 5120, "dspark.kv.input")
            kv = self.w8(a8, scales, prefix + ".wkv", 512, 5120, "dspark.kv")
            kv, _ = self.norm(
                kv, self.matrix(prefix + ".kv_norm.weight", "f32", 512, 1), 512, "dspark.kv.norm"
            )
            kv = self.rope(kv, 1, 512, cosine, sine, "dspark.kv.rope")
            label = f"cache.dspark.{'noise' if noise else 'swa'}.{layer}"
            cache = self.buffer(label, (128, 8, 64), "i8")
            scale = self.buffer(label + ".scale", (128, 8), "f16")
            errors = self.temp("dspark.kv.errors", (self.rows,), "i32")
            self.kernel(
                kv_int8,
                "scatter",
                dict(
                    rows=self.rows,
                    table_rows=128,
                    logical_capacity=0 if noise else self.capacity,
                    retain_tail=not noise,
                ),
                [
                    ("source", kv, "bf16", (self.rows, 8, 64), "read"),
                    ("indices", slots, "i32", (self.rows,), "read"),
                    ("cache", cache, "i8", (128, 8, 64), "read_write"),
                    ("scales", scale, "f16", (128, 8), "read_write"),
                    ("errors", errors, "i32", (self.rows,), "write"),
                ],
            )
            return cache, scale

        def attention(self, x, layer, coordinates, reset):
            # The draft block sees confirmed history plus every noise row.
            rows, count = self.rows, 256
            cosine, sine = coordinates["base_cos"], coordinates["base_sin"]
            prefix = self.layer_name(layer) + ".attn"
            noise, noise_scale = self.store(
                x, layer, cosine, sine, coordinates["c2_slots"], noise=True
            )
            q8, qs = self.quant(x, 5120)
            qa = self.w8(q8, qs, prefix + ".wq_a", 1280, 5120, "q.a")
            qr, qrs = self.norm_quant(
                qa, self.matrix(prefix + ".q_norm.weight", "f32", 1280, 1), 1280, "q.norm.quant"
            )
            query = self.w8(qr, qrs, prefix + ".wq_b", 4096, 1280, "q.b", first=self.rank * 4096)
            query = self.rope(query, 8, 512, cosine, sine, "q.rope")
            swa = self.buffer(f"cache.dspark.swa.{layer}", (128, 8, 64), "i8")
            scale = self.buffer(swa + ".scale", (128, 8), "f16")
            gathered = self.temp("dspark.attention.gathered", (rows, count, 512))
            mask = self.temp("dspark.attention.mask", (rows, count), "i32")
            self.kernel(
                kv_int8,
                "gather_attention",
                dict(
                    rows=rows,
                    capacity=128,
                    compressed_capacity=128,
                    long_selected=128,
                    tile_slots=8,
                    bulk_store=True,
                ),
                [
                    ("swa", swa, "i8", (128, 8, 64), "read"),
                    ("swa_scales", scale, "f16", (128, 8), "read"),
                    ("long_kv", noise, "i8", (128, 8, 64), "read"),
                    ("long_scales", noise_scale, "f16", (128, 8), "read"),
                    ("swa_indices", coordinates["swa_indices"], "i32", (rows, 128), "read"),
                    ("long_indices", coordinates["c1_indices"], "i32", (rows, 128), "read"),
                    ("out", gathered, "bf16", (rows, count, 512), "write"),
                    ("mask", mask, "i32", (rows, count), "write"),
                ],
            )
            sink = self.matrix(prefix + ".attn_sink", "f32", 8, 1, first_row=self.rank * 8)
            score = self.temp("dspark.attention.score", (rows, 16, count), "f32")
            prob = self.temp("dspark.attention.prob", (rows, 16, count))
            attended = self.temp("attention.output", (rows, 8, 512))
            scratch = self.temp("dspark.attention.workspace", (rows, 16, 512), "f32")
            p = dict(rows=rows, heads=8, selected=count)
            self.kernel(
                attention_cube,
                "qk",
                p,
                [
                    ("query", query, "bf16", (rows, 8, 512), "read"),
                    ("kv", gathered, "bf16", (rows, count, 512), "read"),
                    ("scores", score, "f32", (rows, 16, count), "write"),
                ],
            )
            self.kernel(
                attention_cube,
                "softmax",
                p,
                [
                    ("scores", score, "f32", (rows, 16, count), "read"),
                    ("indices", mask, "i32", (rows, count), "read"),
                    ("sink", sink, "f32", (8,), "read"),
                    ("probabilities", prob, "bf16", (rows, 16, count), "write"),
                ],
            )
            self.kernel(
                attention_cube,
                "pv",
                p,
                [
                    ("probabilities", prob, "bf16", (rows, 16, count), "read"),
                    ("kv", gathered, "bf16", (rows, count, 512), "read"),
                    ("output", attended, "bf16", (rows, 8, 512), "write"),
                    ("workspace", scratch, "f32", (rows, 16, 512), "scratch"),
                ],
            )
            return self.attention_output(attended, layer, cosine, sine)

        def head_outputs(self, logits, rows, hidden):
            owners = rows // BLOCK
            self.rows = owners
            axis = dict(
                op="divide",
                lhs=dict(op="add", lhs=dict(op="rows"), rhs=dict(op="constant", value=BLOCK - 1)),
                rhs=dict(op="constant", value=BLOCK),
            )
            begin = len(self.steps)
            ids = self.buffer("draft.input_ids", (rows,), "i32")
            previous = self.temp("markov.previous", (owners,), "i32")
            tokens = self.temp("markov.tokens", (rows,), "i32")
            values = self.temp("markov.values", (rows,), "f32")
            confidence = self.temp("markov.confidence", (rows,), "f32")
            self.kernel(
                model_ops,
                "block_first",
                dict(rows=owners),
                [
                    ("source", ids, "i32", (rows,), "read"),
                    ("out", previous, "i32", (owners,), "write"),
                ],
            )
            embedding = self.temp("markov.embedding", (owners, 256))
            errors = self.temp("markov.errors", (owners,), "i32")
            table = self.matrix("mtp.2.markov_head.embed.weight", "bf16", 129280, 256)
            weight = self.matrix(
                "mtp.2.markov_head.head.weight",
                "bf16",
                16160,
                256,
                first_row=self.rank * 16160,
                float32=True,
            )
            cw = self.matrix("mtp.2.confidence_head.proj.weight", "bf16", 1, 5376, float32=True)
            proposals = self.temp("markov.proposals", (rows, 16160), "f32")
            parts = 8
            for step in range(BLOCK):
                self.kernel(
                    memory,
                    "gather_rows",
                    dict(rows=owners, table_rows=129280, width=256, index_dtype="int32"),
                    [
                        ("table", table, "bf16", (129280, 256), "read"),
                        ("indices", previous, "i32", (owners,), "read"),
                        ("out", embedding, "bf16", (owners, 256), "write"),
                        ("errors", errors, "i32", (owners,), "write"),
                    ],
                )
                bias = self.fp32(embedding, weight, 16160, 256, "markov.bias")
                packed = self.temp("markov.packed", (parts, owners, 2), "f32")
                self.kernel(
                    topk,
                    "add_bias_argmax",
                    dict(rows=owners, vocab=16160, step=step),
                    [
                        ("base", logits, "f32", (rows, 16160), "read"),
                        ("bias", bias, "f32", (owners, 16160), "read"),
                        ("proposals", proposals, "f32", (rows, 16160), "read_write"),
                        ("packed", packed, "f32", (parts, owners, 2), "write"),
                    ],
                )
                self.kernel(
                    vector,
                    "confidence",
                    dict(rows=owners, step=step),
                    [
                        ("hidden", hidden, "bf16", (rows, 5120), "read"),
                        ("embedding", embedding, "bf16", (owners, 256), "read"),
                        ("weight", cw, "f32", (5376,), "read"),
                        ("probabilities", confidence, "f32", (rows,), "read_write"),
                    ],
                )
                all_ranks = self.temp("markov.all_ranks", (8 * parts, owners, 2), "f32")
                self.steps.append(
                    dict(
                        op="gather",
                        source=packed,
                        destination=all_ranks,
                        elements=owners * 2 * parts,
                        fp32=True,
                        elements_expression=dict(
                            op="multiply", lhs=axis, rhs=dict(op="constant", value=2 * parts)
                        ),
                    )
                )
                self.kernel(
                    topk,
                    "reduce_rank_best",
                    dict(
                        rows=owners, step=step, ranks=8 * parts, rank_parts=parts, rank_vocab=16160
                    ),
                    [
                        ("packed", all_ranks, "f32", (8 * parts, owners, 2), "read"),
                        ("previous", previous, "i32", (owners,), "read_write"),
                        ("tokens", tokens, "i32", (rows,), "read_write"),
                        ("values", values, "f32", (rows,), "read_write"),
                    ],
                )
            for instruction in self.steps[begin:]:
                if instruction["op"] == "kernel":
                    instruction["row_expression"] = axis
            self.rows = rows
            self.draft_output = dict(input_ids=ids, confidence=confidence, logits=proposals)
            return tokens, values

    return Draft, Strategy


def compile_source(model, builders, tasks, root, name, workers, owners, rows, compiler=None):
    from tools.model.deepseek_v41 import batch
    from tools.model.deepseek_v41.build import compile_task

    with (
        nullcontext(compiler)
        if compiler is not None
        else ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
    ) as pool:
        paths = dict(pool.map(compile_task, ((t, root) for t in tasks.values())))
    for rank, builder in zip(model["ranks"], builders, strict=True):
        rank["buffers"] = builder.buffers
        for packet in rank["packets"]:
            for instruction in packet["body"] + packet["head"]:
                if instruction["op"] == "kernel":
                    instruction["manifest"] = paths[instruction["manifest"]]
    source, output = root / (name + "-raw.json"), root / (name + "-rows.json")
    atomic_json(source, model)
    result = batch.build(
        source, output, owners, model["capacity"], rows, workers, compiler=compiler
    )
    if name == "draft":
        for rank in result["ranks"]:
            packet = rank["packets"][0]
            for key in ("best_index", "best_value"):
                packet[key] = packet[key].removesuffix(".compact")
            packet["confidence"] = "batch.prefill." + packet["confidence"]
            packet["vocab_first"] = 0
    return result


def rename_contract(model, prefix, role):
    """Keep local pointer ABIs intact while appending their global column IDs."""

    def rename(value):
        if isinstance(value, str):
            return prefix + value
        if isinstance(value, list):
            return [rename(v) for v in value]
        return value

    for rank in model["ranks"]:
        names = {b["name"]: prefix + b["name"] for b in rank["buffers"]}

        def bind(value):
            if isinstance(value, str):
                return names.get(value, value)
            if isinstance(value, list):
                return [bind(v) for v in value]
            if isinstance(value, dict):
                return {k: bind(v) for k, v in value.items()}
            return value

        rank["buffers"] = [bind(b) for b in rank["buffers"]]
        for packet in rank["packets"]:
            for part in (
                "body",
                "head",
                "coordinates",
                "decoder_coordinates",
                "best_value",
                "confidence",
                "best_index",
                "logits",
                "reset",
            ):
                if part in packet:
                    packet[part] = bind(packet[part])
            packet["role"] = role
    contract = model["batch"]
    for column in contract["columns"]:
        column["buffer"] = prefix + column["buffer"]
        column["role"] = role
    contract["private"] = rename(contract["private"])
    for key in ("pointers", "work", "work_ranges", "partitioned_tables"):
        contract[key] = {prefix + k: v for k, v in contract.get(key, {}).items()}
    for key in ("active", "endpoints"):
        contract[key] = prefix + contract[key]
    return model


def build(base_plan, checkpoint, output, workers, base_model=None):
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as compiler:
        return _build(base_plan, checkpoint, output, workers, compiler, base_model)


def _build(base_plan, checkpoint, output, workers, compiler, base_model):
    import os
    from tools.model.deepseek_v41 import prepare
    from tools.model.deepseek_v41.assembly import validate_rows
    from tools.operators.ascend import vector

    root = output.resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    for path in base_plan.parent.rglob("*"):
        if path.is_file() and path.name != "model.json":
            dest = root / path.relative_to(base_plan.parent)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(path, dest)
    target = json.loads(base_plan.read_text())
    owners = target["batch"]["owners"]
    Draft, Strategy = builder_class()
    strategy = Strategy(
        shared_tp=True,
        cube_attention=True,
        parallel=True,
        fusion=True,
        prefill=True,
        parallel_regroup=True,
        batched_rope=True,
        active_expert_rows=True,
        vector_prefix=True,
    )
    templates = {}
    for name, rows in (
        ("warm_decode", owners),
        ("warm", target["ranks"][0]["packets"][1]["rows"]),
        ("draft", owners * BLOCK),
    ):
        model = {
            k: deepcopy(v)
            for k, v in target.items()
            if k not in ("ranks", "batch", "row_selections")
        }
        model["index"] = dict(file="quant_model_weights.safetensors.index.json", sha256="0" * 64)
        model["capacity"] = 128 if name == "draft" else target["capacity"]
        model["ranks"], builders, kernels = [], [], {}
        for rank in range(8):
            b = Draft(
                root, rank, model["capacity"], kernels, strategy=strategy, model_dir=checkpoint
            )
            if name == "draft":
                packet = b.packet(rows, dynamic=True)
                for instruction in packet["body"]:
                    if instruction["op"] == "embed":
                        instruction["tensor"] = "mtp.0.embed.weight"
                # A draft is one common body, including the Markov chain.
                packet["body"].extend(packet["head"])
                packet["head"] = []
                packet["coordinates"].update(
                    draft_block=BLOCK, input_ids=b.draft_output["input_ids"]
                )
                packet.update({k: v for k, v in b.draft_output.items() if k != "input_ids"})
                packet["replicated_output"] = True
            else:
                b.rows, b.prefix, b.dynamic = rows, f"m{rows}.", True
                coordinates = {}
                for key in (
                    "base_cos",
                    "base_sin",
                    "long_cos",
                    "long_sin",
                    "source_cos",
                    "source_sin",
                ):
                    coordinates[key] = b.temp(key, (rows, 32), "f32")
                for key, shape in (
                    ("slots", (rows,)),
                    ("c2_slots", (rows,)),
                    ("swa_indices", (rows, 128)),
                    ("c1_indices", (rows, 1)),
                    ("c2_indices", (rows, 1)),
                    ("c1_lengths", (rows,)),
                    ("c2_lengths", (rows,)),
                    ("ring", (rows, 8)),
                ):
                    coordinates[key] = b.temp(key, shape, "i32")
                coordinates["premix"] = b.temp("premix", (rows, 8), "f32")
                features = b.temp("features", (rows, 3, 5120))
                for section, layer in enumerate(LAYERS):
                    residual = b.temp(f"capture.{layer}", (rows, 4, 5120))
                    b.kernel(
                        vector,
                        "mean_sections",
                        dict(rows=rows, section=section),
                        [
                            ("residual", residual, "bf16", (rows, 4, 5120), "read"),
                            ("features", features, "bf16", (rows, 3, 5120), "read_write"),
                        ],
                    )
                main = b.projection(
                    features,
                    b.matrix("mtp.0.main_proj.weight", "bf16", 5120, 15360),
                    5120,
                    15360,
                    "main.proj",
                )
                main, _ = b.norm(
                    main,
                    b.matrix("mtp.0.main_norm.weight", "bf16", 5120, 1, float32=True),
                    5120,
                    "main.norm",
                )
                for layer in b.layer_ids:
                    b.store(
                        main,
                        layer,
                        coordinates["base_cos"],
                        coordinates["base_sin"],
                        coordinates["slots"],
                    )
                dummy_i, dummy_v = (
                    b.temp("dummy.index", (1,), "i32"),
                    b.temp("dummy.value", (1,), "f32"),
                )
                packet = dict(
                    rows=rows,
                    min_rows=2,
                    coordinates=coordinates,
                    body=b.steps,
                    head=[],
                    reset=[],
                    best_index=dummy_i,
                    best_value=dummy_v,
                    logits=dummy_v,
                    vocab_first=0,
                )
            model["ranks"].append(dict(rank=rank, device=rank, packets=[packet], buffers=[]))
            builders.append(b)
        templates[name] = compile_source(
            model, builders, kernels, root, name, workers, owners, rows, compiler
        )
        print(f"DSpark {name} shared rows ready", flush=True)
    # W4/INT8 shared banks use the existing bounded checkpoint converter.
    for name, model in templates.items():
        raw, prepared = root / (name + "-weights.json"), root / (name + "-prepared.json")
        atomic_json(raw, model)
        prepare.prepare(raw, checkpoint, prepared, workers)
        templates[name] = json.loads(prepared.read_text())
    prepack(templates["draft"], root)
    merge(target, templates, root, workers, compiler)

    validate_rows(target, root, model_dir=base_model)
    atomic_json(output, target)


def prepack(model, root):
    from tools.model.deepseek_v41.weight_layout import convert

    for rank in model["ranks"]:
        buffers = {b["name"]: b for b in rank["buffers"]}
        for packet in rank["packets"]:
            for instruction in packet["body"]:
                if instruction["op"] != "w4" or instruction.get("weight_layout") == "transposed_nz":
                    continue
                definition = buffers[instruction["buffers"][1]]
                source = root / definition["prepared_weight"]
                output = source.with_name(source.stem + "-nz.bin")
                if not output.exists():
                    convert(
                        source,
                        output,
                        instruction["experts"],
                        instruction["input"],
                        instruction["output"],
                    )
                if output.stat().st_size != definition["bytes"]:
                    raise ValueError("DSpark NZ bank extent differs")
                definition["prepared_weight"] = str(output.relative_to(root))
                instruction["weight_layout"] = "transposed_nz"
                buffers[instruction["workspace"]]["bytes"] = 320 << 20


def compose(base_plan, output, workers, base_model=None):
    from tools.model.deepseek_v41.assembly import validate_rows

    root = output.parent
    target = json.loads(base_plan.read_text())
    templates = {
        name: json.loads((root / (name + "-prepared.json")).read_text())
        for name in ("warm_decode", "warm", "draft")
    }
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as compiler:
        prepack(templates["draft"], root)
        merge(target, templates, root, workers, compiler)
        atomic_json(output.with_name("dspark-candidate.json"), target)
        validate_rows(target, root, model_dir=base_model)
        atomic_json(output, target)


def merge(target, templates, root, workers, compiler):

    contract = target["batch"]
    persistent = set()
    for name, auxiliary in templates.items():
        role = "draft" if name == "draft" else "target"
        rename_contract(auxiliary, f"dspark.{name}.", role)
        other = auxiliary["batch"]
        offset = len(contract["columns"])

        # Fixed SWA windows are shared across warm and draft programs. Only
        # their owner address-table columns are program local.
        def alias(key):
            prefix = f"dspark.{name}."
            plain = key.removeprefix(prefix)
            return plain if plain.startswith("cache.dspark.swa.") else key

        for column in other["columns"]:
            column["buffer"] = alias(column["buffer"])
            if name.startswith("warm"):
                column["decode"] = (
                    None if column.get("row_bytes") is None else name == "warm_decode"
                )
                column["decoder"] = name == "warm" and column.get("row_bytes") is not None
        contract["columns"].extend(other["columns"])
        contract["private"].extend(
            alias(n) for n in other["private"] if alias(n) not in contract["private"]
        )
        for key in ("work", "work_ranges", "partitioned_tables"):
            contract.setdefault(key, {}).update(other.get(key, {}))
        contract["pointers"].update(
            {key: [c + offset for c in mapping] for key, mapping in other["pointers"].items()}
        )
        if name == "warm":
            contract["decoder_replay"]["tables"].extend(other["pointers"])
            contract["decoder_replay"]["tables"].extend(other["work"])
        for rank, extra in zip(target["ranks"], auxiliary["ranks"], strict=True):
            existing = {b["name"]: b for b in rank["buffers"]}
            weights = {
                json.dumps(b.get("weight"), sort_keys=True): b["name"]
                for b in rank["buffers"]
                if b.get("weight")
            }
            names = {}
            for definition in extra["buffers"]:
                old = definition["name"]
                new = alias(old)
                if old == other["active"]:
                    new = contract["active"]
                if definition.get("weight"):
                    new = weights.get(json.dumps(definition["weight"], sort_keys=True), new)
                    original_name = old.removeprefix(f"dspark.{name}.")
                    if (
                        definition["weight"].get("name") == "head.weight"
                        and original_name in existing
                    ):
                        new = original_name
                names[old] = new
                if new not in existing:
                    definition["name"] = new
                    rank["buffers"].append(definition)
                    existing[new] = definition
                if new.startswith("cache.dspark.swa."):
                    persistent.add(new)

            def bind(value):
                if isinstance(value, str):
                    return names.get(value, value)
                if isinstance(value, list):
                    return [bind(v) for v in value]
                if isinstance(value, dict):
                    return {k: bind(v) for k, v in value.items()}
                return value

            for definition in extra["buffers"]:
                if "view" in definition:
                    definition["view"] = bind(definition["view"])
            packet = bind(extra["packets"][0])
            if role == "draft":
                rank["packets"].append(packet)
                continue
            captures, seed = packet["body"][:3], packet["body"][3:]
            for target_packet in rank["packets"]:
                if target_packet.get("role", "target") != "target" or target_packet.get(
                    "batch_decode", False
                ) != (name == "warm_decode"):
                    continue
                coordinates = target_packet.get("decoder_coordinates", target_packet["coordinates"])
                coordinate_names = {
                    value: coordinates[key]
                    for key, value in packet["coordinates"].items()
                    if isinstance(value, str) and key in coordinates
                }
                body = []
                for instruction in target_packet["body"]:
                    body.append(instruction)
                    if instruction.get("name") in ("layer36", "layer37", "layer38"):
                        section = int(instruction["name"][-2:]) - 36
                        capture = deepcopy(captures[section])
                        capture["buffers"][0] = instruction["buffer"]
                        body.append(capture)
                for instruction in seed:
                    item = deepcopy(instruction)
                    if item["op"] == "kernel":
                        item["buffers"] = [coordinate_names.get(n, n) for n in item["buffers"]]
                    body.append(item)
                target_packet["body"] = body
            # Coordinate columns must follow the target decoder coordinates.
            for column in contract["columns"][offset:]:
                if column["buffer"] in coordinate_names:
                    column["buffer"] = coordinate_names[column["buffer"]]
        # Every rank has the same coordinate names and column order.
    for name in sorted(persistent):
        layout = dict(
            buffer=name,
            row_bytes=16 if name.endswith(".scale") else 512,
            tokens_per_row=1,
            window=128,
            circular=True,
        )
        contract["decoder_replay"].setdefault("decoder_state", []).append(deepcopy(layout))
    contract["verification_windows"] = {
        f"cache.ring.{layer}": dict(offset=32 * 1024 * 4, row_bytes=1024 * 4, rows=32, history=1)
        for layer in (2, 8, 14)
    }
    converge_captures(target, root, workers, compiler)
    expand_head(target, root, workers, compiler)


def converge_captures(model, root, workers, compiler=None):
    """Gather HC means from the same contiguous SP rows as target projections."""
    from tools.model.deepseek_v41.build import compile_task, kernel_rebuild_task
    from tools.model.row_contract import expression

    tasks = {}
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}
        for packet in rank["packets"]:
            if packet.get("role", "target") != "target":
                continue
            body, features, local_features = packet["body"], None, None
            for index, step in enumerate(body):
                if step.get("name") not in ("layer36", "layer37", "layer38") or index == 0:
                    continue
                transport = body[index - 1]
                if transport.get("op") != "gather" or not transport.get("trace_only"):
                    continue
                if transport["destination"] != step["buffer"]:
                    raise ValueError("HC snapshot does not describe its SP source")
                capture = body[index + 1]
                features = capture["buffers"][1]
                local_features = features + ".sequence"
                if local_features not in definitions:
                    definition = dict(
                        name=local_features, bytes=definitions[features]["bytes"] // 8
                    )
                    definitions[local_features] = definition
                    rank["buffers"].append(definition)
                spec = json.loads((root / capture["manifest"]).read_text())
                key = "@capture" + str(spec["parameters"]["section"])
                if key not in tasks:
                    task = kernel_rebuild_task(spec, key)
                    task["arguments"]["rows"] = packet["rows"] // 8
                    task["row_capacity"] = packet["rows"] // 8
                    for operand in task["specs"]:
                        operand["shape"][0] //= 8
                    tasks[key] = task
                capture["manifest"] = key
                capture["buffers"] = [transport["source"], local_features]
                capture["row_expression"] = expression("(m + 7) // 8")
            if local_features is not None:
                last = max(i for i, step in enumerate(body) if step.get("name") == "layer38") + 2
                body.insert(
                    last,
                    dict(
                        op="gather",
                        source=local_features,
                        destination=features,
                        elements=packet["rows"] // 8 * 15360,
                        fp32=False,
                        elements_expression=expression("((m + 7) // 8) * 15360"),
                    ),
                )
    if tasks:
        with (
            nullcontext(compiler)
            if compiler is not None
            else ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
        ) as pool:
            paths = dict(pool.map(compile_task, ((t, root) for t in tasks.values())))
        for rank in model["ranks"]:
            for packet in rank["packets"]:
                for step in packet["body"]:
                    if step.get("manifest") in paths:
                        step["manifest"] = paths[step["manifest"]]


def expand_head(model, root, workers, compiler=None):
    from tools.build.files import file_hash
    from tools.model.deepseek_v41.build import compile_task, kernel_rebuild_task

    owners = model["batch"]["owners"]
    capacity = owners * (BLOCK + 1)
    model["batch"]["output_capacity"] = capacity
    tasks = {}
    for rank in model["ranks"]:
        buffers = {b["name"]: b for b in rank["buffers"]}
        resized = {model["batch"]["endpoints"]}
        buffers[model["batch"]["endpoints"]]["bytes"] = capacity * 4
        views = {
            name: name + ".decode" for name in (model["batch"]["endpoints"], "batch.head.errors")
        }
        for parent, view in views.items():
            rank["buffers"].append(
                dict(name=view, bytes=owners * 4, view=dict(parent=parent, offset=0))
            )
        for packet in rank["packets"]:
            if packet.get("batch_decode"):
                for instruction in packet["head"]:
                    if "buffers" in instruction:
                        instruction["buffers"] = [views.get(n, n) for n in instruction["buffers"]]
        for packet in rank["packets"]:
            if packet.get("role", "target") != "target" or packet.get("batch_decode"):
                continue
            for instruction in packet["head"]:
                if instruction["op"] != "kernel" or not instruction.get("owners"):
                    continue
                spec = json.loads((root / instruction["manifest"]).read_text())
                task = kernel_rebuild_task(spec, "@head" + str(len(tasks)))
                task["arguments"]["rows"] = capacity
                task["row_capacity"] = capacity
                for operand, name in zip(task["specs"], instruction["buffers"], strict=True):
                    definition = buffers[name]
                    if operand["shape"][0] != owners or definition.get("weight"):
                        continue
                    operand["shape"][0] = capacity
                    if definition.get("prepared_tensor"):
                        if spec["op"] != "topk.local" or operand["name"] != "lengths":
                            raise ValueError("Cannot resize an immutable head matrix")
                        path = root / "data" / (name + "-dspark-constant.bin")
                        path.write_bytes(np.full(capacity, 16160, dtype="<i4").tobytes())
                        definition.pop("prepared_tensor")
                        definition["data"] = dict(
                            file=str(path.relative_to(root)), sha256=file_hash(path)
                        )
                    if name in resized:
                        continue
                    definition["bytes"] = definition["bytes"] // owners * capacity
                    resized.add(name)
                    if definition.get("data"):
                        original = (root / definition["data"]["file"]).read_bytes()
                        if len(original) == definition["bytes"]:
                            continue
                        path = root / "data" / (name + "-dspark-head.bin")
                        path.write_bytes(original[: len(original) // owners] * capacity)
                        definition["data"] = dict(
                            file=str(path.relative_to(root)), sha256=file_hash(path)
                        )
                # Key includes the placeholder; normalize it for cross-rank reuse.
                identity = json.dumps(
                    {k: v for k, v in task.items() if k != "placeholder"}, sort_keys=True
                )
                if identity not in tasks:
                    tasks[identity] = task
                instruction["manifest"] = tasks[identity]["placeholder"]
    with (
        nullcontext(compiler)
        if compiler is not None
        else ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
    ) as pool:
        paths = dict(pool.map(compile_task, ((t, root) for t in tasks.values())))
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            for instruction in packet["head"]:
                if instruction.get("manifest") in paths:
                    instruction["manifest"] = paths[instruction["manifest"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args()
    if args.compose_only:
        compose(args.base_plan, args.output, args.workers, args.base_model)
    else:
        build(args.base_plan, args.checkpoint, args.output, args.workers, args.base_model)


if __name__ == "__main__":
    main()
