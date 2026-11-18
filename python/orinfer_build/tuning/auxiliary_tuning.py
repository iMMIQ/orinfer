"""Measure and publish exact auxiliary kernels through the native model builder.

Fusion changes are registered by the Rust adapter. This tool only constructs
checked operand bindings; it never publishes Python execution plans.
"""

from orinfer_build.common.paths import work_path


import argparse
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

from orinfer_build.compiler.build_cache import BuildCache, identity
from orinfer_build.compiler.host_compiler import Compiler
from orinfer_build.packaging.publication import ROOT, atomic_model, clone_model, load_model
from orinfer_build.common.files import write_json, source_hash


def source_identity():
    files = sorted((ROOT / "kernels/cuda/model").glob("*.py")) + [
        ROOT / "python/orinfer_build/validation/check_auxiliary.py",
        ROOT / "python/orinfer_build/tuning/expert_tuning.py",
        ROOT / "python/orinfer_build/contracts/row_selection.py",
        ROOT / "python/orinfer_build/tuning/row_tuning.py",
        ROOT / "python/orinfer_build/tuning/tuning.py",
        ROOT / "python/orinfer_build/common/files.py",
    ]
    return source_hash(ROOT, files)


def router_geometry(package, operands, buffers, rows):
    """Logical checkpoint dimensions are independent of flattened arena views."""
    text = package["config_signature"]["text"]
    experts, top = text["num_experts"], text["num_experts_per_tok"]
    if (
        any(type(n) is not int or n <= 0 for n in (experts, top, rows))
        or top > min(experts, 32)
        or experts % 32
        or text.get("norm_topk_prob", True) is not True
    ):
        raise ValueError("Unsupported normalized router contract")
    for role, dtype, columns in (
        ("Logits", "f32", experts),
        ("IDs", "i32", top),
        ("Prob", "f32", top),
    ):
        operand = buffers[operands[role]["name"]]
        offset = operands[role].get("offset", 0)
        if (
            operand["dtype"] != dtype
            or operand.get("data") is not None
            or offset % 4
            or math.prod(operand["shape"]) * 4 - offset < rows * columns * 4
        ):
            raise ValueError("Router geometry exceeds its published physical view")
    return experts, top


def fuse_config(family, output, source):
    """Use one aggregate winner; reject candidates with measured regressions."""
    from orinfer_build.validation.check_auxiliary import gating, selection
    from orinfer_build.compiler.cuda.common import environment

    contract = dict(source=source, environment=environment(), family=family)
    contract["environment"].pop("output_root", None)
    cache = BuildCache(work_path("cache", "build", "auxiliary"))

    def measure(folder):
        args = SimpleNamespace(
            output=folder,
            reference_dir=None,
            quick=False,
            qsa_parts=[2, 4],
        )
        records = []
        (gating if family == "ple" else selection)(args, records)
        write_json(folder / "results.json", dict(complete=True, cases=records))

    folder, hit = cache.get(contract, measure)
    records = json.loads((folder / "results.json").read_text())["cases"]
    grouped = {}
    for record in records:
        config = {k: record[k] for k in ("tile", "threads", "parts") if k in record}
        key = json.dumps(config, sort_keys=True)
        grouped.setdefault(key, []).append(record["candidate_ms"] / record["reference_ms"])
    legal = {key: values for key, values in grouped.items() if max(values) <= 1.05}
    best = min(
        legal, key=lambda key: sum(map(math.log, legal[key])) / len(legal[key]), default=None
    )
    config = (
        None
        if best is None or math.exp(sum(map(math.log, legal[best])) / len(legal[best])) >= 0.97
        else json.loads(best)
    )
    receipt = dict(
        contract=contract, cache_hit=hit, config=config, measured=str(folder / "results.json")
    )
    write_json(output / f"{family}-selection.json", receipt)
    return config


def qsa_pair(old, first, compiler, capacity, context, config, layout=None):
    from kernels.cuda.model import qsa_selection
    from kernels.cuda.model.independent_rows import independent_rows
    from orinfer_build.compiler.host_compiler import private_roles

    dynamic = layout is not None or any(a["kind"] == "i32" for a in old["args"])
    parts = config["parts"]
    if layout is None:
        p, first_p = compiler.pointers(old), compiler.pointers(first)
        if "Hist" not in first_p:
            raise ValueError("Missing QSA candidate workspace")
        spec = compiler.buffers[first_p["Hist"]["name"]]
        if math.prod(spec["shape"]) < capacity * parts * 1024:
            raise ValueError("QSA candidate union exceeds the published workspace")
        stages = [
            (
                first,
                dict(Scores=p["Scores"], Position=p["Position"], Selected=first_p["Hist"]),
                {},
                True,
            ),
            (
                old,
                dict(Candidates=first_p["Hist"], Position=p["Position"], Selected=p["Selected"]),
                {},
                False,
            ),
        ]
    else:
        columns = layout["state_columns"]
        stride = len(columns) * len(layout["layers"])
        roles, offsets = private_roles(compiler.read(old["source"]).decode(), columns, stride)
        first_roles, first_offsets = private_roles(
            compiler.read(first["source"]).decode(), columns, stride
        )
        if "Hist" not in first_roles or ((context + 4095) // 4096) * 256 < parts * 1024:
            raise ValueError("Missing bounded private QSA candidate union")
        hist, offset = first_roles["Hist"], first_offsets["Hist"]
        stages = [
            (
                first,
                dict(Scores=roles["Scores"], Position=roles["Position"], Selected=hist),
                dict(Scores=offsets["Scores"], Position=offsets["Position"], Selected=offset),
                True,
            ),
            (
                old,
                dict(Candidates=hist, Position=roles["Position"], Selected=roles["Selected"]),
                dict(Candidates=offset, Position=offsets["Position"], Selected=offsets["Selected"]),
                False,
            ),
        ]
    result = []
    for source, roles, offsets, partial in stages:
        key = (
            "aux-qsa-stage-"
            + identity(
                dict(
                    context=context,
                    config=config,
                    dynamic=dynamic,
                    private=layout is not None,
                    partial=partial,
                    roles=roles if layout else None,
                    offsets=offsets,
                )
            )[:16]
        )

        def build(roles=roles, offsets=offsets, partial=partial):
            rows = 1 if layout else capacity
            fn = (
                qsa_selection.select.__wrapped__(
                    rows, context, dynamic_rows=dynamic and layout is None, **config
                )
                if partial
                else qsa_selection.merge.__wrapped__(
                    rows, parts, dynamic_rows=dynamic and layout is None, threads=config["threads"]
                )
            )
            if layout is not None:
                return independent_rows(
                    fn,
                    128,
                    roles,
                    {n: i for i, n in enumerate(columns)},
                    row_stride=stride,
                    offsets=offsets,
                    dynamic=True,
                )[0]
            return fn

        result.append(
            compiler.emit(
                source,
                source["name"],
                compiler.pointers(source) if layout else roles,
                capacity,
                build=build,
                key=key,
                symbolic=dynamic,
            )
        )
    return result


def publish_qsa_prefill(package, manifest, compiler, *, attention_only=False):
    """Coalesce QSA scores and overlap packed KV reads in native Flash prefill."""
    from kernels.cuda.model import qsa, qsa_attention, qsa_selection

    if package["architecture"] != "flash_next":
        raise ValueError("Native QSA prefill layout requires Flash Next")
    originals = {k["name"]: k for k in package["kernels"]}
    dynamics = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    context = manifest["max_context"]
    replacements, launches = {}, {}
    attention_roles = {"Query", "K", "V", "KS", "VS", "Selected", "Position", "Max", "Den", "Out"}
    for plan in manifest["prefill_plans"]:
        rows = plan["chunk_tokens"]
        operations = manifest["programs"][plan["prefill_program"]]
        for previous, current in zip(operations, operations[1:]):
            selector = originals.get(current.get("name"))
            if selector is not None and selector["grid"] == [rows, 2, 8]:
                operands = compiler.pointers(selector)
                if set(operands) == attention_roles:
                    name = selector["name"]
                    if dynamics[name]["capacity"] != rows or selector["block"] != [128, 1, 1]:
                        raise ValueError("Unexpected QSA prefill attention geometry")
                    replacements[name], launches[name] = compiler.emit(
                        selector,
                        name,
                        operands,
                        rows,
                        build=lambda: qsa_attention.sparse_attention.__wrapped__(
                            rows,
                            context,
                            packed=True,
                            single_buffer=True,
                            dynamic_rows=True,
                            async_scales=True,
                            vector_dequant=True,
                        ),
                        key=f"qsa-single-buffer-vector-store-{context}-{rows}",
                    )
                    continue
            if (
                attention_only
                or selector is None
                or selector["symbol"] != "qsa_select_partial_kernel"
            ):
                continue
            producer = originals[previous["name"]]
            a, b = compiler.pointers(producer), compiler.pointers(selector)
            if (
                set(a) != {"Cache", "Position", "Query", "Scores"}
                or set(b) != {"Position", "Scores", "Selected"}
                or a["Scores"] != b["Scores"]
                or a["Position"] != b["Position"]
                or any(dynamics[k["name"]]["capacity"] != rows for k in (producer, selector))
            ):
                raise ValueError("Unexpected QSA prefill score producer/consumer contract")
            stripes, parts = producer["grid"][1], selector["grid"][1]
            threads = selector["block"][0]
            for old, pointers, kind, build, constants in (
                (
                    producer,
                    a,
                    f"score-{stripes}",
                    lambda: qsa.index_scores.__wrapped__(
                        rows, context, True, None, query_major=True
                    ),
                    {"stripes": stripes},
                ),
                (
                    selector,
                    b,
                    f"select-{parts}-{threads}",
                    lambda: qsa_selection.select.__wrapped__(
                        rows, context, True, 512, threads, parts, query_major=True
                    ),
                    None,
                ),
            ):
                name = old["name"]
                replacements[name], launches[name] = compiler.emit(
                    old,
                    name,
                    pointers,
                    rows,
                    build=build,
                    key=f"qsa-query-major-{context}-{rows}-{kind}",
                    constants=constants,
                )
    if not replacements:
        raise ValueError("Missing Flash Next native prefill QSA bindings")
    package["kernels"] = [replacements.get(k["name"], k) for k in package["kernels"]]
    package["dynamic_batch_kernels"] = [
        launches.get(k["name"], k) for k in package["dynamic_batch_kernels"]
    ]
    return sorted(replacements)


def publish_qsa_sort_prefill(package, manifest, compiler):
    """Keep within-warp exchanges in registers during the exact winner sort."""
    from kernels.cuda.model.qsa_selection import merge

    originals = {k["name"]: k for k in package["kernels"]}
    replacements, launches = {}, {}
    for plan in manifest["prefill_plans"]:
        rows = plan["chunk_tokens"]
        operations = manifest["programs"][plan["prefill_program"]]
        for previous, op in zip(operations, operations[1:]):
            old = originals.get(op.get("name"))
            if old is None or old["symbol"] != "qsa_select_merge_kernel":
                continue
            args = compiler.pointers(old)
            if set(args) != {"Candidates", "Position", "Selected"}:
                raise ValueError("Unexpected native QSA candidate merge ABI")
            partial = originals.get(previous.get("name"))
            if (
                partial is None
                or partial["symbol"] != "qsa_select_partial_kernel"
                or partial["grid"][1] != 2
                or compiler.pointers(partial)["Selected"] != args["Candidates"]
                or old["block"] != [128, 1, 1]
            ):
                raise ValueError("Unexpected native QSA merge geometry")
            parts = partial["grid"][1]
            name = old["name"]
            replacements[name], launches[name] = compiler.emit(
                old,
                name,
                args,
                rows,
                build=lambda: merge.__wrapped__(rows, parts, True, warp_sort=True),
                key=f"qsa-prefill-warp-sort-{rows}-{parts}",
            )
    if not replacements:
        raise ValueError("Missing native QSA prefill merge")
    package["kernels"] = [replacements.get(k["name"], k) for k in package["kernels"]]
    package["dynamic_batch_kernels"] = [
        launches.get(k["name"], k) for k in package["dynamic_batch_kernels"]
    ]
    return sorted(replacements)


def publish_expert_input_prefill(package, manifest, compiler):
    """Write K64-major down-projection bit planes directly after SwiGLU."""
    from kernels.cuda.model.rotation_a8 import rotate_expert_a8

    originals = {k["name"]: k for k in package["kernels"]}
    top = package["config_signature"]["text"]["num_experts_per_tok"]
    added, launches = {}, {}
    for plan in manifest["prefill_plans"]:
        rows = plan["chunk_tokens"]
        ops = manifest["programs"][plan["prefill_program"]]
        for previous, current in zip(ops, ops[1:]):
            old = originals.get(previous.get("name"))
            split = originals.get(current.get("name"))
            if old is None or split is None or split["symbol"] != "pack_exact_a8_kernel":
                continue
            if not any(
                arg.get("name", "").endswith(".ffn_down_exps.weight_signs") for arg in old["args"]
            ):
                continue
            rotation, planes = compiler.pointers(old), compiler.pointers(split)
            if (
                set(rotation) != {"Q", "S", "Signs", "X"}
                or set(planes) != {"A", "High", "Low", "Sum"}
                or rotation["Q"] != planes["A"]
                or compiler.buffers[rotation["Signs"]["name"]]["shape"] != [640]
            ):
                raise ValueError("Incompatible expert down rotation and bit-plane layouts")
            choice = f"/choice/2048-{rows}/1"
            name = old["name"] + choice + "/k0"
            bindings = {role: rotation[role] for role in ("S", "Signs", "X")}
            bindings.update({role: planes[role] for role in ("High", "Low", "Sum")})
            added[name], launches[name] = compiler.emit(
                old,
                name,
                bindings,
                rows,
                multiplier=top,
                build=lambda: rotate_expert_a8.__wrapped__(
                    rows * top,
                    640,
                    swiglu=True,
                    dynamic_rows=True,
                    group_major=True,
                ),
                key=f"flash-expert-down-planes-{rows}-{top}",
            )
            skip = split["name"] + choice + "/skip"
            added[skip] = dict(split, name=skip)
    if not added:
        raise ValueError("Missing native expert down input producers")
    package["kernels"].extend(added.values())
    package["dynamic_batch_kernels"].extend(launches.values())
    package["plan_kernel_contracts"] = sorted(set(package["plan_kernel_contracts"]) | added.keys())
    return sorted(added)


def publish_q4_prefill(data, package, manifest, compiler):
    """Expand one dense Q4 projection into shared temporary INT8 storage."""
    from kernels.cuda.model.int8_projection import int8_projection
    from kernels.cuda.model.q4_projection import unpack_q4

    if package["architecture"] != "flash_next":
        raise ValueError("Dense Q4 prefill expansion requires Flash Next")
    originals = {k["name"]: k for k in package["kernels"]}
    replacements, launches = {}, {}
    width = 2560
    workspaces = {}
    for role, outputs in (("Attention", 12288), ("GdnGate", 6144)):
        name = f"FlashPrefillExpanded{role}"
        if name in compiler.buffers:
            raise ValueError("Duplicate Q4 expansion workspace")
        desc = dict(
            name=name,
            dtype="i8",
            shape=[outputs, width],
            alignment=256,
            layout="native_contiguous",
            access="read_write",
        )
        data["metadata"]["buffers"].append(dict(desc, data=None))
        data["buffer_scopes"][name] = "workspace"
        package["buffer_contracts"].append(desc)
        compiler.buffers[name] = desc
        workspaces[role] = dict(kind="buffer", name=name)
    for plan in manifest["prefill_plans"]:
        rows = plan["chunk_tokens"]
        if rows != 4096:
            raise ValueError("Dense Q4 prefill expansion requires the native 4096-row arena")
        for op in manifest["programs"][plan["prefill_program"]]:
            if op["kind"] != "kernel":
                continue
            old = originals[op["name"]]
            if not any(
                arg.get("name", "").startswith("Q4_blk.") and arg["name"].endswith("_packed")
                for arg in old["args"]
            ):
                continue
            args = compiler.pointers(old)
            if set(args) != {"A", "W", "S", "AS", "Out"}:
                raise ValueError("Unexpected dense Q4 projection ABI")
            weights = compiler.buffers[args["W"]["name"]]
            outputs, packed_width = weights["shape"]
            if (
                weights["dtype"] != "u16"
                or packed_width * 4 != width
                or outputs not in (6144, 10240, 12288)
            ):
                raise ValueError("Unexpected dense Q4 prefill geometry")
            expanded = workspaces["GdnGate" if outputs == 6144 else "Attention"]
            # Use the smaller row tile at the complete native block, where
            # its extra CTA parallelism paid for the expansion on SM87.
            for first, last, block in ((2048, rows - 1, 128), (rows, rows, 64)):
                base = f"{old['name']}/choice/{first}-{last}/1"
                for idx, (bindings, key, build, symbolic) in enumerate(
                    (
                        (
                            {"W": args["W"], "Output": expanded},
                            f"q4-expand-{outputs}",
                            lambda: unpack_q4.__wrapped__(outputs, width),
                            False,
                        ),
                        (
                            {
                                "A": args["A"],
                                "Weight": expanded,
                                "WeightScale": args["S"],
                                "TokenScale": args["AS"],
                                "Output": args["Out"],
                            },
                            f"q4-expanded-int8-{rows}-{outputs}-{block}",
                            lambda: int8_projection.__wrapped__(
                                rows,
                                outputs,
                                width,
                                block_m=block,
                                block_n=128,
                                block_k=128,
                                dynamic_rows=True,
                            ),
                            True,
                        ),
                    )
                ):
                    name = f"{base}/k{idx}"
                    kernel, launch = compiler.emit(
                        old,
                        name,
                        bindings,
                        rows,
                        build=build,
                        key=key,
                        symbolic=symbolic,
                    )
                    replacements[name] = kernel
                    if symbolic:
                        launches[name] = launch
    if not replacements:
        raise ValueError("Missing Flash Next dense Q4 prefill bindings")
    package["kernels"].extend(replacements.values())
    package["dynamic_batch_kernels"].extend(launches.values())
    package["plan_kernel_contracts"] = sorted(
        set(package["plan_kernel_contracts"]) | replacements.keys()
    )
    return sorted(replacements)


def publish_conv_prefill(package, manifest, compiler):
    """Rebuild the native Flash prefill convolution with vector loads."""
    from kernels.cuda.operators.op08_gdn_conv_prep import gdn_conv_prep

    if package["architecture"] != "flash_next":
        raise ValueError("Native prefill convolution rebuild requires Flash Next")
    text = package["config_signature"]["text"]
    hk, hv = text["linear_num_key_heads"], text["linear_num_value_heads"]
    if (
        text["linear_conv_kernel_dim"] != 4
        or text["linear_key_head_dim"] != 128
        or text["linear_value_head_dim"] != 128
    ):
        raise ValueError("Unexpected Flash prefill convolution geometry")
    heads = 2 * hk + hv
    roles = {"X", "W", "HI", "HO", "lengths", "positions", "positions_out", "Q", "K", "V"}
    kernels = {k["name"]: k for k in package["kernels"]}
    dynamic = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    replacements, launches = {}, {}
    for plan in manifest["prefill_plans"]:
        rows = plan["chunk_tokens"]
        for operation in manifest["programs"][plan["prefill_program"]]:
            name = operation.get("name")
            old = kernels.get(name)
            if old is None or old["symbol"] != "kernel_kernel" or old["grid"][1:] != [heads, 1]:
                continue
            operands = compiler.pointers(old)
            if set(operands) != roles:
                continue
            weight = compiler.buffers[operands["W"]["name"]]
            if (
                dynamic.get(name, {}).get("capacity") != rows
                or old["block"] != [128, 1, 1]
                or any(a["kind"] != "buffer" for a in operands.values())
                or weight["dtype"] != "f32"
                or weight["shape"] != [heads * 128, 4]
                or operands["HI"] == operands["HO"]
            ):
                raise ValueError("Unexpected Flash prefill convolution operands")
            replacements[name], launches[name] = compiler.emit(
                old,
                name,
                operands,
                rows,
                build=lambda: gdn_conv_prep.__wrapped__(
                    B=1,
                    tokens=rows,
                    tile_tokens=32,
                    normalize_round_fp16=False,
                    conv_product_round_fp16=False,
                    weight_dtype="float32",
                    dynamic_tokens=True,
                    key_heads=hk,
                    value_heads=hv,
                ),
                key=f"flash-conv-vector-{rows}-{hk}-{hv}-32",
            )
    if not replacements:
        raise ValueError("Missing Flash prefill convolution bindings")
    package["kernels"] = [replacements.get(k["name"], k) for k in package["kernels"]]
    package["dynamic_batch_kernels"] = [
        launches.get(k["name"], k) for k in package["dynamic_batch_kernels"]
    ]
    return sorted(replacements)


def publish(data, package, manifest, compiler, output, families=("router", "ple", "qsa")):
    from kernels.cuda.model import moe, ple
    from orinfer_build.tuning.expert_tuning import MoeTuner
    from orinfer_build.contracts.row_selection import choices
    from orinfer_build.compiler.cuda.common import environment

    output.mkdir(parents=True, exist_ok=True)
    source = source_identity()
    machine = environment()
    machine.pop("output_root", None)
    tuner = MoeTuner(
        output / "router-tuning",
        dict(source=source, toolchain=package["toolchain"], environment=machine),
    )
    dynamics = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    originals = {k["name"]: k for k in package["kernels"]}
    replaced, added, launches, contracts = {}, [], {}, set(package["plan_kernel_contracts"])
    preceding, chains = {}, {}
    gates = {
        k["name"]
        for k in originals.values()
        if set(compiler.pointers(k)) == {"Gated", "Key", "Query", "Value"}
    }
    for operations in manifest["programs"].values():
        for i, op in enumerate(operations):
            if (
                op["kind"] == "kernel"
                and i >= 10
                and all(o["kind"] == "kernel" for o in operations[i - 10 : i])
            ):
                chain = tuple(o["name"] for o in operations[i - 10 : i])
                if op["name"] not in chains or chains[op["name"]] == chain:
                    chains[op["name"]] = chain
            if (
                op["kind"] == "kernel"
                and op["name"] in gates
                and i >= 2
                and all(o["kind"] == "kernel" for o in operations[i - 2 : i])
            ):
                pair = tuple(o["name"] for o in operations[i - 2 : i])
                if op["name"] in preceding and preceding[op["name"]] != pair:
                    raise ValueError("Ambiguous auxiliary producer chain")
                preceding[op["name"]] = pair
    configs = {}
    if data["architecture"] == "flash_next":
        configs = {
            family: fuse_config(family, output, source) for family in families if family != "router"
        }
    report = []
    for old in originals.values():
        name = old["name"]
        if "/rows/" in name:
            continue
        p = compiler.pointers(old)
        roles = set(p)
        dynamic = name in dynamics
        capacity = dynamics[name]["capacity"] if dynamic else 1
        if (
            roles == {"Pointers"}
            and configs.get("qsa") is not None
            and name.startswith("flash_decode_state/")
        ):
            from orinfer_build.compiler.host_compiler import private_roles

            layout = data["metadata"]["batch_layout"]
            columns = layout["state_columns"]
            stride = len(columns) * len(layout["layers"])
            private, offsets = private_roles(compiler.read(old["source"]).decode(), columns, stride)
            if set(private) != {"Greater", "Offsets", "Position", "Prefix", "Scores", "Selected"}:
                continue
            for item, launch in qsa_pair(
                old,
                originals[chains[name][0]],
                compiler,
                capacity,
                data["metadata"]["max_context"],
                configs["qsa"],
                layout,
            ):
                replaced[item["name"]] = item
                launches[item["name"]] = launch
                contracts.add(item["name"])
            report.append(dict(kind="qsa-private", name=name))
            continue
        if roles == {"IDs", "Logits", "Prob"} and "router" in families:
            e, k = router_geometry(package, p, compiler.buffers, capacity)
            selected = tuner.topk(e, k)
            spans = (
                list(choices(selected, capacity))
                if dynamic
                else [
                    (
                        "",
                        selected["configs"][
                            int(
                                next(
                                    (
                                        x["implementation"]
                                        for x in selected["selections"]
                                        if x["min_rows"] <= 1 <= x["max_rows"]
                                    ),
                                    selected["default"],
                                )
                            )
                        ],
                    )
                ]
            )
            for suffix, config in spans:
                fn = moe.router_topk_warp if config["algorithm"] == "warp" else moe.router_topk
                options = {key: value for key, value in config.items() if key != "algorithm"}
                key = f"aux-topk-e{e}-k{k}-{config['algorithm']}-w{options.get('warps', 0)}-{'dynamic' if dynamic else 'single'}"
                item, launch = compiler.emit(
                    old,
                    name + suffix,
                    p,
                    capacity,
                    build=lambda fn=fn, options=options, capacity=capacity, e=e, k=k, dynamic=dynamic: (
                        fn.__wrapped__(capacity, e, k, dynamic_rows=dynamic, **options)
                    ),
                    key=key,
                    symbolic=dynamic,
                )
                if suffix:
                    added.append(item)
                else:
                    replaced[name] = item
                if dynamic:
                    launches[item["name"]] = launch
                contracts.add(item["name"])
            report.append(dict(kind="router", name=name, experts=e, top_k=k))
        elif roles == {"Gated", "Key", "Query", "Value"} and configs.get("ple") is not None:
            if name not in preceding:
                raise ValueError("Missing PLE norm producers")
            norms = [compiler.pointers(originals[n]) for n in preceding[name]]
            if (
                any(set(n) != {"Normed", "Residual", "Weight"} for n in norms)
                or norms[0]["Normed"] != p["Key"]
                or norms[1]["Normed"] != p["Query"]
            ):
                raise ValueError("PLE norm/gate producer mismatch")
            shape = compiler.buffers[norms[0]["Weight"]["name"]]["shape"]
            streams, h = shape
            if compiler.buffers[p["Gated"]["name"]]["dtype"] != "f16":
                raise ValueError("Fused PLE requires the certified FP16 stream")
            bindings = dict(
                Gated=p["Gated"],
                Key=norms[0]["Residual"],
                Query=norms[1]["Residual"],
                KeyWeight=norms[0]["Weight"],
                QueryWeight=norms[1]["Weight"],
                Value=p["Value"],
            )
            key = f"aux-ple-h{h}-s{streams}-{'dynamic' if dynamic else 'single'}"
            item, launch = compiler.emit(
                old,
                name,
                bindings,
                capacity,
                build=lambda capacity=capacity, h=h, streams=streams, dynamic=dynamic: (
                    ple.ple_norm_gate.__wrapped__(capacity, h, streams, dynamic_rows=dynamic)
                ),
                key=key,
                symbolic=dynamic,
            )
            replaced[name] = item
            if dynamic:
                launches[name] = launch
            contracts.add(name)
            report.append(dict(kind="ple", name=name))
        elif (
            roles == {"Greater", "Offsets", "Position", "Prefix", "Scores", "Selected"}
            and configs.get("qsa") is not None
        ):
            config = configs["qsa"]
            context = data["metadata"]["max_context"]
            for item, launch in qsa_pair(
                old, originals[chains[name][0]], compiler, capacity, context, config
            ):
                replaced[item["name"]] = item
                if dynamic:
                    launches[item["name"]] = launch
                contracts.add(item["name"])
            report.append(dict(kind="qsa", name=name))
            continue
    removed = {
        k["name"]
        for k in package["kernels"]
        if "/rows/" in k["name"] and k["name"].split("/rows/")[0] in replaced
    }
    package["kernels"] = [
        replaced.get(k["name"], k) for k in package["kernels"] if k["name"] not in removed
    ] + added
    package["dynamic_batch_kernels"] = [
        launches.pop(k["name"], k)
        for k in package["dynamic_batch_kernels"]
        if k["name"] not in removed
    ] + list(launches.values())
    package["plan_kernel_contracts"] = sorted((contracts - removed) | {k["name"] for k in added})
    write_json(
        output / "published.json", dict(source=source, bindings=report, fusion_configs=configs)
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "model-output", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument(
        "--families",
        nargs="+",
        choices=(
            "router",
            "ple",
            "qsa",
            "qsa-prefill",
            "conv-prefill",
            "q4-prefill",
            "expert-input-prefill",
            "qsa-sort-prefill",
        ),
        default=["router", "ple", "qsa"],
    )
    args = p.parse_args()
    from orinfer_build.compiler.cuda.common import configure
    from orinfer_build.packaging.compact import compact_package

    configure()
    manifest = {}
    data, origin, package = load_model(args.model, plan=manifest)
    with atomic_model(args.model_output) as staging:
        root = clone_model(args.model, staging, origin)
        compiler = Compiler(
            origin, root, args.output, {b["name"]: b for b in data["metadata"]["buffers"]}
        )
        prefill_publishers = {
            "qsa-prefill": publish_qsa_prefill,
            "conv-prefill": publish_conv_prefill,
            "expert-input-prefill": publish_expert_input_prefill,
            "qsa-sort-prefill": publish_qsa_sort_prefill,
        }
        families = tuple(
            family
            for family in args.families
            if family not in prefill_publishers and family != "q4-prefill"
        )
        if "q4-prefill" in args.families:
            names = publish_q4_prefill(data, package, manifest, compiler)
            write_json(args.output / "q4-prefill.json", dict(bindings=names))
        if families:
            publish(data, package, manifest, compiler, args.output, families)
        for family, build in prefill_publishers.items():
            if family in args.families:
                names = build(package, manifest, compiler)
                write_json(args.output / f"{family}.json", dict(bindings=names))
        digest = compact_package(root, package, staging / "cache/packages")
        shutil.rmtree(root)
        data["execution_package"] = digest
        write_json(staging / "cache/model.json", data)
    print("COMPLETE", digest, flush=True)


if __name__ == "__main__":
    main()
