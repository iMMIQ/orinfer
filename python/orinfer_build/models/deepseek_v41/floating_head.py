"""Prepare output-head BF16 components with the shared floating Cube factory."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from orinfer_build.models.deepseek_v41.export_release import CHUNK, Export
from orinfer_build.models.deepseek_v41.floating import bf16_exact, bf16_parts


def head_calls(model, package, output, reduction):
    """Find complete partial/reduce pairs and reject other users of their banks."""
    specs, calls, converted, allowed = {}, [], set(), set()

    def describe(step):
        path = step.get("manifest")
        if path not in specs:
            specs[path] = json.loads((package / path).read_text())
        return specs[path]

    for r, rank in enumerate(model["ranks"]):
        definitions = {b["name"]: b for b in rank["buffers"]}
        for packet_index, packet in enumerate(rank["packets"]):
            for program in ("body", "head"):
                steps = packet[program]
                for i, step in enumerate(steps):
                    if step["op"] != "kernel":
                        continue
                    spec = describe(step)
                    p = spec["parameters"]
                    if spec["op"] != "dense.fp32_partials" or (
                        p.get("output"),
                        p.get("reduction"),
                    ) != (output, reduction):
                        continue
                    capacity = p.get("_row_capacity")
                    if (
                        p.get("activation_dtype") != "bfloat16"
                        or p.get("splits", 8) != 8
                        or not capacity
                        or len(step["buffers"]) != 3
                        or step.get("uncompacted_manifest")
                        or i + 1 >= len(steps)
                    ):
                        raise ValueError("Output head requires the dynamic BF16/F32 projection ABI")
                    following = steps[i + 1]
                    paired = describe(following) if following["op"] == "kernel" else {}
                    if (
                        paired.get("op") != "dense.fp32_reduce"
                        or paired["parameters"].get("output") != output
                        or paired["parameters"].get("_row_capacity") != capacity
                        or paired["parameters"].get("splits", 8) != 8
                        or following["buffers"][0] != step["buffers"][2]
                        or following.get("uncompacted_manifest")
                    ):
                        raise ValueError("Output head partials require their matching FP32 reducer")
                    weight = definitions[step["buffers"][1]]
                    tensor = weight.get("prepared_tensor", {})
                    if (
                        tensor.get("dtype") != "F32"
                        or tensor.get("shape") != [output, reduction]
                        or weight["bytes"] != output * reduction * 4
                        or definitions[step["buffers"][2]]["bytes"] != capacity * output * 8 * 4
                    ):
                        raise ValueError("Output head requires a released physical FP32 bank")
                    converted.add((r, weight["name"]))
                    allowed.add((r, packet_index, program, i))
                    calls.append((r, packet_index, program, i, capacity))
    for r, rank in enumerate(model["ranks"]):
        for packet_index, packet in enumerate(rank["packets"]):
            for program in ("body", "head"):
                for i, step in enumerate(packet[program]):
                    if (
                        any((r, name) in converted for name in step.get("buffers", []))
                        and (
                            r,
                            packet_index,
                            program,
                            i,
                        )
                        not in allowed
                    ):
                        raise ValueError("Converted output-head bank has another consumer")
    if not calls:
        raise ValueError("No released FP32 output-head projections found")
    return calls


class HeadRelease(Export):
    def __init__(self, *args, build_dir, tile_m=64, tile_n=128, markov_bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        if tile_m not in (16, 32, 64, 128) or tile_n not in (32, 64, 128):
            raise ValueError("Require a supported floating projection tile")
        self.original_model = deepcopy(self.model)
        self.build_dir = Path(build_dir)
        self.tile_m, self.tile_n = tile_m, tile_n
        self.markov_bias = markov_bias
        self.weight_parts = 1 if markov_bias else 3

    def plan(self):
        config = json.loads((self.source / self.model["config"]["file"]).read_text())
        config = config.get("text_config", config)
        if config["vocab_size"] % len(self.model["ranks"]):
            raise ValueError("Output vocabulary must divide the tensor-parallel ranks")
        self.n = config["vocab_size"] // len(self.model["ranks"])
        self.k = config.get("dspark_markov_rank") if self.markov_bias else config["hidden_size"]
        if (
            type(self.k) is not int
            or self.k <= 0
            or self.n % 32
            or self.k % (128 if self.markov_bias else 1024)
        ):
            raise ValueError("Output heads require aligned output and eight Cube K splits")
        self.calls = head_calls(self.model, self.package, self.n, self.k)
        self.index = deepcopy(self.old_index)
        old = json.loads((self.source / "release-manifest.json").read_text())
        self.old_records = old["files"]
        source_package = self.manifest["source_package"]
        self.manifest.update(deepcopy(old))
        self.manifest["source_package"] = source_package
        self.manifest["files"] = {}
        key = "markov_bias_layout" if self.markov_bias else "fp32_head_layout"
        layout = "bf16-exact" if self.markov_bias else "bf16-residual3"
        self.manifest[key] = layout
        self.index["metadata"][key] = layout
        self.transforms, self.rewrite_groups = {}, set()
        self.padded = (self.n + self.tile_n - 1) // self.tile_n * self.tile_n
        originals = [
            {b["name"]: b for b in rank["buffers"]} for rank in self.original_model["ranks"]
        ]
        for r, packet, program, i, capacity in self.calls:
            rank = self.model["ranks"][r]
            definitions = {b["name"]: b for b in rank["buffers"]}
            step = rank["packets"][packet][program][i]
            weight, scratch = (definitions[name] for name in step["buffers"][1:])
            original = originals[r][weight["name"]]["prepared_tensor"]["name"]
            if original not in self.transforms:
                suffix = ".bf16_exact" if self.markov_bias else ".bf16_residual3"
                self.transforms[original] = original + suffix
            weight["bytes"] = self.n * self.k * 2 * self.weight_parts
            weight["prepared_tensor"] = dict(
                name=self.transforms[original],
                dtype="BF16",
                shape=[self.weight_parts, self.n, self.k],
            )
            scratch["bytes"] = capacity * self.padded * 8 * 4
        for group, record in self.old_records.items():
            path = self.source / group
            if path not in self.headers:
                from orinfer_build.weights.safetensors_source import local_header

                self.headers[path] = local_header(path)
            start, header = self.headers[path]
            for name in sorted(record["tensors"], key=lambda n: header[n]["data_offsets"][0]):
                spec = header[name]
                source = dict(file=str(path), offset=start + spec["data_offsets"][0])
                dtype, shape = spec["dtype"], spec["shape"]
                size = spec["data_offsets"][1] - spec["data_offsets"][0]
                if name in self.transforms:
                    if dtype != "F32" or shape != [self.n, self.k]:
                        raise ValueError("Released head tensor differs from its original bank")
                    source["bf16_parts"] = self.weight_parts
                    dtype, shape, size = (
                        "BF16",
                        [self.weight_parts, self.n, self.k],
                        self.n * self.k * 2 * self.weight_parts,
                    )
                    replacement = self.transforms[name]
                    if replacement in self.index["weight_map"]:
                        raise ValueError("Residual head tensor name already exists")
                    self.index["weight_map"][replacement] = self.index["weight_map"].pop(name)
                    name = replacement
                    self.rewrite_groups.add(group)
                self.groups[group].append(
                    (name, dict(dtype=dtype, shape=shape, bytes=size, source=source))
                )
        if any(sum(e["bytes"] for _, e in entries) >= 4 << 30 for entries in self.groups.values()):
            raise ValueError("Head conversion exceeds the four-GiB semantic shard budget")
        return self

    def build(self):
        from kernels.shared.rows import row_count
        from orinfer_build.models.deepseek_v41.assembly import publish_package, validate_rows
        from kernels.ascend import floating
        from orinfer_build.compiler.ascend.build import compile_kernel, operand

        source = self.build_dir / "source"
        if source.exists():
            if json.loads((source / "model.json").read_text()) != self.original_model:
                raise ValueError("Head compilation input changed")
        else:
            publish_package(self.original_model, self.package, source)
        self.package = source
        kernels = {}
        for _, _, _, _, capacity in self.calls:
            if capacity in kernels:
                continue
            params = dict(
                rows=capacity,
                output=self.n,
                reduction=self.k,
                input_parts=1,
                weight_parts=self.weight_parts,
                block_k=math.gcd(self.k // 8, 256),
                tile_m=self.tile_m,
                tile_n=self.tile_n,
                reuse_operand=True,
            )
            partial = compile_kernel(
                floating.partials(**dict(params, rows=row_count(capacity, True))),
                "floating.partials",
                [
                    operand("activation", "bf16", [1, capacity, self.k]),
                    operand("weight", "bf16", [self.weight_parts, self.n, self.k]),
                    operand("out", "f32", [capacity, self.padded * 8], "write"),
                ],
                params,
                source / "cache",
                row_capacity=capacity,
                grid_limit=64,
            )
            params = dict(rows=capacity, output=self.n, tile_n=self.tile_n)
            reduce = compile_kernel(
                floating.reduce_partials(**dict(params, rows=row_count(capacity, True))),
                "floating.reduce_partials",
                [
                    operand(
                        "partials", "f32", [capacity, self.padded // self.tile_n, 8, self.tile_n]
                    ),
                    operand("out", "f32", [capacity, self.n], "write"),
                ],
                params,
                source / "cache",
                row_capacity=capacity,
                grid_limit=64,
            )
            kernels[capacity] = str(partial.relative_to(source)), str(reduce.relative_to(source))
        for r, packet, program, i, capacity in self.calls:
            steps = self.model["ranks"][r]["packets"][packet][program]
            steps[i]["manifest"], steps[i + 1]["manifest"] = kernels[capacity]
        validate_rows(self.model, source, model_dir=self.source)

    def chunks(self, entry):
        source = entry["source"]
        if not source.get("bf16_parts"):
            yield from super().chunks(entry)
            return
        with Path(source["file"]).open("rb") as stream:
            stream.seek(source["offset"])
            raw = stream.read(self.n * self.k * 4)
        if len(raw) != self.n * self.k * 4:
            raise ValueError("Truncated original FP32 head")
        values = np.frombuffer(raw, dtype="<f4").reshape(self.n, self.k)
        parts = bf16_exact(values) if self.weight_parts == 1 else bf16_parts(values)
        data = memoryview(parts.astype("<u2", copy=False)).cast("B")
        for start in range(0, len(data), CHUNK):
            yield data[start : start + CHUNK]

    def write_shard(self, group):
        if group in self.rewrite_groups:
            return super().write_shard(group)
        target = self.output / group
        target.parent.mkdir(parents=True, exist_ok=True)
        target.hardlink_to(self.source / group)
        return group, deepcopy(self.old_records[group])


class CompactHeadRelease(HeadRelease):
    """Retain published head planes, with rounding explicitly requested offline."""

    def __init__(self, *args, retained_parts=1, require_exact=True, **kwargs):
        super().__init__(*args, **kwargs)
        if (retained_parts, require_exact) not in ((1, True), (2, False)):
            raise ValueError("Use exact zero-plane compaction or explicitly rounded two-part heads")
        self.weight_parts = retained_parts
        self.require_exact = require_exact

    def plan(self):
        config = json.loads((self.source / self.model["config"]["file"]).read_text())
        config = config.get("text_config", config)
        if config["vocab_size"] % len(self.model["ranks"]):
            raise ValueError("Output vocabulary must divide the tensor-parallel ranks")
        self.n, self.k = config["vocab_size"] // len(self.model["ranks"]), config["hidden_size"]
        descriptions, candidates = {}, {}
        self.calls, self.transforms = [], {}
        for r, rank in enumerate(self.model["ranks"]):
            definitions = {b["name"]: b for b in rank["buffers"]}
            for pi, packet in enumerate(rank["packets"]):
                for program in ("body", "head"):
                    for i, step in enumerate(packet[program]):
                        if step["op"] != "kernel":
                            continue
                        path = step["manifest"]
                        if path not in descriptions:
                            descriptions[path] = json.loads((self.package / path).read_text())
                        spec = descriptions[path]
                        p = spec["parameters"]
                        if spec["op"] != "floating.partials" or (
                            p.get("output"),
                            p.get("reduction"),
                            p.get("input_parts", 1),
                            p.get("weight_parts", 3),
                        ) != (self.n, self.k, 1, 3):
                            continue
                        if len(step["buffers"]) != 3:
                            raise ValueError("Floating head requires its three-operand ABI")
                        bank = definitions[step["buffers"][1]]
                        tensor = bank.get("prepared_tensor", {})
                        if (
                            tensor.get("dtype") != "BF16"
                            or tensor.get("shape") != [3, self.n, self.k]
                            or bank["bytes"] != 3 * self.n * self.k * 2
                            or not p.get("_row_capacity")
                        ):
                            raise ValueError("Head compaction requires three published BF16 planes")
                        candidates.setdefault(tensor["name"], []).append((r, pi, program, i))
        for name, calls in candidates.items():
            source = self.tensor(name)
            if source["dtype"] != "BF16" or source["shape"] != [3, self.n, self.k]:
                raise ValueError("Published head header differs from its physical binding")
            values = self.array(name)
            planes = values.reshape(3, -1)
            if self.require_exact and any(
                np.any(plane[first : first + CHUNK // 2])
                for plane in planes[self.weight_parts :]
                for first in range(0, plane.size, CHUNK // 2)
            ):
                continue
            if any(
                np.any((plane[first : first + CHUNK // 2] & 0x7F80) == 0x7F80)
                for plane in planes
                for first in range(0, planes.shape[1], CHUNK // 2)
            ):
                raise ValueError("Exact head requires finite BF16 weights")
            suffix = ".bf16_residual3"
            layout = "bf16_exact" if self.require_exact else f"bf16_residual{self.weight_parts}"
            replacement = name.removesuffix(suffix) + "." + layout
            if replacement in self.old_index["weight_map"]:
                raise ValueError("Exact head tensor name already exists")
            self.transforms[name] = replacement
            self.calls.extend(calls)
        if not self.calls:
            raise ValueError("No redundant BF16 output-head planes")
        allowed = set(self.calls)
        for r, rank in enumerate(self.model["ranks"]):
            changed = set()
            for bank in rank["buffers"]:
                tensor = bank.get("prepared_tensor", {})
                if tensor.get("name") in self.transforms:
                    changed.add(bank["name"])
                    tensor.update(
                        name=self.transforms[tensor["name"]],
                        shape=[self.weight_parts, self.n, self.k],
                    )
                    bank["bytes"] = self.weight_parts * self.n * self.k * 2
            if any(b.get("view", {}).get("parent") in changed for b in rank["buffers"]):
                raise ValueError("Compacted head cannot have another buffer view")
            for pi, packet in enumerate(rank["packets"]):
                for program in ("body", "head"):
                    for i, step in enumerate(packet[program]):
                        if (
                            changed.intersection(step.get("buffers", []))
                            and (r, pi, program, i) not in allowed
                        ):
                            raise ValueError("Compacted head has another consumer")
        self.index = deepcopy(self.old_index)
        old = json.loads((self.source / "release-manifest.json").read_text())
        self.old_records, self.rewrite_groups = old["files"], set()
        source_package = self.manifest["source_package"]
        self.manifest.update(deepcopy(old))
        self.manifest.update(source_package=source_package, files={})
        if self.require_exact:
            self.manifest["exact_bf16_heads"] = deepcopy(self.transforms)
        else:
            self.manifest["fp32_head_layout"] = f"bf16-residual{self.weight_parts}"
            self.index["metadata"]["fp32_head_layout"] = self.manifest["fp32_head_layout"]
            self.manifest["head_rounding"] = dict(
                source_parts=3, retained_parts=self.weight_parts, exact=False
            )
        for name, replacement in self.transforms.items():
            logical = self.manifest.get("logical_shapes", {})
            if name in logical:
                logical[replacement] = logical.pop(name)
        for group, record in self.old_records.items():
            path = self.source / group
            if path not in self.headers:
                from orinfer_build.weights.safetensors_source import local_header

                self.headers[path] = local_header(path)
            start, header = self.headers[path]
            for name in sorted(record["tensors"], key=lambda n: header[n]["data_offsets"][0]):
                spec = header[name]
                size = spec["data_offsets"][1] - spec["data_offsets"][0]
                dtype, shape = spec["dtype"], spec["shape"]
                source = dict(file=str(path), offset=start + spec["data_offsets"][0])
                if name in self.transforms:
                    if dtype != "BF16" or shape != [3, self.n, self.k]:
                        raise ValueError("Compacted head differs from its published tensor")
                    shape = [self.weight_parts, self.n, self.k]
                    size = self.weight_parts * self.n * self.k * 2
                    replacement = self.transforms[name]
                    self.index["weight_map"][replacement] = self.index["weight_map"].pop(name)
                    name = replacement
                    self.rewrite_groups.add(group)
                self.groups[group].append(
                    (name, dict(dtype=dtype, shape=shape, bytes=size, source=source))
                )
        return self

    def build(self):
        from orinfer_build.models.deepseek_v41.assembly import publish_package, validate_rows
        from orinfer_build.models.deepseek_v41.build import compile_task, kernel_rebuild_task

        source = self.build_dir / "source"
        if source.exists():
            if json.loads((source / "model.json").read_text()) != self.original_model:
                raise ValueError("Head compilation input changed")
        else:
            publish_package(self.original_model, self.package, source)
        self.package = source
        tasks = {}

        def rebuild(path, convert=True):
            if path in tasks:
                return
            spec = json.loads((source / path).read_text())
            p = spec["parameters"]
            if spec["op"] != "floating.partials":
                return
            if convert and (
                p.get("output"),
                p.get("reduction"),
                p.get("input_parts", 1),
                p.get("weight_parts", 3),
            ) != (self.n, self.k, 1, 3):
                raise ValueError("Head variants must retain their original matrix contract")
            task = kernel_rebuild_task(deepcopy(spec), path)
            if convert:
                task["arguments"]["weight_parts"] = self.weight_parts
                if task["specs"][1]["name"] != "weight" or task["specs"][1]["shape"] != [
                    3,
                    self.n,
                    self.k,
                ]:
                    raise ValueError("Head variant physical weight geometry differs")
                task["specs"][1]["shape"][0] = self.weight_parts
            tasks[path] = task

        for r, pi, program, i in self.calls:
            step = self.model["ranks"][r]["packets"][pi][program][i]
            for path in [step["manifest"], step.get("uncompacted_manifest")]:
                if path is None:
                    continue
                rebuild(path)
                if path in self.model.get("row_selections", {}):
                    for variant in self.model["row_selections"][path]:
                        for kernel in variant["kernels"]:
                            rebuild(kernel)
        # Publication pins a single source identity per operator. Rebuild the
        # unchanged consumers of this factory too, retaining their arithmetic.
        references = []
        for step in self.model.get("vision", {}).get("body", []):
            references.append((step, "manifest"))
        for rank in self.model["ranks"]:
            for packet in rank["packets"]:
                for step in packet["body"] + packet["head"]:
                    if step["op"] == "kernel":
                        references.extend(
                            (step, key)
                            for key in ("manifest", "uncompacted_manifest")
                            if step.get(key)
                        )
        for variants in self.model.get("row_selections", {}).values():
            for variant in variants:
                references.extend((variant["kernels"], i) for i in range(len(variant["kernels"])))
        for binding, key in references:
            rebuild(binding[key], convert=False)
        compiled = dict(compile_task((task, source)) for task in tasks.values())
        for binding, key in references:
            binding[key] = compiled.get(binding[key], binding[key])
        self.model["row_selections"] = {
            compiled.get(path, path): variants
            for path, variants in self.model.get("row_selections", {}).items()
        }
        used = {
            s[key]
            for rank in self.model["ranks"]
            for packet in rank["packets"]
            for s in packet["body"] + packet["head"]
            if s["op"] == "kernel"
            for key in ("manifest", "uncompacted_manifest")
            if s.get(key)
        }
        self.model["row_selections"] = {
            k: v for k, v in self.model.get("row_selections", {}).items() if k in used
        }
        validate_rows(self.model, source, model_dir=self.source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=128)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--markov-bias",
        action="store_true",
        help="Convert the configured DSpark Markov bias bank only when exactly representable in BF16",
    )
    mode.add_argument(
        "--compact-zero-planes",
        action="store_true",
        help="Remove fully zero BF16 head residual planes without rounding weights",
    )
    mode.add_argument(
        "--head-residual-parts",
        type=int,
        choices=[2],
        help="Round a published three-part head to two BF16 components; requires quality acceptance",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    compact = args.compact_zero_planes or args.head_residual_parts is not None
    extra = (
        dict(retained_parts=args.head_residual_parts, require_exact=False)
        if args.head_residual_parts is not None
        else {}
    )
    job = (CompactHeadRelease if compact else HeadRelease)(
        args.source,
        args.output,
        source_execution=args.source_execution,
        execution_output=args.execution_output,
        build_dir=args.build_dir,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        markov_bias=args.markov_bias,
        **extra,
    ).plan()
    print(
        json.dumps(
            dict(
                head_banks=len(job.transforms),
                calls=len(job.calls),
                rewritten_shards=len(job.rewrite_groups),
                tile="preserved" if compact else [args.tile_m, args.tile_n],
            )
        ),
        flush=True,
    )
    if not args.plan_only:
        job.build()
        job.export(args.workers)


if __name__ == "__main__":
    main()
