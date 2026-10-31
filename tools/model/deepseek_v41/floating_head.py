"""Prepare released FP32 output heads for the shared floating Cube factory."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from tools.model.deepseek_v41.export_release import CHUNK, Export
from tools.model.deepseek_v41.floating import bf16_parts


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
    def __init__(self, *args, build_dir, tile_m=64, tile_n=128, **kwargs):
        super().__init__(*args, **kwargs)
        if tile_m not in (16, 32, 64, 128) or tile_n not in (32, 64, 128):
            raise ValueError("Require a supported floating projection tile")
        self.original_model = deepcopy(self.model)
        self.build_dir = Path(build_dir)
        self.tile_m, self.tile_n = tile_m, tile_n

    def plan(self):
        config = json.loads((self.source / self.model["config"]["file"]).read_text())
        config = config.get("text_config", config)
        if config["vocab_size"] % len(self.model["ranks"]):
            raise ValueError("Output vocabulary must divide the tensor-parallel ranks")
        self.n = config["vocab_size"] // len(self.model["ranks"])
        self.k = config["hidden_size"]
        if self.n % 32 or self.k % 1024:
            raise ValueError("Output heads require aligned output and eight K128 splits")
        self.calls = head_calls(self.model, self.package, self.n, self.k)
        self.index = deepcopy(self.old_index)
        old = json.loads((self.source / "release-manifest.json").read_text())
        self.old_records = old["files"]
        source_package = self.manifest["source_package"]
        self.manifest.update(deepcopy(old))
        self.manifest["source_package"] = source_package
        self.manifest["files"] = {}
        self.manifest["fp32_head_layout"] = "bf16-residual3"
        self.index["metadata"]["fp32_head_layout"] = "bf16-residual3"
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
                self.transforms[original] = original + ".bf16_residual3"
            weight["bytes"] = self.n * self.k * 6
            weight["prepared_tensor"] = dict(
                name=self.transforms[original], dtype="BF16", shape=[3, self.n, self.k]
            )
            scratch["bytes"] = capacity * self.padded * 8 * 4
        for group, record in self.old_records.items():
            path = self.source / group
            if path not in self.headers:
                from tools.model.safetensors_source import local_header

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
                    source["bf16_parts"] = True
                    dtype, shape, size = "BF16", [3, self.n, self.k], self.n * self.k * 6
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
        from kernels.model.rows import row_count
        from tools.model.deepseek_v41.assembly import publish_package, validate_rows
        from tools.operators.ascend import floating
        from tools.operators.ascend.build import compile_kernel, operand

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
                weight_parts=3,
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
                    operand("weight", "bf16", [3, self.n, self.k]),
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
        parts = bf16_parts(np.frombuffer(raw, dtype="<f4").reshape(self.n, self.k))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    job = HeadRelease(
        args.source,
        args.output,
        source_execution=args.source_execution,
        execution_output=args.execution_output,
        build_dir=args.build_dir,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
    ).plan()
    print(
        json.dumps(
            dict(
                head_banks=len(job.transforms),
                calls=len(job.calls),
                rewritten_shards=len(job.rewrite_groups),
                tile=[args.tile_m, args.tile_n],
            )
        ),
        flush=True,
    )
    if not args.plan_only:
        job.build()
        job.export(args.workers)


if __name__ == "__main__":
    main()
