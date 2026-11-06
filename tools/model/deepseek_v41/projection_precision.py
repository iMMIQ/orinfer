"""Recover selected Q/KV projections from the official checkpoint in the released basis."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import re

import numpy as np

from tools.model.deepseek_v41.export_release import TensorRelease
from tools.model.deepseek_v41.expert_precision import block_rotation
from tools.model.deepseek_v41.floating import bf16_parts
from tools.model.deepseek_v41.official import Checkpoint


def reference_projection(weight, gamma, current_norm, rotation, *, query=False):
    """Preserve norm folding, block32 input basis and shifted low-rank Q basis.

    The source QuaRot recipe applies Rq=block(R) P16 block(R) on Q outputs.
    SmoothQuant stores 1/s in the current norm, so weights absorb s. All
    transformations are offline; the input remains the existing BF16 norm.
    """
    weight = np.asarray(weight, dtype="f4")
    gamma, current_norm, rotation = map(
        lambda x: np.asarray(x, dtype="f4"), (gamma, current_norm, rotation)
    )
    n, k = weight.shape
    if (
        k % 32
        or n % 32
        or gamma.shape != (k,)
        or current_norm.shape != (k,)
        or rotation.shape != (32, 32)
        or np.any(current_norm <= 0)
        or not all(np.isfinite(x).all() for x in (weight, gamma, current_norm, rotation))
    ):
        raise ValueError("Q/KV reference requires finite block32 geometry and positive norm scales")
    result = ((weight * gamma).reshape(n, k // 32, 32) @ rotation).reshape(n, k)
    if query:
        result = (rotation.T @ result.reshape(n // 32, 32, k)).reshape(n, k)
        result = np.roll(result, 16, axis=0)
        result = (rotation.T @ result.reshape(n // 32, 32, k)).reshape(n, k)
    return result / current_norm


def floating_projection_spec(spec):
    """Select one BF16 dynamic fallback from the shared projection factory."""
    spec = deepcopy(spec)
    p = spec["parameters"]
    capacity = p["_row_capacity"]
    if spec["op"] != "dense.projection" or not p.get("integer") or p["reduction"] != 5120:
        raise ValueError("Require the released dynamic INT8 Q/KV projection")
    p.update(
        integer=False,
        unit_scale=True,
        tile_m=16 if capacity <= 32 else 64,
        tile_n=64 if capacity <= 32 else 128,
        block_k=512 if capacity <= 32 else 256,
        pipeline_stages=0 if capacity <= 32 else 2,
    )
    for index in (0, 1):
        spec["operands"][index]["dtype"] = "bf16"
    spec["operands"][5]["dtype"] = "f32"
    return spec


class ProjectionRelease(TensorRelease):
    def __init__(self, *args, official, integer_source, parts, layers, build_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_model = deepcopy(self.model)
        self.official, self.integer_source = Path(official), Path(integer_source)
        self.parts, self.layers = set(parts), set(layers)
        if (
            not self.parts
            or not self.parts <= {"wq_a", "wkv"}
            or not self.layers
            or any(not 0 <= i < 40 for i in self.layers)
        ):
            raise ValueError("Select Q/KV parts and target layers 0-39")
        self.build_dir = Path(build_dir)
        self.specs, self.tasks, self.reports = {}, {}, []

    def spec(self, path):
        if path not in self.specs:
            self.specs[path] = json.loads((self.package / path).read_text())
        return self.specs[path]

    def plan(self):
        from tools.model.deepseek_v41.assembly import check_intermediate_liveness, compact_buffers

        rotation = block_rotation(self.integer_source)
        reference = Checkpoint(self.official)
        replacement_values = {}
        removed = 0
        calls = 0
        try:
            for rank in self.model["ranks"]:
                definitions = {d["name"]: d for d in rank["buffers"]}
                originals = {
                    d["name"]: d["prepared_tensor"]["name"]
                    for d in rank["buffers"]
                    if "prepared_tensor" in d
                }

                def ones(count):
                    name = f"precision.ones.{count}"
                    tensor = "native." + name
                    if name not in definitions:
                        if tensor not in self.payloads:
                            original = next(iter(self.transforms))
                            group = (
                                self.override_groups[self.transforms[original]]
                                if self.overlay
                                else self.old_index["weight_map"][original]
                            )
                            self.add_tensor(
                                tensor, np.ones(count, dtype="<f4"), dtype="F32", group=group
                            )
                        definition = dict(
                            name=name,
                            bytes=count * 4,
                            prepared_tensor=dict(name=tensor, dtype="F32", shape=[count]),
                        )
                        definitions[name] = definition
                        rank["buffers"].append(definition)
                    return name

                for packet in rank["packets"]:
                    producers = {}
                    obsolete = set()
                    body = packet["body"]
                    for index, step in enumerate(body):
                        if step["op"] != "kernel":
                            continue
                        spec = self.spec(step["manifest"])
                        if spec["op"] == "vector.quantize":
                            producers[step["buffers"][1]] = (step["buffers"][0], index)
                        if spec["op"] != "dense.projection":
                            continue
                        match = re.fullmatch(
                            r"w\.layers\.(\d+)\.attn\.(wq_a|wkv)\.weight\.0-(\d+)\.0-5120\.i8",
                            step["buffers"][1],
                        )
                        if not match:
                            continue
                        layer, part, n = match.groups()
                        layer, n = int(layer), int(n)
                        if layer not in self.layers or part not in self.parts:
                            continue
                        weight = definitions[step["buffers"][1]]
                        original = originals[weight["name"]]
                        if original not in replacement_values:
                            prefix = f"layers.{layer}"
                            norm = definitions[f"w.{prefix}.attn_norm.weight.0-5120.0-1.f32"]
                            gamma = reference.read(prefix + ".attn_norm.weight")
                            current = np.array(self.array(norm["prepared_tensor"]["name"])).reshape(
                                -1
                            )
                            canonical = reference_projection(
                                reference.matrix(prefix + ".attn." + part + ".weight"),
                                gamma,
                                current,
                                rotation,
                                query=part == "wq_a",
                            )
                            baseline = self.array(original).astype("i1").astype("f4")
                            scale = self.array(
                                definitions[step["buffers"][3]]["prepared_tensor"]["name"]
                            ).reshape(-1)
                            baseline *= scale[:, None]
                            energy = max(float(np.square(canonical, dtype="f8").sum()), 1e-30)
                            error = (
                                float(np.square(baseline - canonical, dtype="f8").sum()) / energy
                            )
                            if canonical.shape != (n, 5120) or error > 0.002:
                                raise ValueError(
                                    f"Reference basis differs for {prefix}.{part}: {error}"
                                )
                            values = bf16_parts(canonical)[0]
                            replacement_values[original] = values
                            self.reports.append(
                                dict(
                                    layer=layer,
                                    part=part,
                                    baseline_weight_nmse=error,
                                    old_bytes=baseline.size,
                                    new_bytes=values.nbytes,
                                )
                            )
                        if weight["prepared_tensor"]["name"] == original:
                            self.replace_tensor(
                                weight,
                                replacement_values[original],
                                dtype="BF16",
                                suffix=".bf16_reference",
                                group=f"weights/precision/layers-{layer // 4 * 4:02}-{layer // 4 * 4 + 3:02}.safetensors",
                            )
                        if step["buffers"][0] not in producers:
                            raise ValueError("Q/KV input has no BF16 quantization producer")
                        source, producer = producers[step["buffers"][0]]
                        for key in ("manifest", "uncompacted_manifest"):
                            if key not in step:
                                continue
                            path = step[key]
                            if path not in self.tasks:
                                self.tasks[path] = floating_projection_spec(self.spec(path))
                            placeholder = "@projection-" + str(list(self.tasks).index(path))
                            self.specs[placeholder] = self.tasks[path]
                            step[key] = placeholder
                        step["buffers"][0] = source
                        step["buffers"][2:4] = [ones(spec["parameters"]["_row_capacity"]), ones(n)]
                        obsolete.add(producer)
                        calls += 1
                    for index in sorted(obsolete, reverse=True):
                        step = body[index]
                        try:
                            check_intermediate_liveness(
                                body[index + 1 :],
                                step["buffers"][1:],
                                lambda s: self.spec(s["manifest"]),
                                self.model["batch"],
                                definitions,
                            )
                        except ValueError:
                            continue  # The other projection may still consume A8.
                        body.pop(index)
                        removed += 1
        finally:
            reference.close()
        if not calls:
            raise ValueError("No selected target projections")
        self.model = compact_buffers(self.model)
        self.prepare_overrides()
        previous = sum(
            d["bytes"] for d in self.original_model["ranks"][0]["buffers"] if "prepared_tensor" in d
        )
        current = sum(
            d["bytes"] for d in self.model["ranks"][0]["buffers"] if "prepared_tensor" in d
        )
        self.manifest["projection_precision"] = dict(
            parts=sorted(self.parts),
            layers=sorted(self.layers),
            method="official decoded FP8 + original norm/QuaRot/SmoothQuant basis, BF16 activation/weight",
            calls=calls,
            removed_quantizers=removed,
            projections=self.reports,
            per_rank_weight_delta_bytes=sum(r["new_bytes"] - r["old_bytes"] for r in self.reports),
            per_rank_bound_weight_delta_bytes=current - previous,
        )
        return self

    def build(self, workers=4):
        from tools.model.deepseek_v41.assembly import publish_package
        from tools.model.deepseek_v41.build import compile_task, kernel_rebuild_task

        source = self.build_dir / "source"
        if not source.exists():
            publish_package(self.original_model, self.package, source)
        elif json.loads((source / "model.json").read_text()) != self.original_model:
            raise ValueError("Precision build input changed")
        self.package = source
        tasks = [
            kernel_rebuild_task(spec, "@projection-" + str(i))
            for i, spec in enumerate(self.tasks.values())
        ]
        # Publication requires one source identity per factory, including
        # unchanged draft/vision consumers and offline row selections.
        from tools.model.deepseek_v41.publication import manifests

        for path in manifests(self.model):
            if path.startswith("@projection-"):
                continue
            spec = self.spec(path)
            if spec["op"] == "dense.projection":
                tasks.append(kernel_rebuild_task(spec, path))
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            compiled = dict(pool.map(compile_task, [(t, str(source)) for t in tasks]))
        for rank in self.model["ranks"]:
            for packet in rank["packets"]:
                for step in packet["body"] + packet["head"]:
                    for key in ("manifest", "uncompacted_manifest"):
                        if step.get(key) in compiled:
                            step[key] = compiled[step[key]]
        for step in self.model.get("vision", {}).get("body", []):
            if step["manifest"] in compiled:
                step["manifest"] = compiled[step["manifest"]]
        self.model["row_selections"] = {
            compiled.get(slot, slot): [
                dict(v, kernels=[compiled.get(k, k) for k in v["kernels"]]) for v in variants
            ]
            for slot, variants in self.model.get("row_selections", {}).items()
        }
        (self.build_dir / "candidate-plan.json").write_text(json.dumps(self.model))

    def export(self, workers):
        from tools.model.deepseek_v41.assembly import validate_rows

        super().export(workers)
        validate_rows(self.model, self.package, model_dir=self.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "source-execution",
        "official",
        "integer-source",
        "output",
        "execution-output",
        "build-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--parts", choices=["wq_a", "wkv"], nargs="+", default=["wq_a", "wkv"])
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(40)))
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    job = ProjectionRelease(
        args.source,
        args.output,
        source_execution=args.source_execution,
        execution_output=args.execution_output,
        official=args.official,
        integer_source=args.integer_source,
        parts=args.parts,
        layers=args.layers,
        build_dir=args.build_dir,
        overlay=args.overlay,
    ).plan()
    print(json.dumps(job.manifest["projection_precision"]), flush=True)
    job.build(args.workers)
    job.export(args.workers)


if __name__ == "__main__":
    main()
