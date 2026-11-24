"""Shared AOT row bindings and integer projections for independent encoders."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from orinfer_build.common.files import file_hash, write_json
from orinfer_build.common.gpu import gpu_lock
from orinfer_build.common.paths import ROOT, gpu_lock_path, work_path
from orinfer_build.compiler.compile import primitive
from orinfer_build.contracts.projection_policy import encoder_configs
from orinfer_build.packaging.compact import compact_package
from orinfer_build.packaging.frontend import prepare_tokenizer


def encoder_parser(description, cache_name):
    p = argparse.ArgumentParser(description=description)
    for name in ["source", "model-output", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--cache", type=Path, default=work_path("cache", "build", cache_name))
    p.add_argument("--engine", type=Path, default=ROOT / "target/release/orinfer")
    p.add_argument("--gpu-lock", type=Path, default=gpu_lock_path())
    p.add_argument("--gpu-lock-held", action="store_true")
    return p


def validate_row_choices(plan, expected):
    actual = {
        name
        for choices in (plan.get("row_execution") or {}).get("variants", {}).values()
        for choice in choices
        for name in choice["kernels"]
    }
    if actual != set(expected):
        raise ValueError(
            "Native plan lost measured row choices; rebuild the model library and execution package"
        )


class Projections:
    norm_epsilon = 1e-5

    def init_encoder(
        self,
        args,
        root,
        *,
        validate_config,
        capacity,
        patches,
        identity_key,
        identity_value,
        sign_widths,
    ):
        from orinfer_build.models.qwen35.compiler import Aot
        from orinfer_build.models.qwen35.storage import Weights
        from orinfer_build.quantization.e8p import basis
        from orinfer_build.quantization.vq import e8p_selector_table

        self.args, self.root = args, root
        self.config = json.loads((args.source / "config.json").read_text())
        validate_config(self.config, quantized=True)
        self.state = json.loads((args.source / "progress.json").read_text())
        if not self.state.get("complete"):
            raise ValueError("Quantization must finish before building")
        self.capacity, self.patches = capacity, patches
        self.w = Weights(args.source, root, packed_k=128)
        self.lock = None if args.gpu_lock_held else args.gpu_lock
        self.aot = Aot(
            root / "cache",
            args.output,
            args.cache,
            self.w.buffers,
            compile_lock=lambda: gpu_lock(self.lock),
        )
        self.aot.identity["weights"] = hashlib.sha256(self.state["contract"].encode()).hexdigest()
        self.aot.identity[identity_key] = str(identity_value)
        self.tuned = {}
        self.signs = {k: self.w.immutable(f"orinfer.rotation.k{k}.signs") for k in sign_widths}
        original_basis = self.w.array("orinfer.e8p_q4.basis")
        if not np.array_equal(original_basis, basis()):
            raise ValueError("Encoded E8P basis differs from the kernel codebook")
        self.book = self.w.constant("E8PBook", e8p_selector_table(original_basis))

    def publish_encoder(
        self,
        *,
        architecture,
        model_name,
        vocab,
        logits,
        weight_scope,
        config_signature,
        copy_files,
        frontend_files,
        spec_key,
        spec,
    ):
        total = self.w.finish()
        for name in copy_files:
            if (self.args.source / name).is_file():
                shutil.copyfile(self.args.source / name, self.root / name)
        prepare_tokenizer(self.args.source, self.root / "tokenizer.json")
        from orinfer_build.compiler.cuda.target import package_target

        pkg_target = package_target()
        metadata = dict(
            schema_version=2,
            target=pkg_target,
            model=model_name,
            chunk_tokens=self.capacity,
            max_context=self.capacity,
            vocab=vocab,
            toolchain=self.aot.identity,
            buffers=list(self.w.buffers.values()),
            reset_buffers=[],
            input="Input",
            token="Control",
            status="Control",
            logits=logits,
            position="Control",
            weight_bytes=total,
            weight_parameters=self.state["parameters"],
            weight_scope=weight_scope,
            **{spec_key: spec},
        )
        package = dict(
            target=pkg_target,
            architecture=architecture,
            compute_policy="int8_quality",
            config_signature=config_signature,
            prefill_profiles=[],
            kernels=self.aot.kernels,
            dynamic_batch_kernels=self.aot.dynamic,
            buffer_contracts=[
                {k: v for k, v in b.items() if k != "data"} for b in self.w.buffers.values()
            ],
            toolchain=self.aot.identity,
        )
        digest = compact_package(self.root / "cache", package, self.root / "cache/packages")
        write_json(
            self.root / "cache/model.json",
            dict(
                schema_version=1,
                architecture=architecture,
                compute_policy="int8_quality",
                execution_package=digest,
                buffer_scopes=self.w.scopes,
                metadata=metadata,
                frontend_assets={name: file_hash(self.root / name) for name in frontend_files},
            ),
        )
        subprocess.run([str(self.args.engine), "validate-model", str(self.root)], check=True)
        self.validate_published_rows()
        write_json(
            self.args.output / "result.json",
            dict(
                weight_bytes=total,
                parameters=self.state["parameters"],
                effective_weight_bits=total * 8 / self.state["parameters"],
                execution_package=digest,
            ),
        )

    def validate_published_rows(self):
        import json
        import subprocess

        result = subprocess.run(
            [str(self.args.engine), "plan-model", str(self.root)],
            capture_output=True,
            text=True,
            check=True,
        )
        plan = json.loads(result.stdout)["manifest"]
        selected = [k["name"] for k in self.aot.kernels if "/rows/" in k["name"]]
        validate_row_choices(plan, selected)
        write_json(self.args.output / "published-rows.json", plan["row_execution"])

    def emit(self, name, key, build, args, *, vision=False, divisor=1, dynamic=True):
        rows = (self.patches if vision else self.capacity) // divisor
        return self.aot.emit(
            self.kernel_prefix + "/" + name,
            self.kernel_prefix + "-" + key,
            build,
            args,
            rows,
            dynamic=dynamic,
            row_divisor=divisor,
        )

    def row_workspace(self):
        for name, shape in [
            ("RowRanges", [self.capacity, 2]),
            ("Segments", [128, 2]),
            ("SequenceCount", [1]),
        ]:
            self.w.fixed(name, "i32", shape)

    @staticmethod
    def row_spec():
        return dict(
            ranges="RowRanges",
            segments="Segments",
            count="SequenceCount",
            max_sequences=128,
            alignment=32,
        )

    def norm(self, name, source, dest, weight, hidden, *, vision=False, divisor=1):
        from kernels.cuda.model.embedding import rms_norm

        self.emit(
            name,
            f"rms-{hidden}-{self.norm_epsilon}",
            lambda: primitive(rms_norm, self.capacity, hidden, epsilon=self.norm_epsilon),
            dict(X=source, W=self.w.immutable(weight, cast="float16"), Y=dest),
            vision=vision,
            divisor=divisor,
        )

    def activation_input(self, name, source, k, *, compressed=True, vision=False, divisor=1):
        from kernels.cuda.model.embedding import quantize
        from kernels.cuda.model.rotation_a8 import rotate_activation_a8

        self.emit(
            name,
            f"a8-{k}-{compressed}-registers",
            lambda: (
                primitive(rotate_activation_a8, self.capacity, k, dynamic_rows=True, registers=True)
                if compressed
                else primitive(quantize, self.capacity, k)
            ),
            dict(X=source, Q="A8", S="AS", **(dict(Signs=self.signs[k]) if compressed else {})),
            vision=vision,
            divisor=divisor,
        )

    def attention(self, name, hidden, heads, kv_heads, bindings, *, causal=False, vision=False):
        from kernels.cuda.vision.encoder import attention

        options = dict(kv_heads=kv_heads, causal=causal, segmented=not vision)
        if not vision:
            options.update(bn=64, head_group=heads // kv_heads, stages=1, base2=True, full_row=True)
            bindings = dict(bindings, live_k=bindings["K"], live_v=bindings["V"])
        self.emit(
            name,
            f"attention-{hidden}-{heads}-{kv_heads}-{causal}-{vision}-grouped",
            lambda: primitive(attention, hidden, heads, **options),
            bindings,
            vision=vision,
        )

    def projection(
        self, name, weight, n, k, dest, *, vision=False, divisor=1, stride=None, offset=0
    ):
        compressed = (
            weight + ".packed" in self.w.index["weight_map"] or weight + ".packed" in self.w.buffers
        )
        suffix, kind = (".packed", "e8p_q4") if compressed else (".int8", "w8")
        data, scale = self.w.immutable(weight + suffix), self.w.immutable(weight + ".scale")
        capacity = (self.patches if vision else self.capacity) // divisor
        key = (kind, n, k, stride, capacity)

        if key not in self.tuned:
            import torch
            from orinfer_build.common.gpu import gpu_lock
            from orinfer_build.tuning.projection_tuning import ProjectionTuner

            with gpu_lock(self.lock):
                actual = dict(
                    W=torch.from_numpy(self.w.array(weight + suffix).copy()).cuda(),
                    WS=torch.from_numpy(self.w.array(weight + ".scale").copy()).cuda(),
                )
                if compressed:
                    actual["Book"] = torch.from_numpy(self.w.extra[self.book].copy()).cuda()
                tuner = ProjectionTuner(self.args.output / "tuning", self.args.cache / "tuning")
                identity = self.w.buffers[data]["data"]["sha256"]
                options = dict(
                    dynamic=True,
                    weights=actual,
                    weight_identity=identity,
                    measure_rows=sorted(
                        {
                            min(capacity, rows)
                            for rows in (32, 96, 128, 192, 512, 2048, 8192, capacity)
                        }
                    ),
                    objective="total_latency",
                    output_stride=stride,
                    packed_k=128 if compressed else 0,
                    candidate_configs=encoder_configs(kind),
                )
                kernel, config = tuner.compile(kind, capacity, n, k, "float16", **options)
                choices = []
                if not vision:
                    # Short packets cross tile-count boundaries between the coarse
                    # fallback samples. Measure each aligned packet through 512
                    # rows without changing the global fallback objective.
                    local_options = dict(
                        options,
                        measure_rows=sorted(
                            set(options["measure_rows"])
                            | set(range(32, min(capacity, 512) + 1, 32))
                        ),
                    )
                    choices = [
                        (suffix, choice, compiled)
                        for suffix, choice, compiled in tuner.compile_choices(
                            kind, capacity, n, k, "float16", **local_options
                        )
                        if choice != config
                    ]
                self.tuned[key] = (kernel, choices)
                write_json(
                    self.args.output / f"projection-{kind}-{n}-{k}-{stride}-{capacity}.json",
                    dict(
                        default=config,
                        selections=[
                            dict(range=suffix, config=choice) for suffix, choice, _ in choices
                        ],
                    ),
                )

        target = dict(kind="buffer_slice", name=dest, offset=offset * 2) if offset else dest
        bindings = (
            dict(A="A8", W=data, S=scale, AS="AS", Out=target, Book=self.book)
            if compressed
            else dict(A="A8", Weight=data, WeightScale=scale, TokenScale="AS", Output=target)
        )
        kernel, choices = self.tuned[key]
        self.emit(
            name, f"projection-{key}", lambda: kernel, bindings, vision=vision, divisor=divisor
        )
        if not vision:
            for suffix, config, compiled in choices:
                import json

                self.emit(
                    name + suffix,
                    f"projection-{key}-" + json.dumps(config, sort_keys=True),
                    lambda compiled=compiled: compiled,
                    bindings,
                )
