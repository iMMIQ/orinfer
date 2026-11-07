"""Native catalog factoring must preserve real host parameter order and launches."""

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orinfer_build.contracts.catalog import (
    build_library,
    construction_contracts,
    expand,
    factor,
    rust_source,
)
from orinfer_build.contracts.binding_templates import (
    factor as templates,
    expand as expand_templates,
)


class CatalogTests(unittest.TestCase):
    def test_separate_qwen_history_survives_native_construction_filtering(self):
        package = self.package()
        history = copy.deepcopy(package["kernels"][0])
        history["name"] = "qwen_rows/state/layer0/history"
        package["kernels"].append(history)
        factored, bindings = factor(package)
        marked = {b["name"] for b in bindings if b["plan_contract"]}
        self.assertIn(history["name"], marked)
        self.assertNotIn("layer1/projection", marked)
        _, _, contracts = expand(factored["catalog"], bindings)
        self.assertEqual(set(contracts), construction_contracts(package))
        self.assertNotIn(history["name"], package["plan_kernel_contracts"])

    def package(self):
        identity = dict(file="assets.safetensors", tensor="a" * 64, sha256="a" * 64)
        kernel = dict(
            name="layer0/projection",
            module=identity,
            source=identity,
            host_abi=identity,
            symbol="main",
            grid=[128, 1, 1],
            block=[128, 1, 1],
            shared_memory_bytes=1024,
            cooperative=False,
            args=[dict(kind="buffer", name="Weight0"), dict(kind="i32", value=128)],
            readonly_arguments=[True, False],
        )
        second = copy.deepcopy(kernel)
        second["name"] = "layer1/projection"
        second["args"][0]["name"] = "Weight1"
        return dict(
            kernels=[kernel, second],
            plan_kernel_contracts=[kernel["name"]],
            dynamic_batch_kernels=[
                dict(
                    name=kernel["name"],
                    capacity=128,
                    grid=[
                        dict(op="rows"),
                        dict(op="constant", value=1),
                        dict(op="constant", value=1),
                    ],
                    arguments=[dict(index=1, value=dict(op="rows"))],
                )
            ],
        )

    def test_tensor_roles_are_native_and_definitions_are_shared(self):
        package = self.package()
        factored, bindings = factor(package)
        self.assertNotIn("kernels", factored)
        self.assertNotIn("dynamic_batch_kernels", factored)
        self.assertNotIn("plan_kernel_contracts", factored)
        self.assertEqual(len(factored["catalog"]["assets"]), 1)
        self.assertEqual(len(factored["catalog"]["definitions"]), 1)
        kernels, dynamic, contracts = expand(factored["catalog"], bindings)
        self.assertEqual(kernels, package["kernels"])
        self.assertEqual(dynamic, package["dynamic_batch_kernels"])
        self.assertEqual(contracts, package["plan_kernel_contracts"])
        self.assertIn('"Weight1"', rust_source(factored["catalog"], bindings))

    def test_abi_launch_or_access_changes_cannot_share_a_definition(self):
        for field, value in (
            ("block", [64, 1, 1]),
            ("shared_memory_bytes", 0),
            ("readonly_arguments", [False, False]),
            ("symbol", "other"),
        ):
            p = self.package()
            p["kernels"][1][field] = value
            factored, _ = factor(p)
            self.assertEqual(len(factored["catalog"]["definitions"]), 2)
        p = self.package()
        p["kernels"][1]["args"][1] = dict(kind="u32", value=128)
        factored, bindings = factor(p)
        self.assertEqual(len(factored["catalog"]["definitions"]), 2)
        bindings[1]["args"][1]["kind"] = "i32"
        with self.assertRaisesRegex(ValueError, "compiled ABI"):
            expand(factored["catalog"], bindings)

    def test_native_mapping_identity_covers_roles_offsets_and_dynamic_launch(self):
        original = self.package()
        before, _ = factor(original)
        for change in ("role", "offset", "grid", "dynamic"):
            p = self.package()
            if change == "role":
                p["kernels"][0]["args"][0]["name"] = "WrongWeight"
            elif change == "offset":
                p["kernels"][0]["args"][0] = dict(kind="buffer_slice", name="Weight0", offset=64)
            elif change == "grid":
                p["kernels"][0]["grid"][0] = 64
            else:
                p["dynamic_batch_kernels"][0]["capacity"] = 64
            after, _ = factor(p)
            self.assertNotEqual(
                before["catalog"]["binding_signature"], after["catalog"]["binding_signature"]
            )

    def test_duplicate_or_missing_contract_is_rejected(self):
        p = self.package()
        p["kernels"].append(p["kernels"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            factor(p)
        p = self.package()
        p["plan_kernel_contracts"].append("missing")
        with self.assertRaisesRegex(ValueError, "Missing"):
            factor(p)

    def test_layers_share_specialization_tables_with_exact_parameter_maps(self):
        bindings = []
        for layer in range(4):
            for suffix, width in [("", 1024), ("/rows/1-1", 1), ("/rows/2-2", 2)]:
                bindings.append(
                    dict(
                        name=f"target/layer{layer}/projection{suffix}",
                        definition=width,
                        grid=[width, 1, 1],
                        args=[
                            dict(kind="buffer", name=f"Weight{layer}"),
                            dict(kind="buffer_slice", name=f"State{layer}", offset=layer * 64),
                            dict(kind="i32", value=-width),
                            dict(kind="f32", value=-0.0),
                        ],
                        dynamic=None if width == 1 else 0,
                        plan_contract=True,
                    )
                )
        registry = templates(bindings)
        self.assertEqual(len(registry["templates"]), 1)
        self.assertEqual(len(registry["choices"]), 1)
        self.assertEqual(len(registry["choices"][0]), 3)
        self.assertEqual(len(registry["templates"][0]["layers"]), 4)
        self.assertEqual(
            {b["name"]: b for b in expand_templates(registry)},
            {b["name"]: b for b in bindings},
        )
        # A different aliasing pattern must get a separate parameter selector.
        bindings[3]["args"][0] = bindings[3]["args"][1]
        self.assertEqual(len(templates(bindings)["choices"]), 2)

    def test_singletons_and_leading_zero_layer_names_are_preserved(self):
        _, bindings = factor(self.package())
        for name in ("vision/end/k0", "x/layer001/projection", "x/layer2end/projection"):
            binding = dict(bindings[0], name=name)
            self.assertEqual(expand_templates(templates([binding])), [binding])

    def test_template_export_order_does_not_change_catalog_or_binding_identity(self):
        original = self.package()
        before, bindings = factor(original)
        reordered = copy.deepcopy(original)
        reordered["kernels"].reverse()
        reordered["dynamic_batch_kernels"].reverse()
        after, again = factor(reordered)
        self.assertEqual(before, after)
        self.assertEqual(bindings, again)
        # Native export traverses templates, not the compiler input order.
        reconstructed = expand_templates(templates(bindings))
        kernels, dynamic, contracts = expand(before["catalog"], reconstructed)
        third, third_bindings = factor(
            dict(kernels=kernels, dynamic_batch_kernels=dynamic, plan_kernel_contracts=contracts)
        )
        self.assertEqual(third, before)
        self.assertEqual(third_bindings, bindings)

    def test_native_rebuild_preserves_hardlinked_source_package(self):
        package, bindings = factor(self.package())
        package.update(architecture="test", compute_policy="int8_quality")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / "before/lib", root / "after/lib"
            before.mkdir(parents=True)
            after.mkdir(parents=True)
            for name in ("bindings.rs", "BUILD.txt", "model.so"):
                (before / name).write_bytes(b"immutable original")
                os.link(before / name, after / name)
            build_root = root / "builds/test-workspace"
            target = build_root / "native-cargo/release/liborinfer_models.so"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"rebuilt library")
            with (
                patch("orinfer_build.contracts.catalog.ROOT", root),
                patch("orinfer_build.contracts.catalog.work_path", return_value=build_root),
                patch("orinfer_build.contracts.catalog.subprocess.run"),
                patch("orinfer_build.packaging.package.execution_identity", return_value={}),
            ):
                build_library(package, bindings, after.parent)
            for name in ("bindings.rs", "BUILD.txt", "model.so"):
                self.assertEqual((before / name).read_bytes(), b"immutable original")
                self.assertFalse(os.path.samefile(before / name, after / name))
            self.assertEqual((after / "model.so").read_bytes(), b"rebuilt library")
