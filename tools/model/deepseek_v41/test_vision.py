"""CPU vision-bank layout checks, independent of NPU execution."""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.model.deepseek_v41.vision import export, tensor_shapes
from tools.model.deepseek_v41.vision_attach import attach, rebind_masks
from tools.model.deepseek_v41.vision_build import program
from tools.model.row_contract import expression, resolve
from tools.model.safetensors_source import local_header


class VisionExportTests(unittest.TestCase):
    def test_frame_boundaries_preserve_layer_state_and_the_final_aligner(self):
        config = dict(
            text_config=dict(hidden_size=5120),
            image_token_id=129264,
            vision_config=dict(
                hidden_size=1024,
                intermediate_size=2816,
                patch_size=14,
                num_attention_heads=16,
                num_hidden_layers=8,
                downsample_ratio=3,
                min_pixels=295936,
                max_image_tokens=1024,
                rope_theta=10000,
            ),
        )
        contracts = {}

        def describe(module, name, arguments, ports, capacity):
            key = str(len(contracts))
            contracts[key] = (module, name, arguments)
            return key

        plan = program(config, describe)
        self.assertEqual(plan["segment_ends"][-1], len(plan["body"]))
        for end in plan["segment_ends"][:-1]:
            previous = plan["body"][end - 1]
            module, name, arguments = contracts[previous["manifest"]]
            self.assertEqual((module, name), ("vision", "epilogue"))
            self.assertTrue(arguments["residual"])
            self.assertEqual(previous["buffers"][-1], "vision.h0")
            self.assertEqual(plan["body"][end]["buffers"][0], "vision.h0")
        self.assertEqual(plan["body"][-1]["buffers"][-1], "vision.output")

    def test_bf16_payload_and_target_basis_are_preserved_with_only_patch_k_padding(self):
        config = dict(
            quantization_config={"quant_method": "ascend"},
            text_config={"hidden_size": 4},
            vision_config=dict(
                hidden_size=4,
                intermediate_size=8,
                patch_size=2,
                downsample_ratio=3,
                num_hidden_layers=1,
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "export"
            source.mkdir()
            shapes = tensor_shapes(config)
            header, payload = {}, bytearray()
            original = {}
            for name, shape in shapes.items():
                elements = 1
                for dim in shape:
                    elements *= dim
                data = b"\x80\x3f" * elements
                original[name] = data
                begin = len(payload)
                payload.extend(data)
                header[name] = dict(dtype="BF16", shape=shape, data_offsets=[begin, len(payload)])
            name = "layers.0.ffn.gate.bias_vl"
            begin = len(payload)
            payload.extend(struct.pack("<4f", 0.1, 0.2, -0.3, 0.4))
            header[name] = dict(dtype="F32", shape=[4], data_offsets=[begin, len(payload)])
            encoded = json.dumps(header).encode()
            data = struct.pack("<Q", len(encoded)) + encoded + payload
            (source / "bank.safetensors").write_bytes(data)
            (source / "config.json").write_text(json.dumps(config))
            (source / "quant_model_weights.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(header, "bank.safetensors")})
            )
            manifest = export(source, output)
            bank = output / "vision.safetensors"
            start, actual = local_header(bank)
            self.assertEqual(manifest["target_basis"], "ascend_preprocessed")
            self.assertEqual(manifest["patch_reduction"], 128)
            self.assertEqual(manifest["value_compute"]["dtype"], "f16")
            self.assertAlmostEqual(manifest["value_compute"]["maximum"], 5.1)
            self.assertEqual(manifest["sha256"], hashlib.sha256(bank.read_bytes()).hexdigest())
            self.assertEqual((source / "bank.safetensors").read_bytes(), data)
            with bank.open("rb") as file:
                for name, expected in original.items():
                    first, last = actual[name]["data_offsets"]
                    file.seek(start + first)
                    value = file.read(last - first)
                    if name == "vision.patch_embed.proj.weight":
                        self.assertEqual(actual[name]["shape"], [4, 128])
                        self.assertEqual(value, (b"\x80\x3f" * 12 + bytes(232)) * 4)
                    else:
                        self.assertEqual(value, expected)
            with self.assertRaises(ValueError):
                export(source, output)
            config["quantization_config"]["quant_method"] = "original"
            (source / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "target basis"):
                export(source, Path(temp) / "incorrect-basis")
            config["quantization_config"]["quant_method"] = "ascend"
            (source / "config.json").write_text(json.dumps(config))
            begin = 8 + len(encoded) + header["vision.blocks.0.norm1.weight"]["data_offsets"][0]
            for label, value, message in [
                ("overflow", b"\x00\x47", "FP16 compute range"),
                ("nonfinite", b"\xc0\x7f", "Nonfinite"),
            ]:
                changed = bytearray(data)
                changed[begin : begin + 8] = value * 4
                (source / "bank.safetensors").write_bytes(changed)
                with self.assertRaisesRegex(ValueError, message):
                    export(source, Path(temp) / label)
                self.assertFalse((Path(temp) / label).exists())
                self.assertEqual(list(Path(temp).glob(label + ".*")), [])


class VisionPlanTests(unittest.TestCase):
    def test_encoder_refresh_preserves_language_and_original_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            spec = dict(layout="deepseek_v41", hidden=5120)
            rank = dict(rank=0, buffers=[], packets=[], immutable="text")
            original = json.dumps(dict(vision=dict(spec=spec), ranks=[rank]))
            (old / "model.json").write_text(original)
            (old / "text.so").write_bytes(b"existing text arithmetic")
            (new / "encoder.so").write_bytes(b"replacement encoder")
            (new / "vision.json").write_text(
                json.dumps(dict(spec=spec, body=[dict(manifest="encoder.so")]))
            )
            output = attach(old / "model.json", new / "vision.json", root / "output", 1)
            result = json.loads(output.read_text())
            self.assertEqual(result["ranks"], [rank])
            self.assertEqual((old / "model.json").read_text(), original)
            self.assertTrue((output.parent / "text.so").samefile(old / "text.so"))
            self.assertTrue(
                (output.parent / result["vision"]["body"][0]["manifest"]).samefile(
                    new / "encoder.so"
                )
            )
            (new / "vision.json").write_text(
                json.dumps(dict(spec=dict(spec, hidden=4096), body=[]))
            )
            with self.assertRaisesRegex(ValueError, "same frontend"):
                attach(old / "model.json", new / "vision.json", root / "invalid", 1)
            self.assertFalse((root / "invalid").exists())

    def test_source_and_decoder_masks_follow_sequence_views_on_all_ranks(self):
        from copy import deepcopy

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = dict(
                op="routing.route",
                parameters=dict(vision_bias=True),
                operands=[dict(shape=[256, 384])],
            )
            (root / "route.json").write_text(json.dumps(spec))
            step = dict(
                op="kernel",
                manifest="route.json",
                buffers=["logits", "bias", "indices", "weights", "vision_bias", "old_mask"],
                row_expression=expression("(m+7)//8"),
            )
            packet = dict(
                rows=2048,
                coordinates=dict(base_cos="source.base_cos"),
                decoder_coordinates=dict(base_cos="decoder.base_cos"),
                body=[deepcopy(step), dict(op="decoder_rows"), deepcopy(step)],
                head=[],
            )
            model = dict(
                ranks=[dict(rank=i, buffers=[], packets=[deepcopy(packet)]) for i in range(8)]
            )
            rebind_masks(model, root)
            for rank in model["ranks"]:
                buffers = {b["name"]: b for b in rank["buffers"]}
                packet = rank["packets"][0]
                self.assertNotEqual(
                    packet["coordinates"]["image_rows"], packet["decoder_coordinates"]["image_rows"]
                )
                for step, domain in [(packet["body"][0], "source"), (packet["body"][2], "decoder")]:
                    view = buffers[step["buffers"][-1]]
                    self.assertEqual(view["view"]["parent"], domain + ".image_rows")
                    self.assertEqual(buffers[view["view"]["parent"]]["bytes"], 2048 * 4)
                    self.assertEqual(view["bytes"], 256 * 4)
                    for rows in [129, 512, 960, 2048]:
                        self.assertEqual(
                            resolve(view["view"]["offset_expression"], rows),
                            ((rows + 7) // 8) * rank["rank"] * 4,
                        )


if __name__ == "__main__":
    unittest.main()
