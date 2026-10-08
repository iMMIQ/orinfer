"""Attach a checkpoint-exact BF16/FP16 vision encoder to an existing W4 text plan.

Text weights are hard-linked without changing representation. Vision weights
come from the original BF16 checkpoint; all persistent bytes count.
"""

import argparse
import json
import math
import os
from pathlib import Path
import torch
from tools.operators.common import configure, write_json
from kernels.vision import bridge
from tools.vision.config import validate_adapter
from tools.vision.plan import EncoderBuilder


class Assembler(EncoderBuilder):
    def __init__(self, text_manifest, checkpoint, output, max_patches=32768, vision_dtype="f16"):
        self.output = output
        self.exports = {}
        self.dtype = vision_dtype
        self.kernel_dtype = "bfloat16" if vision_dtype == "bf16" else "float16"
        self.config = json.loads((checkpoint / "config.json").read_text())
        self.vision = self.config["vision_config"]
        self.text = self.config["text_config"]
        self.manifest = json.loads(text_manifest.read_text())
        self.max_patches = max_patches
        m = self.manifest
        if m.get("vision") is not None:
            raise ValueError("Text manifest already has a vision adapter")
        validate_adapter(self.config, m)
        root = text_manifest.resolve().parent
        copied = {}

        def adopt(identity):
            if identity["file"] in copied:
                return copied[identity["file"]].copy()
            src = root / identity["file"]
            dst = output / "text" / identity["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)
            result = {"file": "text/" + identity["file"], "sha256": identity["sha256"]}
            copied[identity["file"]] = result
            return result.copy()

        for b in m["buffers"]:
            if b.get("data"):
                b["data"] = adopt(b["data"])
        for k in m["kernels"]:
            for key in ("module", "source", "host_abi"):
                k[key] = adopt(k[key])
        self.buffers = {b["name"]: b for b in m["buffers"]}
        self.source = checkpoint / "model.safetensors"

    def buffer(self, name, shape, dtype="f16", data=None, weight=False):
        b = {
            "name": name,
            "dtype": dtype,
            "shape": list(shape),
            "layout": "contiguous",
            "alignment": 256,
            "access": "read" if weight else "read_write",
            "data": None,
        }
        if data is not None:
            p = self.output / "weights" / f"{name}.bin"
            p.parent.mkdir(exist_ok=True)
            data.contiguous().view(torch.uint8).numpy().tofile(p)
            b["data"] = self.identity(p)
        self.manifest["buffers"].append(b)
        self.buffers[name] = b

    def build(self):
        m = self.manifest
        out = self.vision["out_hidden_size"]
        self.max_features = m["max_context"]
        parameters = self.build_encoder()
        self.compile(
            "embedding_features",
            lambda: bridge.embedding_features(m["vocab"], out, m["max_context"], m["max_context"]),
        )
        self.compile(
            "full_prepare_mrope",
            lambda: bridge.full_prepare_mrope(
                math.ceil(m["max_context"] / 128),
                m["max_context"],
                tuple(self.text["rope_parameters"]["mrope_section"]),
                max_position=self.buffers["Rotary"]["shape"][0],
            ),
        )
        # Replace ABI bindings, preserving all text phase names and layer order.
        embeddings = preparations = 0
        for index, k in enumerate(m["kernels"]):
            args = {a["name"] for a in k["args"] if a["kind"] == "buffer"}
            rows = next((a["value"] for a in k["args"] if a["kind"] == "i32"), None)
            if "Embedding_P" in args:
                embeddings += 1
                m["kernels"][index] = self.kernel(
                    "embedding_features",
                    dict(
                        P="Embedding_P",
                        S="Embedding_S",
                        Z="Embedding_Z",
                        I="Input",
                        Step=m["position"],
                        Index="FeatureIndex",
                        Features="Features",
                        Y="Hidden",
                    ),
                    rows,
                    k["name"],
                )
            elif "FullX" in args and "Rotary" in args:
                preparations += 1
                layer = next(a.removesuffix("_QWeight") for a in args if a.endswith("_QWeight"))
                m["kernels"][index] = self.kernel(
                    "full_prepare_mrope",
                    dict(
                        X="FullX",
                        WQ=layer + "_QWeight",
                        WK=layer + "_KWeight",
                        Cache="Rotary",
                        Req="Req",
                        Pos="Positions",
                        Pages="Pages",
                        Status="PrepareStatus",
                        MRope="MRopePositions",
                        Q="FullQ",
                        Gate="FullGate",
                        K=layer + "_KPages",
                        V=layer + "_VPages",
                    ),
                    rows,
                    k["name"],
                )
        if embeddings == 0 or preparations == 0:
            raise ValueError("Text plan is missing embedding or full-attention bridge kernels")
        m["weight_parameters"] += parameters
        m["weight_bytes"] = sum(
            math.prod(b["shape"])
            * {"f16": 2, "bf16": 2, "f32": 4, "i32": 4, "u8": 1, "i8": 1}[b["dtype"]]
            for b in m["buffers"]
            if b["access"] == "read"
        )
        m["weight_scope"] += (
            f"; original BF16 vision weights stored {self.dtype}, counted in total, images/multi-image adapter"
        )
        write_json(self.output / "model.json", m)
        print(
            "Vision parameters",
            parameters,
            "total bits",
            8 * m["weight_bytes"] / m["weight_parameters"],
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--text-model", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--max-patches", type=int, choices=(1024, 2048, 4096, 8192, 16384, 32768), default=32768
    )
    ap.add_argument("--vision-dtype", choices=("bf16", "f16"), default="f16")
    args = ap.parse_args()
    configure()
    if (args.output / "model.json").exists():
        raise FileExistsError(args.output)
    Assembler(
        args.text_model, args.checkpoint, args.output, args.max_patches, args.vision_dtype
    ).build()


if __name__ == "__main__":
    main()
