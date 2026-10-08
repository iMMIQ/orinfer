"""Attach native one-layer Qwen3_5 MTP to a causal small-M target plan.

Embedding/head and target weights are shared. Shifted inputs use true target
final hidden states during warm/refresh and previous MTP final hidden while
drafting. Target verification supports exact speculative sampling and images.
All draft parameter bytes are accounted.
"""

import argparse
import importlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from tools.model.build import Builder
from tools.operators.common import configure, write_json
from tools.operators.abi import parse_host
from kernels.model import control, speculation
from kernels.model.w4_small_m import w4_small_m
from kernels.model.attention_decode_staged import paged_attention_partials_gqa_staged
from kernels.projections.candidates import fp16_gemm


class Assembler(Builder):
    def __init__(self, source, weights, output, verification_tokens):
        self.out = output
        self.manifest = json.loads(source.read_text())
        self.buffers, self.kernels = self.manifest["buffers"], self.manifest["kernels"]
        self.exports = {}
        self.vision_exports = {}
        self.prefill_tokens, self.i8_grid_order = 512, "nfirst"
        self.reuse_aot, self.reuse_weights = None, None
        self.dense_u4 = False
        self.reset = self.manifest["reset_buffers"]
        self.H = next(b["shape"][1] for b in self.buffers if b["name"] == "Hidden")
        self.F = next(b["shape"][1] for b in self.buffers if b["name"] == "Activated")
        self.V, self.context = self.manifest["vocab"], self.manifest["max_context"]
        self.pages = next(b["shape"][1] for b in self.buffers if b["name"] == "Pages")
        if self.H != 5120 or self.F != 17408 or self.manifest.get("mtp"):
            raise ValueError("Expected current Qwen3_5 dense adapter without MTP")
        self.verification = json.loads((source.parent / "verification-plans.json").read_text())
        if verification_tokens not in {p["tokens"] for p in self.verification["plans"]}:
            raise ValueError("Default verification shape is absent")
        self.verification_tokens = verification_tokens
        adopted = {}

        def adopt(identity):
            if identity["file"] not in adopted:
                old = source.parent / identity["file"]
                relative = Path("target") / identity["file"]
                new = output / relative
                new.parent.mkdir(parents=True, exist_ok=True)
                os.link(old, new)
                adopted[identity["file"]] = dict(file=str(relative), sha256=identity["sha256"])
            return adopted[identity["file"]].copy()

        for b in self.buffers:
            if b.get("data"):
                b["data"] = adopt(b["data"])
        for k in self.kernels:
            for key in ("module", "source", "host_abi"):
                k[key] = adopt(k[key])
        kernels = {k["name"]: k for k in self.kernels}
        if self.manifest.get("vision"):
            vision = self.manifest["vision"]
            for k in self.kernels:
                names = {a.get("name") for a in k["args"]}
                name = (
                    "embedding"
                    if vision["feature_index"] in names
                    else "fullprepare"
                    if vision["mrope_positions"] in names
                    else None
                )
                if name and name not in self.vision_exports:
                    abi = parse_host((output / k["host_abi"]["file"]).read_text())
                    assert len(abi) == 1
                    self.vision_exports[name] = {
                        **abi[0],
                        **{key: k[key] for key in ("module", "source", "host_abi")},
                    }
        for operation in self.manifest["programs"]["head_m512"]:
            if operation["kind"] != "kernel":
                continue
            k = kernels[operation["name"]]
            buffers = {a["name"] for a in k["args"] if a["kind"] == "buffer"}
            name = (
                "topk"
                if "PV" in buffers and "Logits" in buffers
                else ("topkmerge" if "TokenStatus" in buffers else None)
            )
            if name:
                abi = parse_host((output / k["host_abi"]["file"]).read_text())
                assert len(abi) == 1
                self.exports[name] = {
                    **abi[0],
                    **{key: k[key] for key in ("module", "source", "host_abi")},
                }
        self.weight_root = weights
        self.weight_report = json.loads((weights / "weights.json").read_text())
        self.records = {w["tensor"]: w for w in self.weight_report["weights"]}
        self.weight_formats = {}

    def norm_weight(self, name, key):
        record = self.records[key]
        array = np.fromfile(self.weight_root / record["data"]["file"], dtype=np.float16).reshape(
            record["shape"]
        )
        self.buffer(name, "f16", array.shape, torch.from_numpy(array), weight=True)

    def projection_weight(self, name, keys):
        records = [self.records[key] for key in keys]
        if len({r["dtype"] for r in records}) != 1:
            raise ValueError("Mixed projection formats are not supported")
        shapes = [r["shape"] for r in records]
        n, k = sum(s[0] for s in shapes), shapes[0][1]
        assert all(s[1] == k for s in shapes)
        self.weight_formats[name] = records[0]["dtype"]
        if records[0]["dtype"] == "f16":
            arrays = [
                np.fromfile(self.weight_root / r["data"]["file"], dtype=np.float16).reshape(
                    r["shape"]
                )
                for r in records
            ]
            array = np.concatenate(arrays, axis=0)
            self.buffer(name, "f16", array.shape, torch.from_numpy(array), weight=True)
        else:
            for kind, dtype, shape, runtime_dtype in (
                ("P", np.int32, (n // 64, k // 128, 128, 8), "i32"),
                ("S", np.float16, (n, k // 128), "f16"),
                ("Z", np.int8, (n, k // 128), "i8"),
            ):
                arrays = []
                for r in records:
                    rs = r["packed_shape"] if kind == "P" else (r["shape"][0], k // 128)
                    arrays.append(
                        np.fromfile(
                            self.weight_root / r["data"][kind]["file"], dtype=dtype
                        ).reshape(rs)
                    )
                array = np.concatenate(arrays, axis=0)
                assert array.shape == shape
                self.buffer(
                    name + "_" + kind, runtime_dtype, shape, torch.from_numpy(array), weight=True
                )
                if kind == "P":
                    self.buffers[-1]["layout"] = "u4_warp_n64_k128_mma_f16"
        return n, k

    def build(self):
        def op(n):
            return importlib.import_module("kernels.operators.op" + n)

        sizes = {}
        for name, keys in (
            ("MtpFC", ["mtp.fc.weight"]),
            ("MtpIn", [f"mtp.layers.0.self_attn.{p}_proj.weight" for p in ("q", "k", "v")]),
            ("MtpOut", ["mtp.layers.0.self_attn.o_proj.weight"]),
            ("MtpGateUp", [f"mtp.layers.0.mlp.{p}_proj.weight" for p in ("gate", "up")]),
            ("MtpDown", ["mtp.layers.0.mlp.down_proj.weight"]),
        ):
            sizes[name] = self.projection_weight(name, keys)
        for name, key in (
            ("MtpEmbeddingWeight", "mtp.pre_fc_norm_embedding.weight"),
            ("MtpHiddenWeight", "mtp.pre_fc_norm_hidden.weight"),
            ("MtpPreWeight", "mtp.layers.0.input_layernorm.weight"),
            ("MtpPostWeight", "mtp.layers.0.post_attention_layernorm.weight"),
            ("MtpQWeight", "mtp.layers.0.self_attn.q_norm.weight"),
            ("MtpKWeight", "mtp.layers.0.self_attn.k_norm.weight"),
            ("MtpFinalWeight", "mtp.norm.weight"),
        ):
            self.norm_weight(name, key)
        warm_sizes = [1, 2, 3, 4, 8, 16]
        maximum = max(warm_sizes)
        vision = self.manifest.get("vision")
        if vision:
            self.buffer("MtpFeatureIndex", "i32", (self.context,))
        for name, dtype, shape, reset in (
            ("MtpTargetHidden", "f16", (self.context, self.H), False),
            ("MtpInput", "i32", (maximum,), False),
            ("MtpStep", "i32", (1,), True),
            ("MtpSeqLength", "i32", (1,), False),
            ("MtpPositions", "i32", (maximum,), False),
            ("MtpEmbedding", "f16", (maximum, self.H), False),
            ("MtpCondition", "f16", (maximum, self.H), False),
            ("MtpConcat", "f16", (maximum, 2 * self.H), False),
            ("MtpHidden", "f16", (maximum, self.H), False),
            ("MtpNorm", "f16", (maximum, self.H), False),
            ("MtpR0", "f32", (maximum, self.H), False),
            ("MtpR1", "f32", (maximum, self.H), False),
            ("MtpFullX", "f16", (maximum, 14336), False),
            ("MtpFullQ", "f16", (maximum, 24, 256), False),
            ("MtpGate", "f16", (maximum, 24, 256), False),
            ("MtpKPages", "f16", (self.pages, 128, 4, 256), True),
            ("MtpVPages", "f16", (self.pages, 128, 4, 256), True),
            ("MtpAttM", "f32", (maximum, 24, 8), False),
            ("MtpAttL", "f32", (maximum, 24, 8), False),
            ("MtpAttO", "f32", (maximum, 24, 8, 256), False),
            ("MtpMixer", "f16", (maximum, 6144), False),
            ("MtpMix", "f16", (maximum, self.H), False),
            ("MtpGateUpResult", "f16", (maximum, 2 * self.F), False),
            ("MtpActivated", "f16", (maximum, self.F), False),
            ("MtpPartial", "f32", (8, maximum, self.H), False),
            ("MtpLastHidden", "f16", (1, self.H), False),
            ("MtpLogits", "f32", (1, self.V), False),
            ("MtpPV", "f32", (1, math.ceil(self.V / 4096), 1), False),
            ("MtpPI", "i32", (1, math.ceil(self.V / 4096), 1), False),
            ("MtpBad", "i32", (1, math.ceil(self.V / 4096)), False),
            ("MtpValues", "f32", (1, 1), False),
            ("MtpIDs", "i32", (1, 1), False),
            ("MtpToken", "i32", (1,), False),
            ("MtpStatus", "i32", (1,), False),
            ("MtpPrepareStatus", "i32", (1,), True),
        ):
            self.buffer(name, dtype, shape, reset=reset)
        self.compile("mtp_capture", lambda: speculation.capture_target_hidden(self.H, self.context))
        if vision:
            self.exports["mtp_embedding"] = self.vision_exports["embedding"]
        else:
            self.compile(
                "mtp_embedding",
                lambda: op("01_embedding").embedding_u4(vocab=self.V, hidden=self.H),
            )
        self.compile("mtp_concat", lambda: speculation.mtp_norm_concat(self.H))
        self.compile("mtp_swiglu", lambda: op("04_swiglu").swiglu())
        if vision:
            self.exports["mtp_fullprepare"] = self.vision_exports["fullprepare"]
        else:
            self.compile(
                "mtp_fullprepare",
                lambda: op("20_full_prepare").full_prepare(
                    1, self.pages, self.pages, max_position=self.context
                ),
            )
        self.compile(
            "mtp_attentionmerge",
            lambda: op("28_attention_split_merge").attention_split_merge(splits=8),
        )
        self.compile(
            "mtp_head", lambda: w4_small_m(1, self.V, self.H, output_dtype="float32", TILE_N=128)
        )
        capture_plans = []
        for rows in sorted({1, *[p["chunk_tokens"] for p in self.manifest["prefill_plans"]]}):
            p = []
            self.op(
                p,
                "mtp_capture",
                dict(
                    X="Hidden",
                    Residual="R0",
                    Weight="FinalWeight",
                    Step="Step",
                    Out="MtpTargetHidden",
                ),
                dict(rows=rows),
            )
            name = f"mtp_capture_m{rows}"
            self.manifest["programs"][name] = p
            capture_plans.append(dict(tokens=rows, program=name))
        warm_plans = []
        for rows in warm_sizes:
            prefix = f"mtp_m{rows}_"
            self.buffer(
                f"MtpLastIndex{rows}", "i32", (1,), torch.tensor([rows - 1], dtype=torch.int32)
            )
            self.compile(
                prefix + "gather",
                lambda: speculation.gather_target_hidden(rows, self.H, self.context),
            )
            self.compile(prefix + "prepare", lambda: control.prepare(rows))
            self.compile(prefix + "advance", lambda: control.advance(rows))
            self.compile(
                prefix + "norm",
                lambda: op("02_residual_norm").residual_norm(
                    rows, residual_dtype="float32", output_residual_dtype="float32"
                ),
            )
            self.compile(prefix + "merge", lambda: op("32_split_k_merge").split_k_merge(rows))
            self.compile(
                prefix + "attention",
                lambda: paged_attention_partials_gqa_staged(self.pages, self.pages, queries=rows),
            )
            self.compile(prefix + "finalnorm", lambda: op("23_final_norm").final_norm(rows, 1))
            for weight, (n, k) in sizes.items():
                split = 8 if weight in ("MtpFC", "MtpOut", "MtpDown") else 1
                if self.weight_formats[weight] == "f16":
                    self.compile(
                        prefix + weight, lambda n=n, k=k: fp16_gemm(rows, n, k, BM=16, BN=64, BK=64)
                    )
                else:
                    self.compile(
                        prefix + weight,
                        lambda n=n, k=k, split=split: w4_small_m(
                            rows,
                            n,
                            k,
                            split,
                            "float32" if split > 1 else "float16",
                            TILE_N=128 if weight != "MtpDown" else 64,
                        ),
                    )
            p = []
            d = dict(rows=rows, M=rows, batch=rows)

            def emit(name, **bind):
                self.op(
                    p, prefix + name if prefix + name in self.exports else "mtp_" + name, bind, d
                )

            def project(weight, source, destination):
                if self.weight_formats[weight] == "f16":
                    emit(weight, A=source, B=weight, C=destination)
                else:
                    split = weight in ("MtpFC", "MtpOut", "MtpDown")
                    emit(
                        weight,
                        A=source,
                        PP=weight + "_P",
                        S=weight + "_S",
                        Z=weight + "_Z",
                        O="MtpPartial" if split else destination,
                    )
                    if split:
                        emit("merge", P="MtpPartial", O=destination)

            emit("gather", Target="MtpTargetHidden", Step="MtpStep", Out="MtpCondition")
            emit("prepare", Step="MtpStep", Positions="MtpPositions", SeqLength="MtpSeqLength")
            embedding = dict(
                P="Embedding_P", S="Embedding_S", Z="Embedding_Z", I="MtpInput", Y="MtpEmbedding"
            )
            if vision:
                embedding.update(
                    Step="MtpStep", Index="MtpFeatureIndex", Features=vision["features"]
                )
            emit("embedding", **embedding)
            emit(
                "concat",
                Embedding="MtpEmbedding",
                Target="MtpCondition",
                EmbeddingWeight="MtpEmbeddingWeight",
                HiddenWeight="MtpHiddenWeight",
                Out="MtpConcat",
            )
            project("MtpFC", "MtpConcat", "MtpHidden")
            p.append(dict(kind="zero", destination="MtpR0", bytes=rows * self.H * 4))
            emit("norm", X="MtpHidden", R="MtpR0", W="MtpPreWeight", Y="MtpNorm", RO="MtpR1")
            project("MtpIn", "MtpNorm", "MtpFullX")
            prepare = dict(
                X="MtpFullX",
                WQ="MtpQWeight",
                WK="MtpKWeight",
                Cache="Rotary",
                Req="Req",
                Pos="MtpPositions",
                Pages="Pages",
                Status="MtpPrepareStatus",
                Q="MtpFullQ",
                Gate="MtpGate",
                K="MtpKPages",
                V="MtpVPages",
            )
            if vision:
                prepare["MRope"] = vision["mrope_positions"]
            emit("fullprepare", **prepare)
            emit(
                "attention",
                Q="MtpFullQ",
                K="MtpKPages",
                V="MtpVPages",
                Pages="Pages",
                SeqLen="MtpSeqLength",
                QueryPos="MtpPositions",
                M="MtpAttM",
                L="MtpAttL",
                O="MtpAttO",
            )
            emit(
                "attentionmerge",
                M="MtpAttM",
                L="MtpAttL",
                O="MtpAttO",
                RawGate="MtpGate",
                Y="MtpMixer",
            )
            project("MtpOut", "MtpMixer", "MtpMix")
            emit("norm", X="MtpMix", R="MtpR1", W="MtpPostWeight", Y="MtpNorm", RO="MtpR0")
            project("MtpGateUp", "MtpNorm", "MtpGateUpResult")
            emit("swiglu", X="MtpGateUpResult", Y="MtpActivated")
            project("MtpDown", "MtpActivated", "MtpHidden")
            emit("advance", Step="MtpStep")
            name = f"mtp_warm_m{rows}"
            self.manifest["programs"][name] = p
            if rows == 1:
                # Chained drafting consumes the previous MTP final hidden.
                # Only the gather is omitted; token/position handling stays
                # identical to native shifted-input warm/refresh.
                chain = [dict(kind="copy", source="MtpToken", destination="MtpInput", bytes=4)] + p[
                    1:
                ]
            p = []
            emit(
                "finalnorm",
                X="MtpHidden",
                R="MtpR0",
                I=f"MtpLastIndex{rows}",
                W="MtpFinalWeight",
                Y="MtpLastHidden",
            )
            emit("head", A="MtpLastHidden", PP="Head_P", S="Head_S", Z="Head_Z", O="MtpLogits")
            self.op(
                p, "topk", dict(X="MtpLogits", PV="MtpPV", PI="MtpPI", Bad="MtpBad"), dict(rows=1)
            )
            self.op(
                p,
                "topkmerge",
                dict(
                    PV="MtpPV",
                    PI="MtpPI",
                    Bad="MtpBad",
                    Values="MtpValues",
                    IDs="MtpIDs",
                    Token="MtpToken",
                    Status="MtpStatus",
                ),
                dict(rows=1),
            )
            self.copy(p, "MtpLastHidden", "MtpCondition", self.H * 2)
            head_name = f"mtp_head_m{rows}"
            self.manifest["programs"][head_name] = p
            if rows == 1:
                self.manifest["programs"]["mtp_draft"] = chain + p
            warm_plans.append(dict(tokens=rows, program=name, head_program=head_name))
        verify_plans = []
        for plan in self.verification["plans"]:
            rows = plan["tokens"]
            verify_plans.append(
                dict(
                    tokens=rows,
                    program=plan["verify_program"],
                    restore_program=plan["restore_program"],
                    capture_program=f"mtp_capture_m{rows}",
                )
            )
        self.manifest["mtp"] = dict(
            position="MtpStep",
            input="MtpInput",
            token="MtpToken",
            status="MtpStatus",
            verification_tokens="SequenceTokens",
            verification_status="SequenceStatus",
            draft_logits="MtpLogits",
            verification_logits="SequenceLogits",
            feature_index="MtpFeatureIndex" if vision else None,
            accepted_inputs="AcceptedInputs",
            target_length="SeqLength",
            draft_program="mtp_draft",
            default_verification_tokens=self.verification_tokens,
            warm_plans=warm_plans,
            capture_plans=capture_plans,
            verification_plans=verify_plans,
        )
        sizes = dict(u8=1, i8=1, f16=2, f32=4, i32=4)
        self.manifest["weight_bytes"] = sum(
            math.prod(b["shape"]) * sizes[b["dtype"]] for b in self.buffers if b["access"] == "read"
        )
        self.manifest["weight_parameters"] += self.weight_report["weight_parameters"]
        self.manifest["weight_scope"] += (
            "; native Qwen3_5 MTP draft, shared embedding/head, all draft bytes counted"
        )
        write_json(self.out / "model.json", self.manifest)
        write_json(
            self.out / "mtp-build.json",
            dict(
                draft_source=self.weight_report["source"],
                draft_format=self.weight_report["format"],
                draft_weight_bytes=self.weight_report["weight_bytes"],
                effective_total_weight_bits=8
                * self.manifest["weight_bytes"]
                / self.manifest["weight_parameters"],
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-tokens", type=int, default=4)
    args = parser.parse_args()
    if any((args.output / name).exists() for name in ("target", "aot", "model.json")):
        parser.error("output already contains model artifacts")
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    Assembler(
        args.model.resolve(),
        args.weights.resolve(),
        args.output.resolve(),
        args.verification_tokens,
    ).build()


if __name__ == "__main__":
    main()
