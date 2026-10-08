"""Add causal small-M W4A16 verification graphs to an existing Qwen3_5 plan.

All resident weights are reused unchanged. Per-layer FP32 GDN and raw conv
prefix snapshots allow a later speculative scheduler to commit any nonempty
input prefix. Ordinary prefill plans commit the full sequence. This builder
does not enable MTP generation or claim an end-to-end speedup.
"""

import argparse
import copy
import importlib
import json
import os
from pathlib import Path

import torch
from tools.model.build import Builder
from tools.operators.common import configure, write_json
from tools.operators.abi import parse_host
from kernels.model.w4_small_m import w4_small_m
from kernels.model.gdn_sequence import gdn_sequence
from kernels.model import control, speculation
from kernels.model.attention_decode_staged import paged_attention_partials_gqa_staged


class Assembler(Builder):
    def __init__(self, source, output, tokens):
        self.out = output
        self.manifest = json.loads(source.read_text())
        self.buffers = self.manifest["buffers"]
        self.kernels = self.manifest["kernels"]
        self.exports = {}
        self.prefill_tokens = 512
        self.i8_grid_order = "nfirst"
        self.reuse_aot = None
        self.dense_u4 = False
        self.reuse_weights = None
        self.reset = self.manifest["reset_buffers"]
        self.tokens = tokens
        self.H = next(b["shape"][1] for b in self.buffers if b["name"] == "Hidden")
        self.F = next(b["shape"][1] for b in self.buffers if b["name"] == "Activated")
        self.V = self.manifest["vocab"]
        self.pages = next(b["shape"][1] for b in self.buffers if b["name"] == "Pages")
        packed = {b["name"]: b["layout"] for b in self.buffers if b["name"].endswith("_P")}
        self.layouts = {}
        for family in ("In", "Out", "GateUp", "Down"):
            layouts = {packed[f"L{i}_{family}_P"] for i in range(64)}
            if len(layouts) != 1 or not layouts <= {
                "u4_warp_n64_k128_mma_f16",
                "u4_warp_n64_k128_mma_i8",
            }:
                raise ValueError(f"Unsupported/mixed {family} packed layouts: {layouts}")
            self.layouts[family] = "i8" if next(iter(layouts)).endswith("_i8") else "f16"
        if self.H != 5120 or self.F != 17408 or self.manifest.get("mtp"):
            raise ValueError("Expected the current Qwen3_5 dense text adapter without MTP")
        adopted = {}

        def adopt(identity):
            if identity["file"] not in adopted:
                old = source.parent / identity["file"]
                relative = Path("base") / identity["file"]
                new = output / relative
                new.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(old, new)
                except PermissionError:
                    import shutil

                    shutil.copyfile(old, new)
                adopted[identity["file"]] = dict(file=str(relative), sha256=identity["sha256"])
            return adopted[identity["file"]].copy()

        for b in self.buffers:
            if b.get("data"):
                b["data"] = adopt(b["data"])
        for k in self.kernels:
            for key in ("module", "source", "host_abi"):
                k[key] = adopt(k[key])
        kernels = {k["name"]: k for k in self.kernels}
        # The existing dynamic exports retain their actual host ABI. Fixed-M
        # norm/projection/conv/control exports are compiled below instead.
        for op in self.manifest["programs"]["decode"]:
            if op["kind"] != "kernel":
                continue
            k = kernels[op["name"]]
            binds = {a["name"] for a in k["args"] if a["kind"] == "buffer"}
            name = None
            if "Embedding_P" in binds:
                name = "embedding"
            elif "FullX" in binds and "Rotary" in binds:
                name = "fullprepare"
            elif "GateUp" in binds and "Activated" in binds:
                name = "swiglu"
            elif "Zout" in binds and "MixerIn" in binds:
                name = "gatednorm"
            elif "AB" in binds and "g" in binds:
                name = "gates"
            elif "FullGate" in binds and "AttM" in binds:
                name = "attentionmerge"
            elif "Logits" in binds and "PV" in binds:
                name = "topk"
            elif "TokenStatus" in binds:
                name = "topkmerge"
            if name:
                abi = parse_host((output / k["host_abi"]["file"]).read_text())
                assert len(abi) == 1
                self.exports[name] = {
                    **abi[0],
                    **{key: k[key] for key in ("module", "source", "host_abi")},
                }
        for name in ("Partial", "AttM", "AttL", "AttO"):
            b = next(b for b in self.buffers if b["name"] == name)
            if name == "Partial":
                b["shape"][1] = max(tokens)
            else:
                b["shape"].insert(0, max(tokens))
        self.buffer("AcceptedInputs", "i32", (1,))
        self.buffer("SequenceHidden", "f16", (max(tokens), self.H))
        self.buffer("SequenceLogits", "f32", (max(tokens), self.V))
        for name, dtype, shape in (
            ("SequencePV", "f32", (max(tokens), (self.V + 4095) // 4096, 1)),
            ("SequencePI", "i32", (max(tokens), (self.V + 4095) // 4096, 1)),
            ("SequenceBad", "i32", (max(tokens), (self.V + 4095) // 4096)),
            ("SequenceValues", "f32", (max(tokens), 1)),
            ("SequenceIDs", "i32", (max(tokens), 1)),
            ("SequenceTokens", "i32", (max(tokens),)),
            ("SequenceStatus", "i32", (max(tokens),)),
        ):
            self.buffer(name, dtype, shape)
        self.buffer(
            "SequenceIndex", "i32", (max(tokens),), torch.arange(max(tokens), dtype=torch.int32)
        )
        for i in range(64):
            if i % 4 != 3:
                self.buffer(f"L{i}_StatePrefixes", "f32", (max(tokens), 48, 128, 128))
                self.buffer(f"L{i}_HistoryPrefixes", "f16", (max(tokens), 3, 10240))

    def build(self):
        def op(n):
            return importlib.import_module("kernels.operators.op" + n)

        plans = []
        for m in self.tokens:
            prefix = f"sequence_m{m}_"
            self.buffer(f"SequenceLength{m}", "i32", (1,), torch.tensor([m], dtype=torch.int32))
            self.buffer(f"SequenceLast{m}", "i32", (1,), torch.tensor([m - 1], dtype=torch.int32))
            for name, factory in (
                ("prepare", lambda: control.prepare(m)),
                ("advance", lambda: control.advance(m)),
                (
                    "norm",
                    lambda: op("02_residual_norm").residual_norm(
                        m, residual_dtype="float32", output_residual_dtype="float32"
                    ),
                ),
                (
                    "qkvz",
                    lambda: w4_small_m(
                        m,
                        16384,
                        self.H,
                        TILE_N=128,
                        output_layout="qkvz",
                        weight_layout=self.layouts["In"],
                    ),
                ),
                (
                    "fullproj",
                    lambda: w4_small_m(
                        m, 14336, self.H, TILE_N=128, weight_layout=self.layouts["In"]
                    ),
                ),
                (
                    "outproj",
                    lambda: w4_small_m(
                        m, self.H, 6144, 8, "float32", TILE_N=128, weight_layout=self.layouts["Out"]
                    ),
                ),
                (
                    "gateup",
                    lambda: w4_small_m(
                        m, 2 * self.F, self.H, TILE_N=128, weight_layout=self.layouts["GateUp"]
                    ),
                ),
                (
                    "down",
                    lambda: w4_small_m(
                        m,
                        self.H,
                        self.F,
                        8,
                        "float32",
                        TILE_N=64,
                        weight_layout=self.layouts["Down"],
                    ),
                ),
                ("merge", lambda: op("32_split_k_merge").split_k_merge(m)),
                ("ab", lambda: op("07_gdn_ab").gdn_ab_tensorcore(m, BM=16, output_dtype="float16")),
                (
                    "conv",
                    lambda: op("08_gdn_conv_prep").gdn_conv_prep(B=1, tokens=m, tile_tokens=1),
                ),
                ("history", lambda: speculation.conv_history_prefixes(m)),
                ("gdn", lambda: gdn_sequence(m)),
                ("commitstate", lambda: speculation.select_prefix(m, 48 * 128 * 128)),
                ("commithistory", lambda: speculation.select_prefix(m, 3 * 10240, "float16")),
                (
                    "attention",
                    lambda: paged_attention_partials_gqa_staged(self.pages, self.pages, queries=m),
                ),
                ("finalnorm", lambda: op("23_final_norm").final_norm(m, 1)),
                ("allnorm", lambda: op("23_final_norm").final_norm(m, m)),
                (
                    "allhead",
                    lambda: w4_small_m(m, self.V, self.H, output_dtype="float32", TILE_N=128),
                ),
            ):
                self.compile(prefix + name, factory)
            p = []
            d = dict(M=m, rows=m, batch=1, tokens=m)

            def emit(name, **bind):
                self.op(p, prefix + name if prefix + name in self.exports else name, bind, d)

            def common_bind():
                return dict(Step="Step", Positions="Positions", SeqLength="SeqLength")

            emit("prepare", **common_bind())
            emit(
                "embedding",
                P="Embedding_P",
                S="Embedding_S",
                Z="Embedding_Z",
                I="Input",
                Step="Step",
                Index="FeatureIndex",
                Features="Features",
                Y="Hidden",
            )
            p.append(dict(kind="zero", destination="R0", bytes=m * self.H * 4))
            for i in range(64):

                def w(name):
                    return f"L{i}_" + name

                emit("norm", X="Hidden", R="R0", W=w("PreWeight"), Y="Norm", RO="R1")
                if i % 4 != 3:
                    emit(
                        "qkvz",
                        A="Norm",
                        PP=w("In_P"),
                        S=w("In_S"),
                        Z=w("In_Z"),
                        QKV="QKV",
                        ZOUT="Zout",
                    )
                    emit("ab", X="Norm", W_ab=w("ABWeight"), Y="AB")
                    emit(
                        "gates",
                        A="AB",
                        B="AB",
                        Parameter=w("Al"),
                        DtBias=w("Dt"),
                        G="g",
                        Beta="Beta",
                    )
                    emit(
                        "history",
                        X="QKV",
                        History=w("History"),
                        Position="Step",
                        Prefix=w("HistoryPrefixes"),
                    )
                    emit(
                        "conv",
                        X="QKV",
                        W=w("ConvWeight"),
                        HI=w("History"),
                        lengths=f"SequenceLength{m}",
                        positions="Step",
                        Q="Q",
                        K="K",
                        V="V",
                        HO="Ho",
                        positions_out="Po",
                    )
                    self.copy(p, "Ho", w("History"), 3 * 10240 * 2)
                    emit(
                        "gdn",
                        Q="Q",
                        K="K",
                        V="V",
                        G="g",
                        Beta="Beta",
                        State=w("State"),
                        Prefix=w("StatePrefixes"),
                        Out="Y",
                    )
                    emit(
                        "commitstate",
                        Prefix=w("StatePrefixes"),
                        Count=f"SequenceLength{m}",
                        State=w("State"),
                    )
                    emit("gatednorm", X="Y", Z="Zout", W=w("GatedWeight"), Y="MixerIn")
                else:
                    emit("fullproj", A="Norm", PP=w("In_P"), S=w("In_S"), Z=w("In_Z"), O="FullX")
                    emit(
                        "fullprepare",
                        X="FullX",
                        WQ=w("QWeight"),
                        WK=w("KWeight"),
                        Cache="Rotary",
                        Req="Req",
                        Pos="Positions",
                        Pages="Pages",
                        Status="PrepareStatus",
                        MRope="MRopePositions",
                        Q="FullQ",
                        Gate="FullGate",
                        K=w("KPages"),
                        V=w("VPages"),
                    )
                    emit(
                        "attention",
                        Q="FullQ",
                        K=w("KPages"),
                        V=w("VPages"),
                        Pages="Pages",
                        SeqLen="SeqLength",
                        QueryPos="Positions",
                        M="AttM",
                        L="AttL",
                        O="AttO",
                    )
                    self.op(
                        p,
                        "attentionmerge",
                        dict(M="AttM", L="AttL", O="AttO", RawGate="FullGate", Y="MixerIn"),
                        dict(batch=m),
                    )
                emit("outproj", A="MixerIn", PP=w("Out_P"), S=w("Out_S"), Z=w("Out_Z"), O="Partial")
                emit("merge", P="Partial", O="Mix")
                emit("norm", X="Mix", R="R1", W=w("PostWeight"), Y="Norm", RO="R0")
                emit(
                    "gateup",
                    A="Norm",
                    PP=w("GateUp_P"),
                    S=w("GateUp_S"),
                    Z=w("GateUp_Z"),
                    O="GateUp",
                )
                emit("swiglu", X="GateUp", Y="Activated")
                emit(
                    "down", A="Activated", PP=w("Down_P"), S=w("Down_S"), Z=w("Down_Z"), O="Partial"
                )
                emit("merge", P="Partial", O="Hidden")
            emit("advance", Step="Step")
            self.manifest["programs"][f"prefill_m{m}"] = p
            p = []
            emit(
                "finalnorm",
                X="Hidden",
                R="R0",
                I=f"SequenceLast{m}",
                W="FinalWeight",
                Y="LastHidden",
            )
            # Existing M1 head/selection kernels are adopted by their concrete
            # ABI, and finalnorm is replaced with the chosen sequence index.
            old_head = self.manifest["programs"]["head_m512"]
            p.extend(copy.deepcopy(old_head[1:]))
            self.manifest["programs"][f"head_m{m}"] = p
            self.manifest["prefill_plans"].append(
                dict(chunk_tokens=m, prefill_program=f"prefill_m{m}", head_program=f"head_m{m}")
            )
            p = copy.deepcopy(self.manifest["programs"][f"prefill_m{m}"])
            emit(
                "allnorm",
                X="Hidden",
                R="R0",
                I="SequenceIndex",
                W="FinalWeight",
                Y="SequenceHidden",
            )
            emit(
                "allhead",
                A="SequenceHidden",
                PP="Head_P",
                S="Head_S",
                Z="Head_Z",
                O="SequenceLogits",
            )
            emit("topk", X="SequenceLogits", PV="SequencePV", PI="SequencePI", Bad="SequenceBad")
            emit(
                "topkmerge",
                PV="SequencePV",
                PI="SequencePI",
                Bad="SequenceBad",
                Values="SequenceValues",
                IDs="SequenceIDs",
                Token="SequenceTokens",
                Status="SequenceStatus",
            )
            self.manifest["programs"][f"verify_m{m}"] = p
            p = []
            for i in range(64):
                if i % 4 != 3:
                    emit(
                        "commitstate",
                        Prefix=f"L{i}_StatePrefixes",
                        Count="AcceptedInputs",
                        State=f"L{i}_State",
                    )
                    emit(
                        "commithistory",
                        Prefix=f"L{i}_HistoryPrefixes",
                        Count="AcceptedInputs",
                        State=f"L{i}_History",
                    )
            self.manifest["programs"][f"restore_m{m}"] = p
            plans.append(
                dict(tokens=m, verify_program=f"verify_m{m}", restore_program=f"restore_m{m}")
            )
        # Only mutable buffers were added; resident parameter accounting stays
        # exactly the same as the source manifest.
        write_json(self.out / "model.json", self.manifest)
        write_json(
            self.out / "verification-plans.json",
            dict(
                plans=plans,
                accepted_inputs="AcceptedInputs",
                tokens="SequenceTokens",
                status="SequenceStatus",
                hidden="SequenceHidden",
                logits="SequenceLogits",
                contract="Restore 1..M processed inputs; scheduler must also restore Step/SeqLength and select hidden/logits. KV future entries are masked by absolute query position.",
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()
    if (
        not args.tokens
        or len(set(args.tokens)) != len(args.tokens)
        or any(not 1 < m <= 16 for m in args.tokens)
    ):
        parser.error("unique token counts in 2..16 required")
    if any((args.output / name).exists() for name in ("base", "aot", "model.json")):
        parser.error("output already contains model artifacts")
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    Assembler(args.model.resolve(), args.output.resolve(), args.tokens).build()


if __name__ == "__main__":
    main()
