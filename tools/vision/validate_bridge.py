"""Actual exported embedding/position bridge numerical and graph checks."""

import argparse
import ctypes as C
import json
from pathlib import Path
import torch
from tools.operators.common import configure, error, write_json


def bind(root, k, b):
    d = C.CDLL("libcuda.so.1")

    def check(code):
        if code:
            raise RuntimeError(f"CUDA {code}")

    module = C.c_void_p()
    check(d.cuModuleLoad(C.byref(module), str(root / k["module"]["file"]).encode()))
    fn = C.c_void_p()
    check(d.cuModuleGetFunction(C.byref(fn), module, k["symbol"].encode()))
    if k["shared_memory_bytes"] > 49152:
        check(d.cuFuncSetAttribute(fn, 8, k["shared_memory_bytes"]))
    args = [
        C.c_uint64(b[a["name"]].data_ptr()) if a["kind"] == "buffer" else C.c_int32(a["value"])
        for a in k["args"]
    ]
    ptrs = (C.c_void_p * len(args))(*(C.cast(C.byref(a), C.c_void_p) for a in args))

    def launch():
        check(
            d.cuLaunchKernel(
                fn,
                *k["grid"],
                *k["block"],
                k["shared_memory_bytes"],
                C.c_void_p(torch.cuda.current_stream().cuda_stream),
                ptrs,
                None,
            )
        )

    # Keep the loaded module, scalar storage and argument pointers alive.
    launch.resources = (d, module, fn, args, ptrs)
    return launch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    a = ap.parse_args()
    configure()
    m = json.loads(a.model.read_text())
    results = []
    for kind, select in [("embedding", "Embedding_P"), ("mrope", "MRopePositions")]:
        kernel = next(
            k
            for k in m["kernels"]
            if any(x.get("name") == select for x in k["args"])
            and any(x.get("value") == 512 for x in k["args"])
        )
        names = {x["name"] for x in kernel["args"] if x["kind"] == "buffer"}
        b = {
            s["name"]: torch.zeros(
                s["shape"],
                device="cuda",
                dtype={
                    "f16": torch.float16,
                    "i32": torch.int32,
                    "i8": torch.int8,
                    "u8": torch.uint8,
                }[s["dtype"]],
            )
            for s in m["buffers"]
            if s["name"] in names
        }
        if kind == "embedding":
            b["Input"][:512] = torch.arange(512, device="cuda") % 17
            b["Embedding_P"][:17].random_(0, 255)
            b["Embedding_S"].fill_(0.125)
            b["Embedding_Z"].fill_(7)
            b["Step"].fill_(13)
            b["FeatureIndex"].fill_(-1)
            b["FeatureIndex"][13 : 13 + 512 : 7] = 1
            b["Features"][:2].normal_()
            packed = b["Embedding_P"][b["Input"][:512].long()]
            raw = torch.stack((packed & 15, packed >> 4), -1).flatten(1).float()
            ref = ((raw - 7) * 0.125).half()
            ref[::7] = b["Features"][1]
            target = b["Hidden"][:512]
            changed = b["Features"]
            original = changed.clone()
        else:
            b["FullX"][:512].normal_()
            qw = next(n for n in names if n.endswith("_QWeight"))
            kw = next(n for n in names if n.endswith("_KWeight"))
            kp = next(n for n in names if n.endswith("_KPages"))
            vp = next(n for n in names if n.endswith("_VPages"))
            b[qw].normal_(0, 0.05)
            b[kw].normal_(0, 0.05)
            b["Positions"][:512] = torch.arange(512, device="cuda") + 31
            b["Pages"][0] = torch.arange(b["Pages"].shape[1], device="cuda")
            coords = torch.arange(m["max_context"], device="cuda")[:, None].expand(-1, 3).clone()
            coords[:, 1] = coords[:, 1] // 2
            coords[:, 2] = coords[:, 2] // 3
            b["MRopePositions"].copy_(coords)
            freqs = 1 / (1e7 ** (torch.arange(32, device="cuda").float() / 32))
            angle = torch.arange(b["Rotary"].shape[0], device="cuda")[:, None] * freqs
            b["Rotary"].copy_(torch.cat((angle.cos(), angle.sin()), -1).half())
            x = b["FullX"][:512]
            q = x[:, :12288].reshape(512, 24, 512)[:, :, :256]
            k = x[:, 12288:13312].reshape(512, 4, 256)

            def prep(z, w):
                z = (
                    z.float()
                    * torch.rsqrt(z.float().square().mean(-1, keepdim=True) + 1e-6)
                    * (1 + w.float())
                ).half()
                idx = torch.arange(32, device="cuda")
                axis = torch.where(
                    (idx % 3 == 1) & (idx < 33), 1, torch.where((idx % 3 == 2) & (idx < 30), 2, 0)
                )
                pos = coords[b["Positions"][:512].long()][:, axis]
                cos = b["Rotary"][pos, idx][:, None, :]
                sin = b["Rotary"][pos, idx + 32][:, None, :]
                av = (z[:, :, :64].float() * torch.cat((cos, cos), -1)).half()
                partner = torch.cat((z[:, :, 32:64], z[:, :, :32]), -1)
                bv = (partner.float() * torch.cat((sin, sin), -1)).half()
                out = z.clone()
                out[:, :, :32] = (av[:, :, :32].float() - bv[:, :, :32].float()).half()
                out[:, :, 32:64] = (av[:, :, 32:].float() + bv[:, :, 32:].float()).half()
                return out

            ref = prep(q, b[qw])
            kref = prep(k, b[kw])
            target = b["FullQ"][:512]
            changed = b["MRopePositions"]
            original = changed.clone()
        launch = bind(a.model.parent, kernel, b)
        launch()
        torch.cuda.synchronize()
        e = error(target, ref)
        assert e["finite"] and e["relative_l2"] < 0.001, (kind, e)
        if kind == "mrope":
            pos = b["Positions"][:512].long()
            actualk = b[kp][pos // 128, pos % 128]
            assert error(actualk, kref)["relative_l2"] < 0.001
            assert torch.equal(b[vp][pos // 128, pos % 128], x[:, 13312:].reshape(512, 4, 256))
            assert torch.equal(b["FullGate"][:512], x[:, :12288].reshape(512, 24, 512)[:, :, 256:])
        saved = target.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        changed.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.equal(saved, target)
        changed.copy_(original)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(saved, target)
        results.append({"kernel": kind, "error": e, "changed_input_graph_restore": True})
        print(kind, e, flush=True)
        del graph
        del launch
        b = None
    write_json(a.output / "result.json", {"status": "passed", "checks": results})


if __name__ == "__main__":
    main()
