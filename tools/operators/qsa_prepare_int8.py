"""Original NeoX RoPE and group-64 INT8 cache writes at the 256k boundary."""

import argparse
from pathlib import Path
import torch
from kernels.model.flash_next import qsa_prepare
from kernels.model.qsa import kv_store
from tools.operators.flash_next import rotate
from tools.model.flash_next.reference.qsa import quantize_kv
from tools.operators.common import configure, error, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configure()
    m, capacity = 17, 262144
    qg = torch.randn((m, 24, 2, 256), device="cuda").half()
    k = torch.randn((m, 2, 256), device="cuda").half()
    v = torch.randn_like(k)
    # Exact zero and tiny values check the minimum nonzero scale.
    v[0].zero_()
    v[1].fill_(2**-20)
    qw = torch.rand(256, device="cuda") + 0.5
    kw = torch.rand_like(qw) + 0.5
    pos = torch.tensor([capacity - m], device="cuda", dtype=torch.int32)
    q = torch.empty((m, 24, 256), device="cuda", dtype=torch.float16)
    gate = torch.empty_like(q)
    sk = torch.empty_like(k)
    sv = torch.empty_like(v)
    ki = torch.full((capacity, 2, 256), -128, device="cuda", dtype=torch.int8)
    vi = torch.full_like(ki, -128)
    ks = torch.full((capacity, 2, 4), float("nan"), device="cuda", dtype=torch.float16)
    vs = torch.full_like(ks, float("nan"))
    prep = qsa_prepare(m, capacity, is_neox_style=True, staged=True)
    store = kv_store(m, capacity)

    def run():
        prep(qg, k, v, qw, kw, pos, q, sk, sv, gate)
        store(sk, sv, pos, ki, vi, ks, vs)

    def validate():
        start = int(pos.item())
        positions = torch.arange(start, start + m, device="cuda")
        query = qg[:, :, 0].float()
        keys = k.float()
        rq = rotate(
            query * torch.rsqrt(query.square().mean(-1, keepdim=True) + 1e-6) * qw,
            positions,
            64,
            True,
        )
        rk = rotate(
            keys * torch.rsqrt(keys.square().mean(-1, keepdim=True) + 1e-6) * kw,
            positions,
            64,
            True,
        )
        metrics = {"query": error(q, rq), "key": error(sk, rk)}
        assert all(x["finite"] and x["relative_l2"] < 0.004 for x in metrics.values()), metrics
        ik, ss = quantize_kv(sk)
        iv, tt = quantize_kv(sv)
        assert torch.equal(ki[start : start + m], ik) and torch.equal(vi[start : start + m], iv)
        assert torch.equal(ks[start : start + m], ss) and torch.equal(vs[start : start + m], tt)
        assert torch.equal(sv, v) and torch.equal(gate, qg[:, :, 1])
        return metrics

    run()
    torch.cuda.synchronize()
    metric = validate()
    saved = q.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    pos.fill_(2047)
    qg.mul_(0.8)
    graph.replay()
    torch.cuda.synchronize()
    changed = validate()
    assert not torch.equal(q, saved)
    assert bool((ki[:2047] == -128).all()) and bool(torch.isnan(ks[:2047]).all())
    write_json(
        a.output / "results.json",
        {
            "complete": True,
            "capacity": capacity,
            "kv_dtype": "int8",
            "group": 64,
            "errors": metric,
            "changed_graph_errors": changed,
            "quantization_exact": True,
            "causal_prefix_unmodified": True,
        },
    )


if __name__ == "__main__":
    main()
