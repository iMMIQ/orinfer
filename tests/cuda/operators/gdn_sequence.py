"""Validate FP32 GDN state across chunks, verification and accepted prefixes."""

import argparse
from pathlib import Path

import torch
import tilelang

from kernels.cuda.model.gdn_sequence import _gdn_sequence, gdn_commit, gdn_sequence
from kernels.shared.rows import explicit_rows
from orinfer_build.compiler.cuda.common import configure, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    hk, hv, width, length = 16, 48, 128, 37
    q = torch.randn(hk, length, width, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    k = (k / k.float().norm(dim=-1, keepdim=True)).half()
    v = torch.randn(hv, length, width, device="cuda", dtype=torch.float16) * 0.3
    g = -torch.rand(length, hv, device="cuda") * 0.12
    beta = torch.rand_like(g) * 0.6 + 0.1
    initial = torch.randn(hv, width, width, device="cuda") * 0.02
    accepted = torch.zeros(1, device="cuda", dtype=torch.int32)
    report = {"complete": False, "seed": 20261002, "cases": []}
    step = gdn_sequence(1, in_place=True)
    sequence = gdn_sequence(length, in_place=True)
    verify = gdn_sequence(8, compact=True)
    commit = gdn_commit(8)
    state = initial.clone()
    prefix = torch.empty(1, device="cuda")  # Unused by the in-place entry.
    expected = []
    states = [state.clone()]
    for t in range(length):
        out = torch.empty(1, hv * width, device="cuda", dtype=torch.float16)
        step(
            q[:, t : t + 1].contiguous(),
            k[:, t : t + 1].contiguous(),
            v[:, t : t + 1].contiguous(),
            g[t : t + 1],
            beta[t : t + 1],
            state,
            prefix,
            out,
        )
        expected.append(out)
        states.append(state.clone())
    expected = torch.cat(expected)
    actual = torch.empty_like(expected)
    state.copy_(initial)
    sequence(q, k, v, g, beta, state, prefix, actual)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected), "chunk output"
    assert torch.equal(state, states[-1]), "chunk state"
    prefill = tilelang.compile(
        explicit_rows(
            _gdn_sequence(
                4096,
                in_place=True,
                value_tile=16,
                threads=32,
                prefetch_qk=True,
                token_unroll=4,
                dynamic_rows=True,
            )
        ),
        out_idx=[],
        execution_backend="nvrtc",
        target={"kind": "cuda", "arch": "sm_87"},
        pass_configs={
            "tl.disable_thread_storage_sync": True,
            "tl.disable_shared_memory_reuse": True,
        },
    )
    state.copy_(initial)
    prefill.adapter.func(
        q,
        k,
        v,
        g,
        beta,
        state,
        prefix,
        actual,
        length,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, expected), "dynamic prefill output"
    assert torch.equal(state, states[-1]), "dynamic prefill state"
    updates = torch.empty(8, hv, width, device="cuda")
    verified = torch.empty(8, hv * width, device="cuda", dtype=torch.float16)
    state.copy_(initial)
    inputs = [
        q[:, :8].contiguous(),
        k[:, :8].contiguous(),
        v[:, :8].contiguous(),
        g[:8],
        beta[:8],
    ]
    verify(*inputs, state, updates, verified)
    torch.cuda.synchronize()
    assert torch.equal(state, initial), "verification mutated input"
    assert torch.equal(verified, expected[:8]), "verification output"
    for count in range(9):
        state.copy_(initial)
        accepted.fill_(count)
        commit(inputs[1], inputs[3], updates, accepted, state)
        torch.cuda.synchronize()
        assert torch.equal(state, states[count]), ("accepted prefix", count)
    # Replay captured verification with changed queries and unchanged state.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        verify(*inputs, state, updates, verified)
    inputs[0].neg_()
    state.copy_(initial)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(verified, -expected[:8]), "changed-input graph"
    report["cases"].append(
        dict(
            dtype="float32",
            chunk_exact=True,
            dynamic_prefill_exact=True,
            verification_exact=True,
            accepted_prefixes=9,
            changed_input_graph=True,
        )
    )
    print(report["cases"][-1], flush=True)
    report["scope"] = (
        "Synthetic state-contract regression; model quality requires fixed-history scoring."
    )
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
