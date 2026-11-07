"""Check warp-owned input prefetch against the original FP32 recurrence."""

import argparse
from pathlib import Path

import torch

from kernels.cuda.model.gdn_sequence import gdn_sequence
from orinfer_build.compiler.cuda.common import benchmark, configure, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[1, 2, 7, 32, 511, 512, 2048, 4096]
    )
    args = parser.parse_args()
    configure()
    cases = []
    for tokens in args.tokens:
        q = torch.randn((16, tokens, 128), device="cuda", dtype=torch.float16) * 0.1
        k = torch.randn_like(q) * 0.1
        v = torch.randn((48, tokens, 128), device="cuda", dtype=torch.float16) * 0.1
        g = -torch.rand((tokens, 48), device="cuda") * 0.1
        beta = torch.rand_like(g) * 0.1
        initial = torch.randn((48, 128, 128), device="cuda") * 0.01
        state = initial.clone()
        output = torch.empty((tokens, 6144), device="cuda", dtype=torch.float16)
        unused = torch.empty(1, device="cuda")
        reference = gdn_sequence(tokens, threads=32, value_tile=16, in_place=True)
        prefetch = gdn_sequence(tokens, threads=32, value_tile=16, in_place=True, prefetch_qk=True)
        inputs = (q, k, v, g, beta, state, unused, output)
        reference(*inputs)
        expected_state, expected_output = state.clone(), output.clone()

        def run():
            state.copy_(initial)
            prefetch(*inputs)

        run()
        torch.cuda.synchronize()
        assert torch.equal(state, expected_state), (tokens, "state")
        assert torch.equal(output, expected_output), (tokens, "output")
        timing, graph = benchmark(run, repetitions=3)
        saved = q.clone()
        q.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert bool((output == 0).all()), tokens
        q.copy_(saved)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(state, expected_state), (tokens, "graph state")
        assert torch.equal(output, expected_output), (tokens, "graph output")
        saved_inputs = (v.clone(), g.clone(), beta.clone())
        v.neg_()
        g.sub_(0.03)
        beta.mul_(0.7)
        state.copy_(initial)
        reference(*inputs)
        changed_state, changed_output = state.clone(), output.clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(state, changed_state), (tokens, "changed V/G/Beta state")
        assert torch.equal(output, changed_output), (tokens, "changed V/G/Beta output")
        for destination, saved in zip((v, g, beta), saved_inputs, strict=True):
            destination.copy_(saved)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(state, expected_state), (tokens, "restored input state")
        assert torch.equal(output, expected_output), (tokens, "restored input output")
        del graph
        cases.append(
            dict(
                tokens=tokens,
                nonzero_state_exact=True,
                graph_exact=True,
                changed_value_gate_beta_graph=True,
                timing=timing,
            )
        )
        write_json(args.output / "results.json", dict(complete=False, cases=cases))
        print(tokens, "all state and output bytes exact", flush=True)
    write_json(args.output / "results.json", dict(complete=True, cases=cases))


if __name__ == "__main__":
    main()
