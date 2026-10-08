"""GPU boundary and graph-replay checks for the prepared context capacity."""

import argparse
import gc
from pathlib import Path

import torch
from tools.operators.common import configure, error, write_json
from kernels.vision.bridge import full_prepare_mrope
from kernels.model.speculation import capture_target_hidden, gather_target_hidden
from kernels.model.attention_decode_staged import paged_attention_partials_gqa_staged
from kernels.operators.op27_state_lifecycle import paged_kv_gather


def run(context, output, ring):
    configure()
    device = "cuda"
    pages = context // 128
    table = torch.arange(pages, dtype=torch.int32, device=device).reshape(1, pages)
    k = torch.zeros((pages, 128, 4, 256), dtype=torch.float16, device=device)
    v = torch.zeros_like(k)
    # Last-page writes and the original 8704-token boundary, using the real
    # production Q/K norm, MRoPE and physical KV layout.
    positions = torch.tensor([0, 8704, context - 1], dtype=torch.int32, device=device)
    rope = torch.zeros((context, 64), dtype=torch.float16, device=device)
    rope[:, :32] = 1
    mrope = torch.arange(context, dtype=torch.int32, device=device)[:, None].repeat(1, 3)
    x = torch.randn((3, 14336), dtype=torch.float16, device=device)
    weight = torch.zeros(256, dtype=torch.float16, device=device)
    request = torch.zeros(3, dtype=torch.int32, device=device)
    status = torch.zeros(1, dtype=torch.int32, device=device)
    q = torch.empty((3, 24, 256), dtype=torch.float16, device=device)
    gate = torch.empty_like(q)
    prepare = full_prepare_mrope(pages, context, (11, 11, 10), max_position=context)

    def launch_prepare():
        prepare.torch_function(
            x, weight, weight, rope, request, positions, table, status, mrope, q, gate, k, v
        )

    launch_prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_prepare()
    x.copy_(torch.randn_like(x))
    graph.replay()
    expected_k = x[:, 12288:13312].reshape(3, 4, 256).float()
    expected_k = (
        expected_k * torch.rsqrt(expected_k.square().mean(-1, keepdim=True) + 1e-6)
    ).half()
    actual_k = k.reshape(context, 4, 256)[positions.long()]
    check = error(actual_k, expected_k)
    assert check["relative_l2"] < 0.001, check
    assert torch.equal(
        v.reshape(context, 4, 256)[positions.long()], x[:, 13312:].reshape(3, 4, 256)
    )
    assert int(status.item()) == 0
    # Uniform attention makes the last-token contribution analytically known.
    # Moving QueryPos by one must exclude it, including graph replay.
    k.zero_()
    v.zero_()
    v.reshape(context, 4, 256)[-1].fill_(65504)
    query = torch.zeros((1, 24, 256), dtype=torch.float16, device=device)
    length = torch.tensor([context], dtype=torch.int32, device=device)
    pos = torch.tensor([context - 1], dtype=torch.int32, device=device)
    maximum = torch.empty((1, 24, 8), dtype=torch.float32, device=device)
    denom = torch.empty_like(maximum)
    partial = torch.empty((1, 24, 8, 256), dtype=torch.float32, device=device)
    attention = paged_attention_partials_gqa_staged(pages, pages)

    def launch_attention():
        attention.torch_function(query, k, v, table, length, pos, maximum, denom, partial)

    launch_attention()
    actual = partial.sum(2) / denom.sum(2).unsqueeze(-1)
    assert torch.allclose(actual, torch.full_like(actual, 65504 / context), atol=1e-6), actual
    attention_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(attention_graph):
        launch_attention()
    pos.fill_(context - 2)
    attention_graph.replay()
    assert torch.count_nonzero(partial).item() == 0
    assert torch.equal(
        denom.sum(2), torch.full((1, 24), context - 1, device=device, dtype=torch.float32)
    )
    # Bitwise gather ABI uses packed pairs of FP16; do not reinterpret as floats.
    out_k = torch.empty((1, context, 512), dtype=torch.int32, device=device)
    out_v = torch.empty_like(out_k)
    gather = paged_kv_gather()
    gather.torch_function(
        k.view(torch.int32).reshape(pages, 128, 512),
        v.view(torch.int32).reshape(pages, 128, 512),
        table,
        length,
        out_k,
        out_v,
    )
    assert torch.equal(
        out_v.view(torch.float16).reshape(context, 4, 256)[-1],
        torch.full((4, 256), 65504, device=device, dtype=torch.float16),
    )
    length.fill_(context - 1)
    gather_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gather_graph):
        gather.torch_function(
            k.view(torch.int32).reshape(pages, 128, 512),
            v.view(torch.int32).reshape(pages, 128, 512),
            table,
            length,
            out_k,
            out_v,
        )
    out_v.fill_(-1)
    gather_graph.replay()
    assert torch.count_nonzero(out_v[0, -1]).item() == 0
    del graph
    del attention_graph
    del gather_graph
    k = None
    v = None
    del out_k
    del out_v
    rope = None
    mrope = None
    gc.collect()
    ring_error = run_ring(context, ring)
    write_json(
        output / "result.json",
        dict(
            status="passed",
            max_context=context,
            last_page=pages - 1,
            prepare_error=check,
            ring_error=ring_error,
            attention_last_token=True,
            causal_tail=True,
            gather_tail=True,
            replay_changed_inputs=True,
        ),
    )
    print("256k boundary and ring replay passed", flush=True)


def run_ring(context, ring):
    device = "cuda"
    # Capture a non-aligned chunk crossing the hidden ring wrap, then gather
    # the same absolute positions. Change both inputs and Step before replay.
    rows, hidden = 8, 5120
    capture = capture_target_hidden(hidden, ring, ring=True)
    hidden_gather = gather_target_hidden(rows, hidden, ring, ring=True)
    x = torch.randn((rows, hidden), dtype=torch.float16, device=device)
    residual = torch.randn((rows, hidden), dtype=torch.float32, device=device)
    weight = torch.zeros(hidden, dtype=torch.float16, device=device)
    target = torch.empty((ring, hidden), dtype=torch.float16, device=device)
    step = torch.tensor([ring + 3], dtype=torch.int32, device=device)
    read_step = step - rows
    result = torch.empty_like(x)

    def launch_ring():
        capture.torch_function(x, residual, weight, step, target)
        hidden_gather.torch_function(target, read_step, result)

    launch_ring()
    ring_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ring_graph):
        launch_ring()
    step.fill_(context)
    read_step.fill_(context - rows)
    x.copy_(torch.randn_like(x))
    ring_graph.replay()
    expected = x.float() + residual
    expected = (expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)).half()
    ring_error = error(result, expected)
    assert ring_error["relative_l2"] < 0.001, ring_error
    return ring_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-ring", type=int, default=8192)
    parser.add_argument("--only-ring", action="store_true")
    args = parser.parse_args()
    if args.only_ring:
        configure()
        result = run_ring(args.max_context, args.hidden_ring)
        write_json(
            args.output / "result.json",
            dict(status="passed", ring_tokens=args.hidden_ring, ring_error=result),
        )
    else:
        run(args.max_context, args.output, args.hidden_ring)


if __name__ == "__main__":
    main()
