"""Validate and time real-weight small-M projections and causal GDN snapshots.

This is an operator test, not an end-to-end MTP throughput measurement.
Run with tools/operators/run.sh to serialize GPU work and record provenance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from common import configure, environment, error, benchmark, export_kernel, write_json
from kernels.model.w4_decode_register_mma import w4_decode_register_mma
from kernels.model.w4_decode_i8layout_vector import w4_decode_i8layout_vector
from kernels.model.w4_small_m import w4_small_m
from kernels.model.gdn_recurrent_inplace import gdn_recurrent_inplace
from kernels.model.gdn_sequence import gdn_sequence
from kernels.model import speculation
from kernels.model.attention_decode_staged import paged_attention_partials_gqa_staged
from kernels.operators.op23_final_norm import final_norm


def launch(kernel, *inputs):
    kernel.adapter.func(*inputs, stream=torch.cuda.current_stream().cuda_stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/model/w4a8-multiplan-lut4-gdn-fused01/model.json"),
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3, 4, 8, 16])
    parser.add_argument("--skip-projections", action="store_true")
    args = parser.parse_args()
    configure()
    manifest = json.loads(args.model.read_text())
    buffers = {b["name"]: b for b in manifest["buffers"]}
    results = []
    args.output.mkdir(parents=True, exist_ok=True)
    projection_shapes = (
        ("L0_In", 16384, 5120, 1, 128),
        ("L3_In", 14336, 5120, 1, 128),
        ("L0_Out", 5120, 6144, 8, 128),
        ("L0_GateUp", 34816, 5120, 1, 128),
        ("L0_Down", 5120, 17408, 8, 64),
        ("Head", 248320, 5120, 1, 128),
    )
    for weight, n, k, split, tile in () if args.skip_projections else projection_shapes:
        arrays = []
        for suffix, dtype in (("_P", np.uint32), ("_S", np.float16), ("_Z", np.int8)):
            b = buffers[weight + suffix]
            array = np.fromfile(args.model.parent / b["data"]["file"], dtype=dtype).reshape(
                b["shape"]
            )
            arrays.append(torch.from_numpy(array).cuda())
        layout = buffers[weight + "_P"]["layout"]
        assert layout in ("u4_warp_n64_k128_mma_f16", "u4_warp_n64_k128_mma_i8"), layout
        input_layout = "i8" if layout.endswith("_i8") else "f16"
        baseline_factory = (
            w4_decode_i8layout_vector if input_layout == "i8" else w4_decode_register_mma
        )
        baseline = baseline_factory(
            n, k, split, "float32" if split > 1 or weight == "Head" else "float16", TILE_N=tile
        )
        for m in args.tokens:
            dtype = torch.float32 if split > 1 or weight == "Head" else torch.float16
            kernel = w4_small_m(
                m,
                n,
                k,
                split,
                "float32" if dtype == torch.float32 else "float16",
                TILE_N=tile,
                weight_layout=input_layout,
            )
            a = torch.randn((m, k), device="cuda", dtype=torch.float16)
            output = torch.empty((split, m, n), device="cuda", dtype=dtype)
            reference_rows = [torch.empty((split, n), device="cuda", dtype=dtype) for _ in range(m)]

            def sequential():
                for i in range(m):
                    launch(baseline, a[i : i + 1], *arrays, reference_rows[i])

            def run():
                launch(kernel, a, *arrays, output)

            sequential()
            run()
            torch.cuda.synchronize()
            reference = torch.stack(reference_rows, dim=1)
            metrics = error(output, reference)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
            sequential_timing, _ = benchmark(sequential, repetitions=8)
            timing, graph = benchmark(run, repetitions=8)
            a.mul_(0.5)
            sequential()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, torch.stack(reference_rows, dim=1), rtol=0, atol=0)
            export_kernel(kernel, args.output / "aot" / f"{weight}_m{m}")
            row = dict(
                weight=weight,
                weight_layout=layout,
                tokens=m,
                error=metrics,
                timing=timing,
                sequential_timing=sequential_timing,
                speedup=sequential_timing["median_ms"] / timing["median_ms"],
                changed_input_replay=True,
            )
            if weight == "L0_In":
                separated = w4_small_m(
                    m, n, k, TILE_N=tile, output_layout="qkvz", weight_layout=input_layout
                )
                qkv = torch.empty((m, 10240), device="cuda", dtype=torch.float16)
                z = torch.empty((m, 6144), device="cuda", dtype=torch.float16)
                launch(separated, a, *arrays, qkv, z)
                torch.testing.assert_close(torch.cat((qkv, z), dim=1), output[0], rtol=0, atol=0)
                export_kernel(separated, args.output / "aot" / f"qkvz_m{m}")
            results.append(row)
            print(json.dumps(row), flush=True)
            write_json(args.output / "result.json", dict(status="running", projections=results))
        arrays = None
        del baseline
        del kernel
        a = None
        del output
        del reference
        reference_rows = None
        torch.cuda.empty_cache()

    recurrence = gdn_recurrent_inplace(q_scale=128**-0.5)
    gdn_results = []
    for m in args.tokens:
        q = torch.nn.functional.normalize(torch.randn(16, m, 128, device="cuda"), dim=-1).half()
        k = torch.nn.functional.normalize(torch.randn(16, m, 128, device="cuda"), dim=-1).half()
        v = torch.randn(48, m, 128, device="cuda").half()
        g = -torch.rand(m, 48, device="cuda")
        beta = torch.rand(m, 48, device="cuda").half().float()
        state = torch.randn(48, 128, 128, device="cuda") * 0.1
        saved_state = state.clone()
        prefixes = torch.empty(m, 48, 128, 128, device="cuda")
        output = torch.empty(m, 6144, device="cuda", dtype=torch.float16)
        kernel = gdn_sequence(m)
        launch(kernel, q, k, v, g, beta, state, prefixes, output)
        torch.testing.assert_close(state, saved_state, rtol=0, atol=0)
        running = state.clone()
        expected = torch.empty(m, 48, 128, device="cuda", dtype=torch.float16)
        for i in range(m):
            launch(
                recurrence,
                q[:, i].contiguous().unsqueeze(0),
                k[:, i].contiguous().unsqueeze(0),
                v[:, i].contiguous().unsqueeze(0),
                g[i : i + 1],
                beta[i : i + 1],
                running.unsqueeze(0),
                expected[i : i + 1],
            )
            torch.testing.assert_close(prefixes[i], running, rtol=1e-6, atol=1e-6)
        metrics = error(output, expected.reshape(m, 6144))
        torch.testing.assert_close(output, expected.reshape(m, 6144), rtol=1e-3, atol=2e-4)
        # Every rejection position must restore a state that reproduces the
        # next causal output, not just the final fully accepted endpoint.
        for accepted in range(1, m):
            restored = prefixes[accepted - 1].clone()
            continued = torch.empty(1, 48, 128, device="cuda", dtype=torch.float16)
            launch(
                recurrence,
                q[:, accepted].contiguous().unsqueeze(0),
                k[:, accepted].contiguous().unsqueeze(0),
                v[:, accepted].contiguous().unsqueeze(0),
                g[accepted : accepted + 1],
                beta[accepted : accepted + 1],
                restored.unsqueeze(0),
                continued,
            )
            torch.testing.assert_close(restored, prefixes[accepted], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(continued[0], expected[accepted], rtol=1e-3, atol=2e-4)
        timing, graph = benchmark(lambda: launch(kernel, q, k, v, g, beta, state, prefixes, output))
        state.mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(state, saved_state * 0.5, rtol=0, atol=0)
        running.copy_(state)
        commit = speculation.select_prefix(m, 48 * 128 * 128)
        chosen = torch.empty_like(state)
        count = torch.ones(1, device="cuda", dtype=torch.int32)
        for i in range(m):
            launch(
                recurrence,
                q[:, i].contiguous().unsqueeze(0),
                k[:, i].contiguous().unsqueeze(0),
                v[:, i].contiguous().unsqueeze(0),
                g[i : i + 1],
                beta[i : i + 1],
                running.unsqueeze(0),
                expected[i : i + 1],
            )
            count.fill_(i + 1)
            launch(commit, prefixes, count, chosen)
            torch.testing.assert_close(chosen, running, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(output, expected.reshape(m, 6144), rtol=1e-3, atol=2e-4)
        history_kernel = speculation.conv_history_prefixes(m)
        history_commit = speculation.select_prefix(m, 3 * 10240, "float16")
        raw = torch.randn(m, 10240, device="cuda").half()
        history = torch.randn(3, 10240, device="cuda").half()
        saved_history = torch.empty(m, 3, 10240, device="cuda", dtype=torch.float16)
        chosen_history = torch.empty_like(history)
        position = torch.zeros(1, device="cuda", dtype=torch.int32)
        for start in (0, 1, 3, 17):
            position.fill_(start)
            launch(history_kernel, raw, history, position, saved_history)
            valid_history = history.clone()
            if start < 3:
                valid_history[: 3 - start].zero_()
            for i in range(m):
                count.fill_(i + 1)
                launch(history_commit, saved_history, count, chosen_history)
                expected_history = torch.cat((valid_history, raw[: i + 1]), dim=0)[-3:]
                torch.testing.assert_close(chosen_history, expected_history, rtol=0, atol=0)
        export_kernel(kernel, args.output / "aot" / f"gdn_sequence_m{m}")
        row = dict(
            tokens=m,
            error=metrics,
            timing=timing,
            prefix_restore=True,
            changed_input_replay=True,
            convolution_history_prefixes=True,
        )
        gdn_results.append(row)
        print(json.dumps(row), flush=True)
    attention_results = []
    baseline = paged_attention_partials_gqa_staged(8, 8)
    key = torch.randn(8, 128, 1024, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    pages = torch.randperm(8, device="cuda", dtype=torch.int32).reshape(1, 8)
    seq_len = torch.ones(1, device="cuda", dtype=torch.int32)
    for m in args.tokens:
        kernel = paged_attention_partials_gqa_staged(8, 8, queries=m)
        query = torch.randn(m, 24, 256, device="cuda", dtype=torch.float16) * 0.1
        pos = torch.arange(m, device="cuda", dtype=torch.int32) + 127
        seq_len.fill_(127 + m)
        output = [
            torch.empty((m, 24, 8), device="cuda"),
            torch.empty((m, 24, 8), device="cuda"),
            torch.empty((m, 24, 8, 256), device="cuda"),
        ]
        reference = [torch.empty_like(x) for x in output]
        for i in range(m):
            launch(
                baseline,
                query[i : i + 1],
                key,
                value,
                pages,
                seq_len,
                pos[i : i + 1],
                *(x[i : i + 1] for x in reference),
            )
        launch(kernel, query, key, value, pages, seq_len, pos, *output)
        for actual, expected in zip(output, reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        timing, graph = benchmark(
            lambda: launch(kernel, query, key, value, pages, seq_len, pos, *output)
        )
        query.mul_(0.5)
        graph.replay()
        for i in range(m):
            launch(
                baseline,
                query[i : i + 1],
                key,
                value,
                pages,
                seq_len,
                pos[i : i + 1],
                *(x[i : i + 1] for x in reference),
            )
        for actual, expected in zip(output, reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Mutating keys/values after the first query must not affect its
        # statistics, even though they are visible to later queries.
        first = [x[0].clone() for x in output]
        physical_page = pages[0, 1].item()
        key[physical_page, : max(m - 1, 1)].mul_(2)
        value[physical_page, : max(m - 1, 1)].mul_(2)
        graph.replay()
        for actual, expected in zip(output, first):
            torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
        attention_results.append(
            dict(
                tokens=m,
                timing=timing,
                causal=True,
                m1_statistics_bytes_equal=True,
                changed_input_replay=True,
            )
        )
        export_kernel(kernel, args.output / "aot" / f"attention_sequence_m{m}")
    norm = speculation.mtp_norm_concat(5120)
    embedding = torch.randn(4, 5120, device="cuda").half()
    target = torch.randn_like(embedding)
    ew = (torch.randn(5120, device="cuda") * 0.1).half()
    hw = (torch.randn_like(ew) * 0.1).half()
    joined = torch.empty(4, 10240, device="cuda", dtype=torch.float16)

    def norm_reference(x, w):
        xf = x.float()
        return (
            xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * (1 + w.float())
        ).half()

    launch(norm, embedding, target, ew, hw, joined)
    reference = torch.cat((norm_reference(embedding, ew), norm_reference(target, hw)), dim=-1)
    norm_error = error(joined, reference)
    torch.testing.assert_close(joined, reference, rtol=1e-3, atol=2e-3)
    export_kernel(norm, args.output / "aot" / "mtp_norm_concat")
    # Capture must match the real final norm exactly, including the FP32
    # residual. Changed-input replay also checks absolute-position placement.
    capture = speculation.capture_target_hidden(5120, 256)
    final = final_norm()
    capture_results = []
    for m in (1, 2, 4, 8, 16):
        x = torch.randn(m, 5120, device="cuda").half()
        residual = torch.randn(m, 5120, device="cuda")
        weight = (torch.randn(5120, device="cuda") * 0.1).half()
        step = torch.tensor([127 + m], device="cuda", dtype=torch.int32)
        target_store = torch.full((256, 5120), 123.0, device="cuda", dtype=torch.float16)
        expected = torch.empty_like(x)
        index = torch.arange(m, device="cuda", dtype=torch.int32)
        gather = speculation.gather_target_hidden(m, 5120, 256)
        gather_step = torch.tensor([127], device="cuda", dtype=torch.int32)
        gathered = torch.empty_like(x)
        launch(final, x, residual, index, weight, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(capture, x, residual, weight, step, target_store)
        graph.replay()
        torch.testing.assert_close(target_store[127 : 127 + m], expected, rtol=0, atol=0)
        assert (target_store[:127] == 123).all() and (target_store[127 + m :] == 123).all()
        x.mul_(0.5)
        residual.mul_(2)
        step.fill_(m)
        graph.replay()
        launch(final, x, residual, index, weight, expected)
        torch.testing.assert_close(target_store[:m], expected, rtol=0, atol=0)
        gather_step.zero_()
        launch(gather, target_store, gather_step, gathered)
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
        capture_results.append(
            dict(
                tokens=m,
                final_norm_bytes_equal=True,
                changed_input_and_position_replay=True,
                gather_bytes_equal=True,
            )
        )
        export_kernel(gather, args.output / "aot" / f"mtp_gather_m{m}")
    export_kernel(capture, args.output / "aot" / "mtp_capture")
    write_json(
        args.output / "result.json",
        dict(
            status="passed",
            environment=environment(),
            projections=results,
            gdn=gdn_results,
            attention=attention_results,
            mtp_norm=norm_error,
            mtp_capture=capture_results,
            scope="Operators only; exact M1 projection comparison and causal FP32 GDN prefix checks",
        ),
    )


if __name__ == "__main__":
    main()
