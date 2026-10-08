"""Offline full-model exact state lifecycle and paged/token-major bridge checks."""

import argparse
import gc
import re
import time
from pathlib import Path
import torch
from common import ROOT, benchmark, configure, environment, export_kernel, identity, write_json
from tools.operators.abi import parse_host
from kernels.operators.op27_state_lifecycle import (
    GDN_WORDS,
    CONV_WORDS,
    request_word_copy,
    request_word_zero,
    paged_kv_gather,
    validate_request_map,
    validate_buffer_ranges,
    validate_paged_metadata,
)

BATCHES = (1, 2, 3, 4, 5, 7, 8)


def spans(*named):
    validate_buffer_ranges([(n, x.data_ptr(), x.numel() * x.element_size()) for n, x in named])


def invoke(kernel, *args):
    return kernel(*args, stream=torch.cuda.current_stream().cuda_stream)


def export(kernel, directory, spec):
    files = export_kernel(kernel, directory)
    host = (directory / "host.txt").read_text()
    cuda = (directory / "kernel.cu").read_text()
    info = {
        "operator": "op27_state_lifecycle",
        "sm": 87,
        "toolchain": environment(),
        "actual_launch_abi": parse_host(host),
        "actual_cuda_declarations": re.findall(r"__global__\s+void\s+\w+\s*\([^)]*\)", cuda),
        "workspace_bytes": 0,
        "cooperative_launch": False,
        "stream": "explicit capture-current stream on every call",
        **spec,
        **files,
    }
    write_json(directory / "abi.json", info)
    return info


def rejection_tests():
    tests = [
        ("negative_source", lambda: validate_request_map([-1], [0], 2, 2)),
        ("source_oob", lambda: validate_request_map([2], [0], 2, 2)),
        ("dest_oob", lambda: validate_request_map([0], [2], 2, 2)),
        ("dest_duplicate", lambda: validate_request_map([0, 1], [1, 1], 2, 2)),
        ("empty", lambda: validate_request_map([], [], 2, 2)),
        ("bool_index", lambda: validate_request_map([True], [0], 2, 2)),
        ("alias_exact", lambda: validate_buffer_ranges([("src", 1000, 100), ("dst", 1000, 100)])),
        ("alias_partial", lambda: validate_buffer_ranges([("src", 1000, 100), ("dst", 1099, 100)])),
        ("overflow_span", lambda: validate_buffer_ranges([("src", 2**64 - 8, 16)])),
        ("page_negative", lambda: validate_paged_metadata([[-1]], [1], 1, 128)),
        ("page_oob", lambda: validate_paged_metadata([[1]], [1], 1, 128)),
        ("length_oob", lambda: validate_paged_metadata([[0]], [129], 1, 128)),
        ("table_short", lambda: validate_paged_metadata([[0]], [129], 1, 256)),
        ("ragged_table", lambda: validate_paged_metadata([[0], [0, 0]], [1, 1], 1, 128)),
    ]
    rejected = []
    for name, call in tests:
        try:
            call()
        except ValueError as exc:
            rejected.append({"case": name, "reason": str(exc)})
        else:
            raise AssertionError(f"{name} should reject")
    validate_request_map([0, 0], [0, 1], 2, 2)
    validate_paged_metadata([[0, -1], [0, -1]], [128, 1], 1, 128)
    validate_paged_metadata([[-1]], [0], 1, 128)
    return rejected


def allocate_state(pool):
    state = torch.empty((48, pool, GDN_WORDS), dtype=torch.int32, device="cuda")
    conv = torch.empty((48, pool, CONV_WORDS), dtype=torch.int32, device="cuda")
    position = torch.empty((1, pool, 2), dtype=torch.int32, device="cuda")
    for x in (state, conv, position):
        x.random_(-2147483648, 2147483647)
    # Explicit NaN payloads, +/-Inf, signed zero and subnormal in every request/layer.
    specials = torch.tensor(
        [0, -2147483648, 0x7FC12345, 0x7F800000, -8388608, 1, 0x7FFFFFFF, -1],
        device="cuda",
        dtype=torch.int32,
    )
    state[:, :, :8] = specials
    # Position logical dtype is signed int64, including values > int32.
    position.view(torch.int64).reshape(-1).copy_(
        torch.arange(pool, device="cuda", dtype=torch.int64) * 2**33 + 8448
    )
    return state, conv, position


def expect_copy(src, dst, si, di, untouched=0x12345678):
    for s, d in zip(si, di):
        assert torch.equal(dst[:, d], src[:, s]), "bit-exact request copy failed"
    for d in set(range(dst.shape[1])) - set(di):
        assert bool((dst[:, d] == untouched).all()), "unselected request modified"


def state_case(b, kernels, repetitions):
    src = allocate_state(b + 1)
    dst = tuple(torch.full_like(x, 0x12345678) for x in src)
    si_cpu, di_cpu = list(range(b)), list(range(1, b + 1))
    validate_request_map(si_cpu, di_cpu, b + 1, b + 1)
    si = torch.tensor(si_cpu, dtype=torch.int32, device="cuda")
    di = torch.tensor(di_cpu, dtype=torch.int32, device="cuda")
    spans(
        *[(f"src{i}", x) for i, x in enumerate(src)],
        *[(f"dst{i}", x) for i, x in enumerate(dst)],
        ("si", si),
        ("di", di),
    )
    timings, graphs, first = {}, {}, {}
    for i, name in enumerate(("gdn", "conv", "position")):

        def call(i=i, name=name):
            return invoke(kernels[name], src[i], si, di, dst[i])

        begun = time.perf_counter()
        call()
        torch.cuda.synchronize()
        first[name] = time.perf_counter() - begun
        expect_copy(src[i], dst[i], si_cpu, di_cpu)
        timings[name], graphs[name] = benchmark(
            call, repetitions=repetitions, calls_per_replay=1 if name == "gdn" else 16
        )

    def combo():
        for i, name in enumerate(("gdn", "conv", "position")):
            invoke(kernels[name], src[i], si, di, dst[i])

    timings["complete_clone"], graph = benchmark(combo, repetitions=repetitions)
    # Source data AND device maps change at stable graph addresses.
    for x in src:
        x.bitwise_xor_(0x35713571)
    changed_si, changed_di = [(s + 1) % (b + 1) for s in si_cpu], list(reversed(di_cpu))
    validate_request_map(changed_si, changed_di, b + 1, b + 1)
    si.copy_(torch.tensor(changed_si, dtype=torch.int32, device="cuda"))
    di.copy_(torch.tensor(changed_di, dtype=torch.int32, device="cuda"))
    for x in dst:
        x.fill_(0x12345678)
    graph.replay()
    torch.cuda.synchronize()
    for x, y in zip(src, dst):
        expect_copy(x, y, changed_si, changed_di)
    for x in src:
        x.bitwise_xor_(0x35713571)
    si.copy_(torch.tensor(si_cpu, dtype=torch.int32, device="cuda"))
    di.copy_(torch.tensor(di_cpu, dtype=torch.int32, device="cuda"))
    for x in dst:
        x.fill_(0x12345678)
    graph.replay()
    torch.cuda.synchronize()
    for x, y in zip(src, dst):
        expect_copy(x, y, si_cpu, di_cpu)

    # Init writes only selected slots. All three initialization primitives timed.
    def init():
        for i, name in enumerate(("gdn", "conv", "position")):
            invoke(kernels["zero_" + name], di, dst[i])

    timings["complete_init"], init_graph = benchmark(init, repetitions=repetitions)
    for x in dst:
        x.fill_(0x12345678)
    init_graph.replay()
    torch.cuda.synchronize()
    for x in dst:
        assert bool((x[:, 1:] == 0).all()) and bool((x[:, 0] == 0x12345678).all())
    # Init graph consumes mutable request map, not a capture-time constant.
    changed_init = list(range(b))
    validate_request_map(changed_init, changed_init, b + 1, b + 1)
    di.copy_(torch.tensor(changed_init, dtype=torch.int32, device="cuda"))
    for x in dst:
        x.fill_(0x12345678)
    init_graph.replay()
    torch.cuda.synchronize()
    for x in dst:
        assert bool((x[:, :b] == 0).all()) and bool((x[:, b] == 0x12345678).all())
    di.copy_(torch.tensor(di_cpu, dtype=torch.int32, device="cuda"))
    for x in dst:
        x.fill_(0x12345678)
    init_graph.replay()
    torch.cuda.synchronize()
    for x in dst:
        assert bool((x[:, 1:] == 0).all()) and bool((x[:, 0] == 0x12345678).all())
    size = 48 * GDN_WORDS * 4
    return {
        "batch": b,
        "pool": b + 1,
        "gdn_bytes_per_request": size,
        "conv_bytes_per_request": 48 * CONV_WORDS * 4,
        "position_bytes_per_request": 8,
        "canonical_gdn_layout": "[48,pool,48,128,128] FP32 K,V",
        "canonical_conv_layout": "[48,pool,3,10240] FP16",
        "bit_exact": True,
        "nan_payload_inf_signedzero_subnormal": True,
        "request_isolation": True,
        "graph_changed_source_indices_poison_restore": True,
        "init_graph_changed_request_set_poison_restore": True,
        "first_launch_s": first,
        "timing": timings,
        "gdn_clone_target_ms": 3.0,
        "gdn_clone_budget_met": timings["gdn"]["median_ms"] <= 3.0 if b == 1 else None,
        "gdn_per_request_ms": timings["gdn"]["median_ms"] / b,
        "gdn_B1_target_applies_directly": b == 1,
        "allocated_state_bytes": sum(x.numel() * 4 for x in src + dst),
    }


def branch_restore(kernels, repetitions):
    source = allocate_state(3)
    checkpoint = tuple(
        torch.empty((x.shape[0], 2, x.shape[2]), dtype=x.dtype, device="cuda") for x in source
    )
    dest = tuple(torch.full_like(x, 0x12345678) for x in source)
    si = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
    ci = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    validate_request_map([1, 1], [0, 1], 3, 2)
    for i, name in enumerate(("gdn", "conv", "position")):
        invoke(kernels[name], source[i], si, ci, checkpoint[i])
        assert torch.equal(checkpoint[i][:, 0], checkpoint[i][:, 1])
        checkpoint[i][:, 0].bitwise_xor_(0x11111111)
        assert torch.equal(checkpoint[i][:, 1], source[i][:, 1])
        assert not torch.equal(checkpoint[i][:, 0], source[i][:, 1])
    ri = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    di = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    validate_request_map([1, 0], [0, 2], 2, 3)

    def restore():
        for i, name in enumerate(("gdn", "conv", "position")):
            invoke(kernels[name], checkpoint[i], ri, di, dest[i])

    timing, graph = benchmark(restore, repetitions=repetitions)
    for x in dest:
        x.fill_(0x12345678)
    graph.replay()
    torch.cuda.synchronize()
    for s, c, d in zip(source, checkpoint, dest):
        assert torch.equal(d[:, 0], s[:, 1]) and torch.equal(d[:, 2], c[:, 0])
        assert bool((d[:, 1] == 0x12345678).all())
    return {
        "full_layers": 48,
        "branches": 2,
        "checkpoint_then_private_mutation": True,
        "eviction_simulation": "destination buffers poisoned then restored from independent checkpoint",
        "checkpoint_restore_bit_exact": True,
        "branch_isolation": True,
        "complete_restore_timing": timing,
        "checkpoint_gdn_bytes": 2 * 48 * GDN_WORDS * 4,
    }


def endpoint_case(hidden_kernel, position_kernel, repetitions):
    """Complete cached endpoint needs hidden FP16[5120] and int64 position."""
    pool, count = 8, 7
    hidden = torch.empty((pool, 5120), device="cuda", dtype=torch.int16)
    hidden.random_(-32768, 32767)
    hidden[:, :8] = torch.tensor(
        [0, -32768, 0x7E55, -461, 0x7C00, -1024, 1, 0x7BFF], device="cuda", dtype=torch.int16
    )
    positions = torch.tensor(
        [-(2**63), 2**63 - 1, 0, 2**33, -(2**33), 8448, -1, -(2**63) + 1],
        device="cuda",
        dtype=torch.int64,
    )
    source = (
        hidden.view(torch.int32).reshape(1, pool, 2560),
        positions.view(torch.int32).reshape(1, pool, 2),
    )
    dest = tuple(torch.full_like(x, 0x12345678) for x in source)
    si_cpu, di_cpu = [7, 0, 6, 1, 3, 3, 2], list(range(count))
    validate_request_map(si_cpu, di_cpu, pool, pool)
    si = torch.tensor(si_cpu, device="cuda", dtype=torch.int32)
    di = torch.tensor(di_cpu, device="cuda", dtype=torch.int32)
    spans(
        ("hidden", source[0]),
        ("position", source[1]),
        ("hout", dest[0]),
        ("pout", dest[1]),
        ("si", si),
        ("di", di),
    )

    def copy():
        invoke(hidden_kernel, source[0], si, di, dest[0])
        invoke(position_kernel, source[1], si, di, dest[1])

    timing, graph = benchmark(copy, repetitions=repetitions, calls_per_replay=16)
    for x, y in zip(source, dest):
        expect_copy(x, y, si_cpu, di_cpu)
        x.bitwise_xor_(0x13571357)
    new_si, new_di = [(i + 1) % pool for i in si_cpu], list(reversed(di_cpu))
    validate_request_map(new_si, new_di, pool, pool)
    si.copy_(torch.tensor(new_si, device="cuda", dtype=torch.int32))
    di.copy_(torch.tensor(new_di, device="cuda", dtype=torch.int32))
    for y in dest:
        y.fill_(0x12345678)
    graph.replay()
    torch.cuda.synchronize()
    for x, y in zip(source, dest):
        expect_copy(x, y, new_si, new_di)
        x.bitwise_xor_(0x13571357)
    si.copy_(torch.tensor(si_cpu, device="cuda", dtype=torch.int32))
    di.copy_(torch.tensor(di_cpu, device="cuda", dtype=torch.int32))
    for y in dest:
        y.fill_(0x12345678)
    graph.replay()
    torch.cuda.synchronize()
    for x, y in zip(source, dest):
        expect_copy(x, y, si_cpu, di_cpu)
    return {
        "hidden_public_layout": "[B,5120] FP16",
        "hidden_raw_layout": "[1,B,2560] int32",
        "position_public_layout": "[B] int64",
        "position_raw_layout": "[1,B,2] int32",
        "hidden_bytes_per_request": 10240,
        "position_bytes_per_request": 8,
        "batch": count,
        "hidden_nan_payload_signedzero_subnormal_inf": True,
        "position_int64_min_max_and_beyond_int32": True,
        "bit_exact": True,
        "graph_changed_source_indices_poison_restore": True,
        "timing": timing,
        "scope": "endpoint hidden/position only; not included in GDN+conv+position clone timings",
    }


def kv_reference(k, v, table, lengths, tokens):
    ko = torch.zeros((len(lengths), tokens, 4, 256), dtype=torch.float16, device="cuda")
    vo = torch.zeros_like(ko)
    for b, length in enumerate(lengths):
        for t in range(0, length, 128):
            n = min(128, length - t)
            page = table[b][t // 128]
            ko[b, t : t + n] = k[page, :n]
            vo[b, t : t + n] = v[page, :n]
    return ko.view(torch.int32), vo.view(torch.int32)


def kv_case(b, valid, kernel, repetitions):
    tokens = ((valid + 127) // 128) * 128
    width = tokens // 128
    pages = 1 + b * width
    k = torch.full((pages, 128, 4, 256), float("nan"), dtype=torch.float16, device="cuda")
    v = torch.full_like(k, float("nan"))
    lengths = [max(0, valid - i * 37) for i in range(b)]
    if b == 7:
        lengths[-1] = 0
    table = [[0] + [1 + r * width + j for j in range(1, width)] for r in range(b)]
    # Shared prefix page only read. Fill only valid physical slots; pad retains NaN.
    for r, length in enumerate(lengths):
        for t in range(0, length, 128):
            n = min(128, length - t)
            page = table[r][t // 128]
            k[page, :n].normal_()
            v[page, :n].normal_()
    validate_paged_metadata(table, lengths, pages, tokens)
    pt = torch.tensor(table, dtype=torch.int32, device="cuda")
    sl = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    ko = torch.empty((b, tokens, 4, 256), dtype=torch.float16, device="cuda")
    vo = torch.empty_like(ko)
    spans(("kp", k), ("vp", v), ("table", pt), ("lengths", sl), ("ko", ko), ("vo", vo))
    kw, vw = (
        k.view(torch.int32).reshape(pages, 128, 512),
        v.view(torch.int32).reshape(pages, 128, 512),
    )
    kow, vow = (
        ko.view(torch.int32).reshape(b, tokens, 512),
        vo.view(torch.int32).reshape(b, tokens, 512),
    )

    def call():
        invoke(kernel, kw, vw, pt, sl, kow, vow)

    started = time.perf_counter()
    call()
    torch.cuda.synchronize()
    first = time.perf_counter() - started
    expected = kv_reference(k, v, table, lengths, tokens)
    assert torch.equal(ko.view(torch.int32), expected[0]) and torch.equal(
        vo.view(torch.int32), expected[1]
    )
    del expected
    timing, graph = benchmark(call, repetitions=repetitions, calls_per_replay=16)
    # Flip table order, shorten lengths, and change pages at fixed pointers.
    newtable = [list(reversed(row)) for row in table]
    newlengths = [max(0, length - 17) for length in lengths]
    validate_paged_metadata(newtable, newlengths, pages, tokens)
    k.view(torch.int32).bitwise_xor_(0x00110011)
    v.view(torch.int32).bitwise_xor_(0x00220022)
    pt.copy_(torch.tensor(newtable, device="cuda", dtype=torch.int32))
    sl.copy_(torch.tensor(newlengths, device="cuda", dtype=torch.int32))
    ko.fill_(float("nan"))
    vo.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    expected = kv_reference(k, v, newtable, newlengths, tokens)
    assert torch.equal(ko.view(torch.int32), expected[0]) and torch.equal(
        vo.view(torch.int32), expected[1]
    )
    del expected
    k.view(torch.int32).bitwise_xor_(0x00110011)
    v.view(torch.int32).bitwise_xor_(0x00220022)
    pt.copy_(torch.tensor(table, device="cuda", dtype=torch.int32))
    sl.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
    ko.fill_(float("nan"))
    vo.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    expected = kv_reference(k, v, table, lengths, tokens)
    assert torch.equal(ko.view(torch.int32), expected[0]) and torch.equal(
        vo.view(torch.int32), expected[1]
    )
    return {
        "batch": b,
        "valid_max": valid,
        "physical_tokens": tokens,
        "lengths": lengths,
        "bit_exact": True,
        "shared_prefix_readonly": True,
        "zero_length_request": 0 in lengths,
        "ragged_pad_zero_without_nanpad_read": True,
        "graph_changed_pages_table_lengths_poison_restore": True,
        "first_launch_s": first,
        "timing": timing,
        "output_bytes": ko.numel() * 4,
        "minimum_io_bytes": sum(lengths) * 8192 + (b * tokens - sum(lengths)) * 4096,
        "allocated_io_bytes": (k.numel() + v.numel() + ko.numel() + vo.numel()) * 2
        + pt.numel() * 4
        + sl.numel() * 4,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--block", type=int, default=4096)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configure()
    results = {
        "status": "in_progress",
        "environment": environment(),
        "cpu_rejections": rejection_tests(),
        "exports": [],
        "state_cases": [],
        "kv_cases": [],
    }
    kernels = {}
    for name, layers, words in [
        ("gdn", 48, GDN_WORDS),
        ("conv", 48, CONV_WORDS),
        ("position", 1, 2),
        ("endpoint_hidden", 1, 2560),
    ]:
        for zero in (False, True):
            key = ("zero_" if zero else "") + name
            started = time.perf_counter()
            kernels[key] = (request_word_zero if zero else request_word_copy)(
                layers, words, args.block
            )
            results["exports"].append(
                {
                    "kind": key,
                    "compile_s": time.perf_counter() - started,
                    "abi": export(
                        kernels[key],
                        output / "aot" / key,
                        {
                            "kind": key,
                            "layers": layers,
                            "words_per_request_layer": words,
                            "block_words": args.block,
                            "dtype": "int32 bitwords",
                            "batch": "dynamic count/source_pool/destination_pool",
                            "alias": "all allocation spans disjoint; unique destination request IDs",
                            "math": "integer bitwise copy" if not zero else "integer zero store",
                        },
                    ),
                }
            )
    started = time.perf_counter()
    gather = paged_kv_gather()
    results["exports"].append(
        {
            "kind": "paged_kv_gather",
            "compile_s": time.perf_counter() - started,
            "abi": export(
                gather,
                output / "aot" / "paged_kv_gather",
                {
                    "public_dtype": "float16",
                    "raw_dtype": "int32 paired FP16 words",
                    "public_pages": "[P,128,4,256]",
                    "public_output": "[B,Tphysical,4,256]",
                    "kv_layout": "token_major",
                    "dynamic": ["P", "B", "max_pages", "Tphysical"],
                    "block_words": 2048,
                    "metadata": "pageTable[B,max_pages]/seqLengths[B] int32, CPU validated",
                    "readonly_shared_pages": True,
                    "pad": "exact positive-zero without loading page",
                },
            ),
        }
    )
    write_json(output / "results.json", results)
    results["endpoint_case"] = endpoint_case(
        kernels["endpoint_hidden"], kernels["position"], args.repetitions
    )
    write_json(output / "results.json", results)
    for b in BATCHES:
        item = state_case(b, kernels, args.repetitions)
        results["state_cases"].append(item)
        write_json(output / "results.json", results)
        print(
            f"state B{b} bitexact gdn={item['timing']['gdn']['median_ms']:.6f}ms combo={item['timing']['complete_clone']['median_ms']:.6f}ms",
            flush=True,
        )
        gc.collect()
    results["branch_restore"] = branch_restore(kernels, args.repetitions)
    write_json(output / "results.json", results)
    for valid in (511, 512, 513, 2048, 8192, 8448, 8449):
        for b in BATCHES:
            item = kv_case(b, valid, gather, args.repetitions)
            results["kv_cases"].append(item)
            write_json(output / "results.json", results)
            print(f"kv B{b} T{valid} bitexact {item['timing']['median_ms']:.6f}ms", flush=True)
            gc.collect()
    results["memory"] = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    results["implementation_identity"] = [
        identity(ROOT / "kernels/operators/op27_state_lifecycle.py"),
        identity(Path(__file__)),
    ]
    results["status"] = (
        "passed exact state/branch/metadata/graph and paged gather checks; standalone primitives"
    )
    results["limits"] = [
        "No Rust scheduler, ownership, COW/refcounts, endpoint hidden state, or model prefix integration.",
        "3ms target applies to a single request 144MiB GDN clone/restore; multi-B per-request figure is diagnostic.",
        "All graph metadata changes require CPU revalidation before upload; shape change needs graph recapture.",
    ]
    write_json(output / "results.json", results)
    print("op27 complete; wrapper releases GPU lock after cleanup", flush=True)


if __name__ == "__main__":
    main()
