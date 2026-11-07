"""Verify compact dynamic prefill strides, state writes and graph replay.

Each candidate is compiled once and compared with fixed-shape kernels over
multiple non-aligned lengths. This is execution equivalence, not quantization
quality or a substitute for the Rust model's end-to-end performance gate.
"""

import argparse
from pathlib import Path

import torch
import tilelang
from tvm import tirx as tir

from kernels.cuda.model import control, flash_control, flash_next, hc_prefill, ple, qsa
from kernels.cuda.model.e8p_prefill import split_prefill_a8
from kernels.cuda.model.gdn_sequence import _gdn_sequence
from kernels.cuda.model.qsa_attention import sparse_attention
from kernels.cuda.operators.op08_gdn_conv_prep import gdn_conv_prep
from orinfer_build.compiler.cuda.common import configure, environment, export_kernel, write_json


def arguments(function, rows):
    """Materialize compact views from the exported primitive's real shapes."""
    result = []
    for parameter in function.params:
        if parameter not in function.buffer_map:
            result.append(rows)
            continue
        buffer = function.buffer_map[parameter]
        shape = []
        for expression in buffer.shape:
            replacements = {}

            def visit(node):
                if isinstance(node, tir.Var):
                    if node.name != "m":
                        raise ValueError(f"Unexpected dynamic dimension {node.name}")
                    replacements[node] = tir.IntImm(node.dtype, rows)

            tir.stmt_functor.post_order_visit(expression, visit)
            expression = tir.stmt_functor.substitute(expression, replacements)
            import tvm

            shape.append(int(tvm.arith.Analyzer().simplify(expression)))
        dtype = getattr(torch, str(buffer.dtype))
        if dtype.is_floating_point:
            value = (torch.rand(shape, device="cuda") * 0.02 - 0.01).to(dtype)
            if buffer.name == "G":
                value = -value.abs()
            if buffer.name in ("Beta", "Den", "InvRms"):
                value = value.abs() + 0.1
        else:
            value = torch.zeros(shape, device="cuda", dtype=dtype)
            if buffer.name == "A":
                value = torch.randint(-128, 128, shape, device="cuda", dtype=dtype)
            if buffer.name in ("Position", "Step"):
                value.fill_(37)
            if buffer.name == "positions":
                value.fill_(37)
            if buffer.name == "lengths":
                value.fill_(rows)
        result.append(value)
    return result


def launch(kernel, inputs):
    kernel.adapter.func(*inputs, stream=torch.cuda.current_stream().cuda_stream)


def copy_arguments(inputs):
    return [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]


def check_equal(actual, expected, label):
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        if isinstance(a, torch.Tensor) and not torch.equal(a, b):
            raise AssertionError(f"{label}: operand/state {index} differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 2, 15, 17, 129, 421, 512])
    parser.add_argument("--operators", nargs="+", help="Run only named operator checks")
    args = parser.parse_args()
    if any(not 1 <= n <= 4096 for n in args.lengths):
        parser.error("Lengths must be within the target prefill workspace")
    configure()
    report = dict(complete=False, environment=environment(), cases=[])
    constructors = {
        "gdn": lambda n, dynamic: _gdn_sequence(n, in_place=True, dynamic_rows=dynamic),
        "gdn-prefetch": lambda n, dynamic: _gdn_sequence(
            n,
            in_place=True,
            value_tile=16,
            threads=32,
            prefetch_qk=True,
            token_unroll=4,
            dynamic_rows=dynamic,
        ),
        "gdn-conv": lambda n, dynamic: gdn_conv_prep.__wrapped__(
            B=1,
            tokens=n,
            normalize_round_fp16=False,
            conv_product_round_fp16=False,
            weight_dtype="float32",
            dynamic_tokens=dynamic,
        ),
        "mtp-capture": lambda n, dynamic: flash_control.capture.__wrapped__(
            n, 128, dynamic_rows=dynamic
        ),
        "mtp-capture-4096": lambda n, dynamic: flash_control.capture.__wrapped__(
            n, 128, 4096, dynamic_rows=dynamic
        ),
        "ple-conv": lambda n, dynamic: ple.ple_conv.__wrapped__(n, 128, dynamic_rows=dynamic),
        "ple-history": lambda n, dynamic: ple.ple_history.__wrapped__(n, 128, dynamic_rows=dynamic),
        "hc-combine-norm": lambda n, dynamic: hc_prefill.hc_combine_norm_prefill.__wrapped__(
            n, 256, dynamic_rows=dynamic
        ),
        "hc-up-mix": lambda n, dynamic: hc_prefill.hc_up_mix_prefill.__wrapped__(
            n, 256, 320, dynamic_rows=dynamic
        ),
        "qsa-prepare": lambda n, dynamic: flash_next.qsa_prepare.__wrapped__(
            n, 8192, staged=True, is_neox_style=True, dynamic_rows=dynamic
        ),
        "qsa-compress": lambda n, dynamic: qsa.index_compress.__wrapped__(
            n, 8192, dynamic_rows=dynamic
        ),
        "qsa-pending": lambda n, dynamic: qsa.index_pending.__wrapped__(n, dynamic_rows=dynamic),
        "qsa-store": lambda n, dynamic: qsa.kv_store.__wrapped__(n, 8192, dynamic_rows=dynamic),
        "qsa-attention": lambda n, dynamic: sparse_attention.__wrapped__(
            n, 8192, packed=True, dynamic_rows=dynamic
        ),
        "qsa-merge": lambda n, dynamic: qsa.sparse_merge.__wrapped__(n, dynamic_rows=dynamic),
        "e8p-a8-planes": lambda n, dynamic: split_prefill_a8.__wrapped__(
            n, 640, dynamic_rows=dynamic
        ),
        "head-gather": lambda n, dynamic: flash_control.last_hidden.__wrapped__(
            n, 2560, dynamic_rows=dynamic
        ),
        "advance": lambda n, dynamic: control.advance.__wrapped__(n, dynamic_rows=dynamic),
    }

    def compile_function(function):
        return tilelang.compile(
            function,
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
            pass_configs=(
                {"tl.disable_thread_storage_sync": True, "tl.disable_shared_memory_reuse": True}
                if label == "gdn-prefetch"
                else {}
            ),
        )

    if args.operators and set(args.operators) - constructors.keys():
        parser.error("Unknown operator check")

    for label, factory in constructors.items():
        if args.operators and label not in args.operators:
            continue
        primitive = factory(4096, True)
        candidate = compile_function(primitive)
        export_kernel(candidate, args.output / "aot" / label)
        for rows in args.lengths:
            initial = arguments(primitive, rows)
            if label.startswith("mtp-capture"):
                initial[1].fill_(37 + rows)
            actual, expected = copy_arguments(initial), copy_arguments(initial)
            if label == "advance":
                expected[0].add_(rows)
            elif label.startswith("mtp-capture"):
                ring = expected[2].shape[0]
                tail = torch.arange(max(0, rows - ring), rows, device="cuda")
                expected[2][(tail + 37) % ring] = expected[0][tail]
            else:
                reference = compile_function(factory(rows, False))
                launch(reference, [x for x in expected if isinstance(x, torch.Tensor)])
            launch(candidate, actual)
            torch.cuda.synchronize()
            check_equal(actual, expected, f"{label}/{rows}")
            # Reset all operands, including mutated persistent state, before
            # graph capture and replay. Compact tensor addresses stay fixed.
            for a, original in zip(actual, initial, strict=True):
                if isinstance(a, torch.Tensor):
                    a.copy_(original)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(candidate, actual)
            for a, original in zip(actual, initial, strict=True):
                if isinstance(a, torch.Tensor):
                    a.copy_(original)
            graph.replay()
            torch.cuda.synchronize()
            check_equal(actual, expected, f"{label}/{rows}/graph")
            if label.startswith("mtp-capture"):
                actual[0].neg_()
                expected[0].neg_()
                actual[1].add_(83)
                expected[1].add_(83)
                expected[2][(tail + 120) % ring] = expected[0][tail]
                graph.replay()
                torch.cuda.synchronize()
                check_equal(actual, expected, f"{label}/{rows}/changed-input-graph")
            row = dict(operator=label, rows=rows, graph=True, equal=True)
            report["cases"].append(row)
            write_json(args.output / "results.json", report)
            print("DYNAMIC PREFILL", label, rows, "equal", flush=True)
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
