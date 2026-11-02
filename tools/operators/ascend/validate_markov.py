"""Compare shared Markov step bindings with published step-specialized kernels."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.deepseek_v41.dspark import MARKOV_STEP_OPS


def kernels(plan):
    model = json.loads(plan.read_text())
    found = {}
    for packet in model["ranks"][0]["packets"]:
        for step in packet["body"]:
            if step["op"] != "kernel":
                continue
            path = plan.parent / step["manifest"]
            spec = json.loads(path.read_text())
            if spec["op"] in MARKOV_STEP_OPS:
                key = (spec["op"], spec["parameters"].get("step"))
                if key in found and found[key][0] != path:
                    raise ValueError("Markov step has multiple implementations")
                found[key] = (path, spec)
    return found


def entry(path, spec):
    library = ctypes.CDLL(str((path.parent / spec["library"]["file"]).resolve()))
    function = library.orinfer_ascend_kernel_v2
    function.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int32

    def launch(tensors, rows, stream):
        pointers = (ctypes.c_void_p * len(tensors))(*(t.data_ptr() for t in tensors))
        status = function(pointers, len(tensors), rows, stream)
        if status:
            raise RuntimeError(f"Markov launch failed: {status}")

    launch.library = library
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    old, new, reports = kernels(args.baseline), kernels(args.plan), []
    if set(new) != {(op, None) for op in MARKOV_STEP_OPS}:
        raise ValueError("Require exactly one shared implementation per Markov operator")
    for (op, _), (path, spec) in sorted(new.items()):
        launch = entry(path, spec)
        capacity = spec["parameters"]["_row_capacity"]
        stride = spec["parameters"]["block_size"]
        slots = spec["operands"][:-1]
        output_slots = [i for i, slot in enumerate(slots) if slot["access"] != "read"]
        tensors = []
        for slot in slots:
            shape = list(slot["shape"])
            if slot["access"] != "read":
                shape[0] += 1
            dtype = dict(f32=torch.float32, bf16=torch.bfloat16, i32=torch.int32)[slot["dtype"]]
            tensor = (
                torch.randn(shape).to(dtype)
                if dtype != torch.int32
                else torch.zeros(shape, dtype=dtype)
            )
            if op == "topk.reduce_rank_best" and slot["name"] == "packed":
                tensor[..., 1] = torch.arange(shape[0])[:, None]
            tensors.append(tensor.npu())
        reference = [t.clone() for t in tensors]
        step_index = torch.zeros(1, dtype=torch.int32, device="npu")
        stream = torch.npu.current_stream()._as_parameter_
        for rows in sorted({1, min(3, capacity), capacity}):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                launch([*tensors, step_index], rows, torch.npu.current_stream()._as_parameter_)
            for step in sorted({0, min(4, stride - 1), min(7, stride - 1), stride - 1}):
                baseline_path, baseline_spec = old[(op, step)]
                baseline = entry(baseline_path, baseline_spec)
                tensors[0].neg_()
                for i, tensor in enumerate(tensors):
                    if i in output_slots:
                        tensor.fill_(17)
                    reference[i].copy_(tensor)
                step_index.fill_(step)
                graph.replay()
                baseline(reference, rows, stream)
                torch.npu.synchronize()
                for i in output_slots:
                    torch.testing.assert_close(tensors[i], reference[i], atol=0, rtol=0)
                reports.append(
                    dict(op=op, rows=rows, step=step, exact=True, changed_input_graph=True)
                )
                print(json.dumps(reports[-1]), flush=True)
    atomic_json(args.output, reports)


if __name__ == "__main__":
    main()
