"""Verify published query/head movement, padded tails and modified Graph inputs."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.row_contract import resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    model = json.loads(args.plan.read_text())
    peers = len(model["ranks"])
    entries, specs, layouts = {}, {}, []
    for rank in model["ranks"]:
        packet = next(p for p in rank["packets"] if any(s["op"] == "exchange" for s in p["body"]))
        selected = {}
        for step in packet["body"]:
            if step["op"] != "kernel":
                continue
            path = step["manifest"]
            if path not in specs:
                specs[path] = json.loads((args.plan.parent / path).read_text())
            spec = specs[path]
            p = spec["parameters"]
            kind = None
            if spec["op"] == "memory.partition_rows" and p.get("all_partitions"):
                kind = "restore_pack" if "row_expression" in step else "query_pack"
            elif (
                spec["op"] == "memory.partition_rows"
                and p.get("width") == 128
                and p.get("dtype") == "int32"
            ):
                kind = "swa"
            elif spec["op"] == "memory.gather_columns" and p["width"] == 32768:
                kind = "heads"
            elif spec["op"] == "memory.zero_row_tail" and p.get("first") is not None:
                kind = "tail"
            elif spec["op"] == "memory.interleave_rows" and p["width"] == 4096:
                kind = "restore"
            if kind is not None:
                selected[kind] = step
                if path not in entries:
                    library = ctypes.CDLL(
                        str((args.plan.parent / path).parent / spec["library"]["file"])
                    )
                    fn = library.orinfer_ascend_kernel_v2
                    fn.argtypes = [
                        ctypes.POINTER(ctypes.c_void_p),
                        ctypes.c_uint64,
                        ctypes.c_uint64,
                        ctypes.c_void_p,
                    ]
                    fn.restype = ctypes.c_int32
                    entries[path] = library, fn
        assert set(selected) == {"query_pack", "restore_pack", "swa", "heads", "tail", "restore"}
        layouts.append(selected)
    capacity = packet["rows"]
    local_capacity = capacity // peers
    source = [torch.empty((capacity, 4096), dtype=torch.bfloat16, device="npu") for _ in layouts]
    send = [torch.empty_like(v) for v in source]
    receive = [torch.empty_like(v) for v in source]
    output = [torch.empty_like(v) for v in source]
    original = [torch.empty_like(v) for v in source]
    swa = torch.arange(capacity * 128, dtype=torch.int32).reshape(capacity, 128).npu()
    positions = torch.arange(capacity, dtype=torch.int32).npu()
    local_swa = [
        torch.empty((local_capacity, 128), dtype=torch.int32, device="npu") for _ in layouts
    ]

    def launch(step, values, rows):
        pointers = (ctypes.c_void_p * len(values))(*[v.data_ptr() for v in values])
        count = resolve(step["row_expression"], rows) if "row_expression" in step else rows
        status = entries[step["manifest"]][1](
            pointers, len(values), count, torch.npu.current_stream()._as_parameter_
        )
        if status:
            raise RuntimeError(f"Movement C ABI failed: {status}")

    reports = []
    for rows in [1, 7, 8, 9, 511, 512, 513, 2047, 2048]:
        count = (rows + peers - 1) // peers
        padded = count * peers

        def exchange():
            # Device copies simulate the peer permutation. Native eight-context
            # AlltoAllV and its Graph/count lifetime are validated separately.
            for dest in range(peers):
                for peer in range(peers):
                    receive[dest][peer * count : (peer + 1) * count].copy_(
                        send[peer][dest * count : (dest + 1) * count]
                    )

        def chain():
            for rank, layout in enumerate(layouts):
                launch(layout["query_pack"], [source[rank], send[rank]], rows)
                launch(layout["swa"], [swa, local_swa[rank]], rows)
            exchange()
            for rank, layout in enumerate(layouts):
                launch(layout["heads"], [receive[rank], source[rank]], rows)
                output[rank].fill_(17)
                actual = (rows + peers - 1 - rank) // peers
                if actual:
                    output[rank].view(local_capacity, 32768)[:actual].copy_(
                        source[rank].view(local_capacity, 32768)[:actual]
                    )
                launch(layout["tail"], [positions, output[rank]], rows)
                launch(layout["restore_pack"], [output[rank], send[rank]], rows)
            exchange()
            for rank, layout in enumerate(layouts):
                launch(layout["restore"], [receive[rank], output[rank]], rows)

        chain()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            chain()
        for phase in range(3):
            for rank in range(peers):
                source[rank].normal_()
                original[rank].copy_(source[rank])
            swa.add_(1)
            graph.replay()
            torch.npu.synchronize()
            for rank in range(peers):
                indices = torch.arange(count, dtype=torch.int64).npu() * peers + rank
                valid = indices < rows
                expected = torch.cat([v.index_select(0, indices) for v in original], 1)
                expected[~valid] = 0
                assert torch.equal(
                    source[rank].view(local_capacity, 32768)[:count].cpu(), expected.cpu()
                )
                expected_swa = swa.index_select(0, indices)
                expected_swa[~valid] = 0
                assert torch.equal(local_swa[rank][:count].cpu(), expected_swa.cpu())
                assert torch.equal(output[rank][:rows].cpu(), original[rank][:rows].cpu())
                if padded < capacity:
                    assert torch.all(output[rank][padded:] == 17)
            reports.append(dict(rows=rows, phase=phase, ranks=peers, exact=True))
            atomic_json(args.output / "validation.json", reports)
    print(
        f"Verified {len(reports)} movement/Graph phases across{peers} simulated query partitions",
        flush=True,
    )


if __name__ == "__main__":
    main()
