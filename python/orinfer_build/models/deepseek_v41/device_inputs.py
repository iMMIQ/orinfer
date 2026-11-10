"""Attach shared device continuation kernels to the existing A2 packet plan."""

import argparse
import json
from pathlib import Path

from orinfer_build.common.files import atomic_json
from orinfer_build.common.shared_json import decode


def attach(model, base):
    from kernels.ascend import frontier
    from kernels.shared.rows import row_count
    from orinfer_build.compiler.ascend.build import compile_kernel, operand

    owners = model["batch"]["owners"]
    outputs = model["batch"].get("output_capacity") or owners
    capacity = max(p["rows"] for r in model["ranks"] for p in r["packets"])
    engram = json.loads((base / model["engram"]["file"]).read_text())
    if len(engram["barriers"]) != 2:
        raise ValueError("Device hashing requires the DSv4.1 image barrier pair")
    if not model.get("sampling") or capacity > 4096 or owners > 128:
        raise ValueError("Device continuation requires bounded packets and device sampling")

    def build(name, cap, params, ports):
        function = getattr(frontier, name)(rows=row_count(cap, True), **params)
        buffers = [function.buffer_map[p] for p in function.params if p in function.buffer_map]
        types = {
            "int32": "i32",
            "int64": "i64",
            "uint64": "u64",
            "bfloat16": "bf16",
            "float32": "f32",
        }
        arguments = [
            operand(str(b.name), types[str(b.dtype)], shape, access)
            for b, (shape, access) in zip(buffers, ports, strict=True)
        ]
        path = compile_kernel(
            function,
            "frontier." + name,
            arguments,
            params,
            base / "cache",
            row_capacity=cap,
        )
        return str(path.relative_to(base))

    def row_ports(cap):
        return [
            ([cap, 8], "read"),
            ([owners, 4], "read"),
            ([owners, 16], "read"),
            ([cap], "write"),
            ([cap], "write"),
        ]

    preparations, embeddings = {}, {}
    for rank in model["ranks"]:
        for packet in rank["packets"]:
            cap = packet["rows"]
            if packet.get("decoder_coordinates"):
                # The runtime copies these already-computed lengths at the
                # embedding boundary; pure continuations share both row domains.
                embedding = next(
                    (i for i, step in enumerate(packet["body"]) if step["op"] == "embed"), 0
                )
                if embedding < 2:
                    raise ValueError("Device input lengths require the causal embedding boundary")
                for ratio, step in enumerate(packet["body"][embedding - 2 : embedding], 1):
                    spec = json.loads((base / step["manifest"]).read_text())
                    if (
                        spec["op"] != "model_ops.causal_indices"
                        or spec["parameters"]["ratio"] != ratio
                        or step["buffers"][0] != packet["coordinates"]["slots"]
                        or step["buffers"][2] != packet["coordinates"][f"c{ratio}_lengths"]
                    ):
                        raise ValueError("Device input lengths require the shared causal prologue")
            for coord in [packet["coordinates"], packet.get("decoder_coordinates")]:
                if coord is None:
                    continue
                if not coord.get("device_indices", False) and not coord.get("draft_block"):
                    raise ValueError("Target continuation requires device causal indices")
                draft = coord.get("draft_block", 0)
                source = bool(coord.get("source_swa_indices"))
                key = f"{cap}:{draft}:{int(source)}"
                if key not in preparations:
                    tokens = build("tokens", cap, dict(owners=owners), row_ports(cap))
                    params = dict(
                        context=model["capacity"],
                        draft=int(bool(draft)),
                        storage=coord.get("draft_swa_storage_rows", 128),
                        source=source,
                    )
                    coordinates = build(
                        "coordinates",
                        cap,
                        params,
                        [
                            ([cap, 8], "read"),
                            ([cap], "read"),
                            ([4, model["capacity"], 64], "read"),
                            *[([cap, 64], "write") for _ in range(6)],
                            ([cap], "write"),
                            ([cap], "write"),
                            ([cap, 128], "write"),
                            ([cap, 128], "write"),
                            ([cap, 8], "write"),
                            ([cap, 8], "write"),
                        ],
                    )
                    preparations[key] = dict(tokens=tokens, coordinates=coordinates)
            for instruction in packet["body"]:
                if instruction["op"] != "embed":
                    continue
                if packet.get("role") == "draft":
                    if (
                        instruction.get("tensor") != "mtp.0.embed.weight"
                        or instruction.get("columns", 5120) != 5120
                        or instruction.get("first_column", 0)
                    ):
                        raise ValueError(
                            "Draft input cache requires the full MTP embedding boundary"
                        )
                    continue
                width = instruction.get("columns", 5120)
                key = f"{cap}:{width}"
                if key not in embeddings:
                    embeddings[key] = build(
                        "embedding",
                        cap,
                        dict(vocab=129280, width=width),
                        [
                            ([129280, width], "read"),
                            ([cap], "read"),
                            ([cap, width], "write"),
                        ],
                    )
    local_vocab = model["sampling"]["vocab"]
    if local_vocab * len(model["ranks"]) != 129280:
        raise ValueError("Draft input vocabulary shards differ from the sampler")
    draft_capacities = {
        p["rows"] for r in model["ranks"] for p in r["packets"] if p.get("role") == "draft"
    }
    if len(draft_capacities) != 1:
        raise ValueError("Draft embedding requires one shared dynamic row capacity")
    draft_capacity = draft_capacities.pop()
    draft = build(
        "draft_embedding",
        draft_capacity,
        dict(owners=owners),
        [
            ([draft_capacity, 8], "read"),
            ([owners, 5120], "read"),
            ([5120], "read"),
            ([draft_capacity, 5120], "write"),
        ],
    )
    merge = build(
        "merge",
        outputs,
        dict(capacity=outputs, ranks=len(model["ranks"]), vocab=model["sampling"]["vocab"]),
        [
            ([len(model["ranks"]) * outputs * 2], "read"),
            ([outputs, 2], "write"),
        ],
    )
    commit = build(
        "commit",
        owners,
        dict(owners=owners, capacity=outputs, input_capacity=capacity, vocab=local_vocab),
        [
            ([owners, 8], "read"),
            ([outputs, 2], "read"),
            ([capacity], "read"),
            ([owners, 16], "read"),
            ([owners, 4], "write"),
            ([capacity], "read"),
            ([owners, 4], "read_write"),
            ([local_vocab, 5120], "read"),
            ([owners, 5120], "write"),
        ],
    )
    scatter = build(
        "scatter_embedding",
        owners,
        dict(owners=owners),
        [([owners, 8], "read"), ([owners, 5120], "read"), ([owners, 5120], "read_write")],
    )
    hash_params = dict(
        owners=owners,
        layers=len(engram["layers"]),
        heads=engram["heads"],
        vocab=len(engram["token_map"]),
        pad=engram["pad_token"],
        barrier_a=engram["barriers"][0],
        barrier_b=engram["barriers"][1],
    )
    hashes = build(
        "hashes",
        capacity,
        hash_params,
        [
            ([capacity, 8], "read"),
            ([capacity], "read"),
            ([owners, 4], "read"),
            ([hash_params["vocab"]], "read"),
            ([hash_params["layers"], 4], "read"),
            ([hash_params["layers"], hash_params["heads"] * 3, 2], "read"),
            ([capacity, hash_params["layers"] * hash_params["heads"] * 3], "write"),
            ([capacity], "write"),
        ],
    )
    strides = {
        p["draft_stride"] for r in model["ranks"] for p in r["packets"] if p.get("role") == "draft"
    }
    if len(strides) != 1:
        raise ValueError("Device continuation requires one physical draft stride")
    stride = strides.pop()
    save_draft = build(
        "save_draft",
        owners,
        dict(owners=owners, capacity=owners * stride, stride=stride),
        [
            ([owners, 8], "read"),
            ([owners * stride], "read"),
            ([owners, 16], "write"),
        ],
    )
    result = dict(
        capacity=capacity,
        preparations=preparations,
        embeddings=embeddings,
        merge=merge,
        scatter_embedding=scatter,
        draft_embedding=draft,
        commit=commit,
        save_draft=save_draft,
        hashes=hashes,
        hash_width=len(engram["layers"]) * engram["heads"] * 3,
    )
    model["device_inputs"] = result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Device-input build requires a new output directory")
    from orinfer_build.models.deepseek_v41.assembly import publish_package

    model = decode(json.loads(args.plan.read_text()))
    model.pop("device_inputs", None)
    publish_package(model, args.plan.parent, args.output)
    attach(model, args.output)
    atomic_json(args.output / "model.json", model, compact=True)


if __name__ == "__main__":
    main()
