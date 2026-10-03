"""Attribute Nsight graph nodes to the actual AOT model plan (CPU only).

The runtime captures programs in sorted phase order on one stream. Original
node creation order is checked against every operation, then clone identities
map replay activity back to that plan. Request/launch counts must match exactly.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def summarize(directory, manifest_path):
    if manifest_path.is_dir():
        manifest_path = manifest_path / 'cache/manifest.json'
    manifest_raw = manifest_path.read_bytes()
    model = json.loads(manifest_raw)
    report = json.loads((directory / "report.json").read_text())
    assert report["manifest_sha256"] == hashlib.sha256(manifest_raw).hexdigest()
    requests = json.loads((directory / "requests.json").read_text())["requests"]
    assert len(requests) == len(report["requests"])
    connection = sqlite3.connect(f"file:{directory / 'trace.sqlite'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    events = list(connection.execute("SELECT * FROM CUDA_GRAPH_NODE_EVENTS"))
    clones = {r["graphNodeId"]: r["originalGraphNodeId"] for r in events
              if r["originalGraphNodeId"] is not None}
    roots = collections.defaultdict(set)
    for row in events:
        node = row["graphNodeId"]
        if node not in clones:
            roots[node >> 32].add(node)
    phases = sorted(model["programs"])
    # Instantiation may also create driver-internal nodes without original IDs.
    # These are not captured programs; every executed node is still checked below.
    expected_counts = {len(model["programs"][phase]) for phase in phases}
    # Multiple fixed-shape plans can have the same operation count. Their
    # identities come from capture order (Rust BTreeMap), not counts alone;
    # each executed node is checked against its actual grid/block/shared below.
    roots = {base: nodes for base, nodes in roots.items() if len(nodes) in expected_counts}
    assert len(roots) == len(phases)
    identities = {}
    for (_, nodes), phase in zip(sorted(roots.items()), phases):
        ops = model["programs"][phase]
        assert len(nodes) == len(ops), (phase, len(nodes), len(ops))
        for node, index in zip(sorted(nodes), range(len(ops))):
            identities[node] = (phase, index)
    kernels = {k["name"]: k for k in model["kernels"]}
    launches = collections.defaultdict(list)
    for table, kind in [("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
                        ("CUPTI_ACTIVITY_KIND_MEMCPY", "copy"),
                        ("CUPTI_ACTIVITY_KIND_MEMSET", "zero")]:
        for raw in connection.execute(f"SELECT * FROM {table} WHERE graphNodeId IS NOT NULL"):
            row = dict(raw)
            node = row["graphNodeId"]
            seen = set()
            while node in clones:
                assert node not in seen
                seen.add(node)
                node = clones[node]
            phase, index = identities[node]
            op = model["programs"][phase][index]
            assert op["kind"] == kind
            family = kind
            if kind == "kernel":
                kernel = kernels[op["name"]]
                family = Path(kernel["module"]["file"]).parent.name
                assert [row[f"grid{axis}"] for axis in "XYZ"] == kernel["grid"]
                assert [row[f"block{axis}"] for axis in "XYZ"] == kernel["block"]
                assert row["dynamicSharedMemory"] == kernel["shared_memory_bytes"]
            else:
                assert row["bytes"] == op["bytes"]
            row.update(phase=phase, index=index, family=family,
                       ms=(row["end"] - row["start"]) / 1e6)
            launches[row["correlationId"]].append(row)
    ordered = []
    for correlation, rows in sorted(launches.items(), key=lambda item: min(r["start"] for r in item[1])):
        phase = rows[0]["phase"]
        assert {r["phase"] for r in rows} == {phase}
        assert sorted(r["index"] for r in rows) == list(range(len(model["programs"][phase])))
        families = collections.Counter()
        for row in rows:
            families[row["family"]] += row["ms"]
        span = (max(r["end"] for r in rows) - min(r["start"] for r in rows)) / 1e6
        ordered.append(dict(correlation_id=correlation, phase=phase, node_count=len(rows),
                            gpu_span_ms=span, gpu_active_ms=sum(families.values()),
                            families_ms=dict(families)))
    cursor = 0
    summaries = []
    for request, measured in zip(requests, report["requests"]):
        assert request["id"] == measured["id"]
        chunk = measured.get("prefill_chunk_tokens", model["chunk_tokens"])
        prefill_program = measured.get("prefill_program", "prefill")
        if model.get("prefill_plans"):
            plan = next(p for p in model["prefill_plans"]
                        if p["chunk_tokens"] == chunk and p["prefill_program"] == prefill_program)
            head_program = plan["head_program"]
        else:
            assert chunk == model["chunk_tokens"] and prefill_program == "prefill"
            head_program = "head"
        assert len(request["input_tokens"]) % chunk == 0
        nprefill = len(request["input_tokens"]) // chunk
        ndecode = request["max_new_tokens"] - 1
        sequence = [prefill_program] * nprefill + [head_program] + ["decode"] * ndecode
        subset = ordered[cursor:cursor + len(sequence)]
        assert [l["phase"] for l in subset] == sequence
        cursor += len(sequence)
        phases_summary = {}
        for phase in ["prefill", "head", "decode"]:
            program = dict(prefill=prefill_program, head=head_program, decode="decode")[phase]
            selected = [l for l in subset if l["phase"] == program]
            if not selected:
                continue
            divisor = len(selected) if phase == "decode" else 1
            families = collections.Counter()
            for launch in selected:
                families.update(launch["families_ms"])
            span = sum(l["gpu_span_ms"] for l in selected) / divisor
            active = sum(l["gpu_active_ms"] for l in selected) / divisor
            wall = measured[f"{phase}_s"] * 1000 / divisor
            phases_summary[phase] = dict(launch_count=len(selected), gpu_span_ms=span,
                                         gpu_active_ms=active, gpu_inter_node_gap_ms=span-active,
                                         profiled_wall_ms=wall,
                                         families_ms={k: v / divisor for k, v in sorted(families.items(), key=lambda kv: -kv[1])})
        summaries.append(dict(id=request["id"], input_tokens=len(request["input_tokens"]),
                              output_tokens=request["max_new_tokens"], phases=phases_summary))
    assert cursor == len(ordered)
    buffers = {b["name"]: b for b in model["buffers"]}
    dtype_bytes = dict(u8=1, i8=1, f16=2, i32=4, f32=4)
    def size(name):
        b = buffers[name]
        return math.prod(b["shape"]) * dtype_bytes[b["dtype"]]
    expansion_elements = 0
    expansion_weights = set()
    decode_weights = set()
    # Expansion parameter/byte counts are per plan; all fixed-M plans must
    # share the same logical projection weights. Count the default once.
    for phase in ["prefill", "decode"]:
        for op in model["programs"][phase]:
            if op["kind"] != "kernel":
                continue
            kernel = kernels[op["name"]]
            args = [a["name"] for a in kernel["args"] if a["kind"] == "buffer"]
            if phase == "decode":
                decode_weights.update(n for n in args if buffers[n]["access"] == "read" and not n.startswith("Embedding_"))
            elif (Path(kernel["module"]["file"]).parent.name == "w8_expand"
                  or Path(kernel["module"]["file"]).parent.name.startswith("w8_expand_")
                  or kernel["symbol"] == "i8_expand_kernel"):
                packed = next(n for n in args if n.endswith("_P"))
                # Warp packing exposes uint32 words as i32 in the Rust ABI;
                # physical words contain eight nibbles rather than two.
                assert buffers[packed]['layout'] in ('contiguous','u4_warp_n64_k128_mma_f16',
                                                     'u4_warp_n64_k128_mma_i8')
                expansion_elements += size(packed) * 2
                expansion_weights.update(n for n in args if n.endswith(("_P", "_S", "_Z")))
    counters = dict(big_projection_parameters=expansion_elements,
                    big_projection_operations_per_input=2 * expansion_elements,
                    prefill_chunk_tokens=model['chunk_tokens'],
                    w4_expansion_source_bytes_per_chunk=sum(size(n) for n in expansion_weights),
                    w8_expansion_write_bytes_per_chunk=expansion_elements,
                    decode_hot_weight_bytes=sum(size(n) for n in decode_weights),
                    nominal_bandwidth_GB_s=204.8,
                    synthetic_int8_TOPS=84.70935063039562)
    return dict(status="measured_diagnostic_profile", manifest_sha256=report["manifest_sha256"],
                scope="Nsight Systems CUDA graph node trace; prefill totals, decode mean per step; not formal TPS acceptance",
                checks="All replay nodes mapped exactly once; kernel grid/block/shared and copy sizes match manifest",
                requests=summaries, counters=counters, launches=ordered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory, args.manifest)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "requests": [r["id"] for r in result["requests"]], "counters": result["counters"]}, indent=2))


if __name__ == "__main__":
    main()
