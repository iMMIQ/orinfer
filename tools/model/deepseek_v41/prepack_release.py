"""Prepare one resident transposed-NZ W4 representation from a native release.

Unchanged host/shared shards are hardlinked. The standard layer/rank GPU shards
retain their semantic grouping, with weights and execution packages separate.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from tools.model.deepseek_v41.export_release import Export
from tools.model.deepseek_v41.weight_layout import transposed_nz_file_chunks


class PrepackRelease(Export):
    def plan(self):
        self.index = deepcopy(self.old_index)
        self.index["metadata"]["w4_layout"] = "transposed_nz"
        old_manifest = json.loads((self.source / "release-manifest.json").read_text())
        self.old_records = old_manifest["files"]
        self.manifest["logical_shapes"] = old_manifest["logical_shapes"]
        self.manifest["w4_layout"] = "transposed_nz"
        self.rewrite_groups = set()
        transforms = {}
        for rank in self.model["ranks"]:
            definitions = {b["name"]: b for b in rank["buffers"]}
            workspace_sizes = {}
            for packet in rank["packets"]:
                for step in packet["body"] + packet["head"]:
                    if step["op"] != "w4":
                        continue
                    if step.get("weight_layout", "nd") != "nd":
                        raise ValueError("Prepacking requires the original ND representation")
                    if step.get("expected_expert_rows") != 16:
                        raise ValueError("NZ capacity/workspace selection was measured with hint16")
                    e, k, n = step["experts"], step["input"], step["output"]
                    if e != 48 or (k, n) not in [(5120, 4608), (2304, 5120)]:
                        raise ValueError("Unmeasured W4 NZ geometry")
                    name = definitions[step["buffers"][1]]["prepared_tensor"]["name"]
                    geometry = [e, k, n]
                    if name in transforms and transforms[name] != geometry:
                        raise ValueError("Conflicting W4 physical shapes")
                    transforms[name] = geometry
                    definitions[step["buffers"][1]]["prepared_tensor"]["shape"] = [
                        e,
                        k // 64,
                        n // 16,
                        16,
                        32,
                    ]
                    step["weight_layout"] = "transposed_nz"
                    workspace = step["workspace"]
                    required = 32 << 20 if step["rows"] <= 192 else 320 << 20
                    workspace_sizes[workspace] = max(workspace_sizes.get(workspace, 0), required)
            for name, size in workspace_sizes.items():
                definitions[name]["bytes"] = size
        for group, record in self.old_records.items():
            path = self.source / group
            if path not in self.headers:
                from tools.model.safetensors_source import local_header

                self.headers[path] = local_header(path)
            start, header = self.headers[path]
            # Preserve physical order, not lexical tensor-name order.
            names = sorted(record["tensors"], key=lambda n: header[n]["data_offsets"][0])
            for name in names:
                spec = header[name]
                offset = start + spec["data_offsets"][0]
                source = dict(file=str(path), offset=offset)
                shape = spec["shape"]
                if name in transforms:
                    e, k, n = transforms[name]
                    if spec["dtype"] != "I8" or shape != [e, k, n // 2]:
                        raise ValueError("Original packed W4 tensor contract differs")
                    source = dict(nz_source=str(path), offset=offset, geometry=[e, k, n])
                    shape = [e, k // 64, n // 16, 16, 32]
                    self.rewrite_groups.add(group)
                self.groups[group].append(
                    (
                        name,
                        dict(
                            dtype=spec["dtype"],
                            shape=shape,
                            bytes=spec["data_offsets"][1] - spec["data_offsets"][0],
                            source=source,
                        ),
                    )
                )
        if len(transforms) != 640 or len(self.rewrite_groups) != 80:
            raise ValueError("Expected complete 40-layer TP8 up/down W4 banks in 80 GPU shards")
        return self

    def chunks(self, entry):
        source = entry["source"]
        if "nz_source" in source:
            yield from transposed_nz_file_chunks(
                source["nz_source"], source["offset"], *source["geometry"]
            )
        else:
            yield from super().chunks(entry)

    def write_shard(self, group):
        if group in self.rewrite_groups:
            return super().write_shard(group)
        target = self.output / group
        target.parent.mkdir(parents=True, exist_ok=True)
        target.hardlink_to(self.source / group)
        return group, deepcopy(self.old_records[group])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    job = PrepackRelease(
        args.source,
        args.output,
        execution_output=args.execution_output,
        source_execution=args.source_execution,
    ).plan()
    print(
        json.dumps(
            dict(
                shards=len(job.groups),
                rewritten_shards=len(job.rewrite_groups),
                reused_shards=len(job.groups) - len(job.rewrite_groups),
            )
        ),
        flush=True,
    )
    if not args.plan_only:
        job.export(args.workers)


if __name__ == "__main__":
    main()
