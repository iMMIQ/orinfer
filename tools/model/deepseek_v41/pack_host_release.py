"""Interleave exact host INT8 values and FP32 scales into one bounded row record."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from tools.model.deepseek_v41.export_release import Export
from tools.model.safetensors_source import local_header


def record_chunks(weight, scale, rows, width, block_rows=16384):
    """Two sequential input ranges produce unchanged value/scale bytes in row order."""
    if rows < 1 or width < 1 or width % 32 or block_rows < 1:
        raise ValueError("Host records require positive group-32 INT8 rows")
    scale_bytes = width // 32 * 4
    with Path(weight[0]).open("rb", buffering=8 << 20) as w:
        with Path(scale[0]).open("rb", buffering=8 << 20) as s:
            w.seek(weight[1])
            s.seek(scale[1])
            for first in range(0, rows, block_rows):
                count = min(block_rows, rows - first)
                values = w.read(count * width)
                scales = s.read(count * scale_bytes)
                if len(values) != count * width or len(scales) != count * scale_bytes:
                    raise EOFError("Truncated host INT8 value/scale range")
                out = np.empty((count, width + scale_bytes), dtype=np.uint8)
                out[:, :width] = np.frombuffer(values, dtype=np.uint8).reshape(count, width)
                out[:, width:] = np.frombuffer(scales, dtype=np.uint8).reshape(count, scale_bytes)
                yield memoryview(out).cast("B")


class HostRecordRelease(Export):
    def plan(self):
        self.index = deepcopy(self.old_index)
        self.index["metadata"]["host_row_layout"] = "i8-f32-g32-v1"
        if self.index.get("row_records"):
            raise ValueError("Host records are already packed")
        self.index["row_records"] = {}
        old = json.loads((self.source / "release-manifest.json").read_text())
        self.manifest.update({k: v for k, v in old.items() if k != "files"})
        self.manifest["source_package"] = self.model_source_digest()
        self.manifest["host_row_layout"] = "i8-f32-g32-v1"
        self.old_records = old["files"]
        self.rewrite_groups = set()
        replacements = {}
        removed = set()
        for logical, values in list(self.index["row_shards"].items()):
            if not logical.endswith(".weight"):
                continue
            prefix = logical.removesuffix(".weight")
            scales = self.index["row_shards"].get(prefix + ".scale")
            if not scales or len(values) != len(scales):
                raise ValueError("Host value/scale shard inventory differs")
            parts = []
            for i, (value, scale) in enumerate(zip(values, scales, strict=True)):
                group = self.index["weight_map"][value]
                if group != self.index["weight_map"][scale]:
                    raise ValueError("Host value/scale pair must share one semantic shard")
                path = self.source / group
                if path not in self.headers:
                    self.headers[path] = local_header(path)
                start, header = self.headers[path]
                w, s = header[value], header[scale]
                rows, width = w["shape"]
                if (
                    w["dtype"] != "I8"
                    or s["dtype"] != "F32"
                    or s["shape"] != [rows, width // 32]
                    or width % 32
                ):
                    raise ValueError("Host value/scale dtype or geometry differs")
                name = prefix + f".records.part{i:04}"
                entry = dict(
                    dtype="U8",
                    shape=[rows, width + width // 32 * 4],
                    bytes=rows * (width + width // 32 * 4),
                    source=dict(
                        row_record=dict(
                            weight=[str(path), start + w["data_offsets"][0]],
                            scale=[str(path), start + s["data_offsets"][0]],
                            rows=rows,
                            width=width,
                        )
                    ),
                )
                replacements[value] = (name, entry)
                removed.add(scale)
                self.index["weight_map"].pop(value)
                self.index["weight_map"].pop(scale)
                self.index["weight_map"][name] = group
                self.rewrite_groups.add(group)
                parts.append(name)
            self.index["row_records"][prefix] = dict(format="i8-f32-g32-v1", parts=parts)
            self.index["row_shards"].pop(logical)
            self.index["row_shards"].pop(prefix + ".scale")
        if not replacements:
            raise ValueError("Missing split host row tables")
        for group, record in self.old_records.items():
            path = self.source / group
            if path not in self.headers:
                self.headers[path] = local_header(path)
            start, header = self.headers[path]
            for name in sorted(record["tensors"], key=lambda n: header[n]["data_offsets"][0]):
                if name in removed:
                    continue
                if name in replacements:
                    self.groups[group].append(replacements[name])
                else:
                    spec = header[name]
                    self.groups[group].append(
                        (
                            name,
                            dict(
                                dtype=spec["dtype"],
                                shape=spec["shape"],
                                bytes=spec["data_offsets"][1] - spec["data_offsets"][0],
                                source=dict(file=str(path), offset=start + spec["data_offsets"][0]),
                            ),
                        )
                    )
        return self

    def model_source_digest(self):
        return json.loads((self.source / "cache/model.json").read_text())["execution_package"]

    def chunks(self, entry):
        if "row_record" in entry["source"]:
            yield from record_chunks(**entry["source"]["row_record"])
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
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-execution", "output", "execution-output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    job = HostRecordRelease(
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
