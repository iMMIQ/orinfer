"""Fetch only original vision tensors from a pinned HF safetensors checkpoint."""

import argparse
import json
from pathlib import Path
import tempfile
import urllib.request

from tools.model.publication import file_hash, write_json
from tools.model.safetensors_source import Source


def fetch_processor(repo, revision, destination):
    output = destination / "preprocessor_config.json"
    if output.is_file():
        raw = output.read_bytes()
    else:
        with urllib.request.urlopen(
            f"https://huggingface.co/{repo}/resolve/{revision}/preprocessor_config.json", timeout=60
        ) as response:
            raw = response.read(65537)
    if len(raw) > 65536:
        raise ValueError("Oversized image processor configuration")
    config = json.loads(raw)
    expected = dict(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        size=dict(longest_edge=16777216, shortest_edge=65536),
    )
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported checkpoint image processor configuration")
    defaults = dict(
        do_resize=True,
        do_normalize=True,
        do_rescale=True,
        do_convert_rgb=True,
        rescale_factor=1 / 255,
        resample=3,
    )
    if any(config.get(key, value) != value for key, value in defaults.items()):
        raise ValueError("Unsupported checkpoint image processor arithmetic")
    if not output.is_file():
        output.write_bytes(raw)
    return output


def fetch_vision(config, destination):
    quant = config["quantization_config"]
    repo, revision = quant["source"], quant["source_revision"]
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "model.safetensors"
    identity = destination / "source.json"
    if output.exists():
        recorded = json.loads(identity.read_text())
        if (recorded["repo"], recorded["revision"], recorded["sha256"]) != (
            repo,
            revision,
            file_hash(output),
        ):
            raise ValueError("Cached vision weights differ from the pinned source")
        fetch_processor(repo, revision, destination)
        return output
    # Source validates the repository, pinned revision and every tensor extent.
    with urllib.request.urlopen(
        f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors.index.json", timeout=60
    ) as response:
        index = response.read(16 * 1024 * 1024 + 1)
    if len(index) > 16 * 1024 * 1024:
        raise ValueError("Oversized checkpoint index")
    index_path = destination / "model.safetensors.index.json"
    index_path.write_bytes(index)
    source = Source(index_path, repo=repo, revision=revision, cache=destination / "headers")
    names = sorted(n for n in source.weight_map if n.startswith("model.visual."))
    if not names:
        raise ValueError("Checkpoint has no vision tensors")
    tensors = [(name, *source.tensor(name)) for name in names]
    if any(entry["dtype"] != "BF16" for _, _, _, entry in tensors):
        raise ValueError("Expected original BF16 vision weights")
    # Merge adjacent source ranges; keep transfer memory bounded to 8 MiB.
    tensors.sort(key=lambda item: (item[1], item[3]["data_offsets"][0]))
    header, ranges, offset = {}, [], 0
    for name, filename, begin, entry in tensors:
        start, stop = (begin + v for v in entry["data_offsets"])
        size = stop - start
        header[name] = dict(
            dtype=entry["dtype"], shape=entry["shape"], data_offsets=[offset, offset + size]
        )
        offset += size
        if ranges and ranges[-1][0] == filename and ranges[-1][2] == start:
            ranges[-1] = (filename, ranges[-1][1], stop)
        else:
            ranges.append((filename, start, stop))
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(len(raw_header).to_bytes(8, "little"))
            stream.write(raw_header)
            downloaded = 0
            for filename, start, stop in ranges:
                while start < stop:
                    count = min(8 * 1024 * 1024, stop - start)
                    stream.write(source.read(filename, start, count))
                    start += count
                    downloaded += count
                    print(f"VISION DOWNLOAD {downloaded}/{offset}", flush=True)
        digest = file_hash(temporary)
        temporary.replace(output)
        write_json(
            identity, dict(repo=repo, revision=revision, sha256=digest, parameters=offset // 2)
        )
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    fetch_processor(repo, revision, destination)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(fetch_vision(json.loads((args.model / "config.json").read_text()), args.output))


if __name__ == "__main__":
    main()
