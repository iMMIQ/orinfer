"""Shared file identities and atomic offline metadata publication."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash(root, paths):
    """Pin source contents and relative names with stable, deduplicated ordering."""
    entries = {str(Path(p).relative_to(root)): file_hash(p) for p in sorted(set(paths))}
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def atomic_json(path, data, *, compact=False):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(
                data,
                stream,
                indent=None if compact else 2,
                separators=(",", ":") if compact else None,
                ensure_ascii=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
