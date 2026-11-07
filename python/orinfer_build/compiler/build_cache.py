"""Atomic, verified offline construction objects; no model weights are cached."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from orinfer_build.common.files import file_hash
from orinfer_build.weights.safetensors_source import relative_name


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class BuildCache:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "locks").mkdir(parents=True, exist_ok=True)
        (self.root / "objects").mkdir(exist_ok=True)

    @contextmanager
    def locked(self, key):
        with (self.root / "locks" / (key + ".lock")).open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    @staticmethod
    def valid(path, contract):
        try:
            entry = json.loads((path / "entry.json").read_text())
            if (
                not isinstance(entry, dict)
                or entry.get("contract") != contract
                or not isinstance(entry.get("files"), dict)
                or not entry["files"]
            ):
                return False
            for name, sha in entry["files"].items():
                relative_name(name)
                file = (path / name).resolve(strict=True)
                if not file.is_relative_to(path.resolve()) or file_hash(file) != sha:
                    return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def get(self, contract, build):
        key = identity(contract)
        target = self.root / "objects" / key
        with self.locked(key):
            if target.exists() and self.valid(target, contract):
                return target, True
            if target.exists():
                shutil.rmtree(target)
            with tempfile.TemporaryDirectory(prefix=".building-", dir=target.parent) as temporary:
                staging = Path(temporary) / "object"
                staging.mkdir()
                build(staging)
                files = {
                    str(p.relative_to(staging)): file_hash(p)
                    for p in staging.rglob("*")
                    if p.is_file()
                }
                if not files:
                    raise ValueError("Empty construction cache object")
                (staging / "entry.json").write_text(
                    json.dumps(dict(contract=contract, files=files), sort_keys=True)
                )
                if not self.valid(staging, contract):
                    raise ValueError("Construction object failed publication validation")
                staging.rename(target)
        return target, False
