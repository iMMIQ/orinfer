"""Audit release ages before dependency installation; checking is offline and stdlib-only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from functools import cache
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import tomllib
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
RECORD = Path("configs/dependency-releases.json")
LOCKS = ("Cargo.lock", "uv.lock", "tools/environments/cuda/uv.lock")
MINIMUM_AGE = timedelta(days=7)


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Release timestamps must include a timezone")
    return parsed.astimezone(UTC)


def inputs(root):
    paths = [
        *LOCKS,
        "Cargo.toml",
        "rust-toolchain.toml",
        "pyproject.toml",
        "tools/environments/cuda/pyproject.toml",
        ".github/workflows/check.yml",
        "tools/environments/ascend/run.sh",
        "tools/environments/ascend/Dockerfile",
        "tools/build/jetson_wheels.json",
    ]
    paths += [str(p.relative_to(root)) for p in (root / "crates").glob("*/Cargo.toml")]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sorted(paths)}


def verify(record, fingerprints, now):
    if record.get("schema") != 1 or record.get("minimum_age_days") != 7:
        raise ValueError("Unsupported dependency release record")
    if record["files"] != fingerprints:
        raise ValueError("Dependency inputs changed; refresh release metadata before installing")
    cutoff = now - MINIMUM_AGE
    for name, release in record["releases"].items():
        if timestamp(release["published"]) > cutoff:
            raise ValueError(f"Dependency has not aged 7 days: {name} ({release['published']})")
    if not record["releases"]:
        raise ValueError("Empty dependency release record")


def check(root=ROOT):
    record = json.loads((root / RECORD).read_text())
    verify(record, inputs(root), datetime.now(UTC))
    print(f"Dependency age: {len(record['releases'])} releases verified; minimum 7 days (offline)")


class Registry:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    @cache
    def raw(self, url):
        path = self.directory / hashlib.sha256(url.encode()).hexdigest()
        if path.exists() and time.time() - path.stat().st_mtime < 86400:
            return path.read_bytes()
        for attempt in range(4):
            try:
                request = Request(
                    url,
                    headers={"User-Agent": "orinfer-release-age-audit (github.com/iMMIQ/orinfer)"},
                )
                with urlopen(request, timeout=30) as response:
                    data = response.read(16 * 1024**2 + 1)
                if len(data) > 16 * 1024**2:
                    raise ValueError("Unbounded dependency metadata")
                break
            except OSError as error:
                if attempt == 3:
                    raise ValueError(f"Cannot read release metadata {url}: {error}") from error
                time.sleep(2**attempt)
        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(path)
        return data

    def json(self, url):
        return json.loads(self.raw(url))


def crate_url(name):
    key = (
        name
        if len(name) == 1
        else f"2/{name}"
        if len(name) == 2
        else f"3/{name[0]}/{name}"
        if len(name) == 3
        else f"{name[:2]}/{name[2:4]}/{name}"
    )
    return "https://index.crates.io/" + key.lower()


def pypi_url(name, version):
    return f"https://pypi.org/pypi/{name}/{version}/json"


def github_asset(registry, url, checksum):
    parts = urlsplit(url)
    if parts.hostname != "github.com":
        raise ValueError(f"Unreviewed direct dependency: {url}")
    owner, repo, releases, download, tag, filename = unquote(parts.path).strip("/").split("/")
    if (releases, download) != ("releases", "download"):
        raise ValueError("Direct dependencies must identify a release asset")
    metadata = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    release = registry.json(metadata)
    if release["draft"] or release["prerelease"]:
        raise ValueError("Direct dependency is not a stable release")
    asset = next(v for v in release["assets"] if v["name"] == filename)
    if asset.get("digest") and asset["digest"] != "sha256:" + checksum:
        raise ValueError(f"Release asset checksum changed: {filename}")
    return dict(
        published=max(release["published_at"], asset["created_at"], key=timestamp), url=metadata
    )


def refresh(root=ROOT):
    registry = Registry(root / "artifacts/cache/dependency-metadata")
    cargo = tomllib.loads((root / "Cargo.lock").read_text())["package"]
    python = {lock: tomllib.loads((root / lock).read_text())["package"] for lock in LOCKS[1:]}
    project = tomllib.loads((root / "pyproject.toml").read_text())
    for path in [root / "pyproject.toml", root / "tools/environments/cuda/pyproject.toml"]:
        if tomllib.loads(path.read_text())["tool"]["uv"].get("exclude-newer") != "7 days":
            raise ValueError("Both uv environments must enforce exclude-newer = '7 days'")
    build = []
    for requirement in project["build-system"]["requires"]:
        match = re.fullmatch(r"([\w-]+)==([\w.]+)", requirement)
        if not match:
            raise ValueError("Build dependencies must pin an exact stable version")
        build.append(match.groups())
    urls = {crate_url(p["name"]) for p in cargo if p.get("source")}
    urls.update(
        pypi_url(p["name"], p["version"])
        for packages in python.values()
        for p in packages
        if "registry" in p["source"]
    )
    urls.update(pypi_url(*item) for item in build)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, _ in enumerate(pool.map(registry.raw, sorted(urls)), 1):
            if index % 40 == 0:
                print(f"Read {index}/{len(urls)} registry metadata records", flush=True)
    releases = {}
    for package in cargo:
        if not package.get("source"):
            continue
        if package["source"] != "registry+https://github.com/rust-lang/crates.io-index":
            raise ValueError("Unreviewed Cargo registry or git dependency")
        url = crate_url(package["name"])
        rows = [json.loads(line) for line in registry.raw(url).splitlines()]
        release = next(v for v in rows if v["vers"] == package["version"])
        if release["cksum"] != package["checksum"] or release["yanked"]:
            raise ValueError("Cargo release checksum changed or release was yanked")
        releases[f"cargo/{package['name']}@{package['version']}"] = dict(
            published=release["pubtime"], url=url
        )
    for lock, packages in python.items():
        for package in packages:
            source = package["source"]
            key = f"{lock}/{package['name']}@{package['version']}"
            if "editable" in source or "virtual" in source:
                continue
            if "url" in source:
                artifact = package.get("sdist") or package["wheels"][0]
                releases[key] = github_asset(
                    registry, source["url"], artifact["hash"].removeprefix("sha256:")
                )
                continue
            if source.get("registry") != "https://pypi.org/simple":
                raise ValueError("Unreviewed Python package source")
            url = pypi_url(package["name"], package["version"])
            by_hash = {v["digests"]["sha256"]: v for v in registry.json(url)["urls"]}
            files = [
                *package.get("wheels", []),
                *([package["sdist"]] if "sdist" in package else []),
            ]
            if not files:
                raise ValueError("Python release has no locked artifacts")
            selected = [by_hash[v["hash"].removeprefix("sha256:")] for v in files]
            if any(v["yanked"] for v in selected):
                raise ValueError("Locked Python artifact was yanked")
            releases[key] = dict(
                published=max((v["upload_time_iso_8601"] for v in selected), key=timestamp), url=url
            )
    for name, version in build:
        url = pypi_url(name, version)
        files = registry.json(url)["urls"]
        if not files or any(v["yanked"] for v in files):
            raise ValueError("Build dependency has no usable release")
        releases[f"build/{name}@{version}"] = dict(
            published=max((v["upload_time_iso_8601"] for v in files), key=timestamp), url=url
        )
    channel = tomllib.loads((root / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
    rust_url = f"https://static.rust-lang.org/dist/channel-rust-{channel}.toml"
    manifest = tomllib.loads(registry.raw(rust_url).decode())
    if manifest["pkg"]["rust"]["version"].split()[0] != channel:
        raise ValueError("Rust distribution identity mismatch")
    releases[f"rust/{channel}"] = dict(published=manifest["date"] + "T23:59:59Z", url=rust_url)
    workflow = (root / ".github/workflows/check.yml").read_text()
    actions = re.findall(r"uses: ([\w.-]+/[\w.-]+)@([a-f0-9]{40}) # (v[\w.]+)", workflow)
    if len(actions) != len(re.findall(r"uses:", workflow)):
        raise ValueError("CI actions must pin a release commit and record its version")
    for repository, commit, tag in actions:
        url = f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
        release = registry.json(url)
        ref = registry.json(f"https://api.github.com/repos/{repository}/git/ref/tags/{tag}")
        while ref["object"]["type"] == "tag":
            ref = registry.json(ref["object"]["url"])
        if ref["object"]["sha"] != commit or release["draft"] or release["prerelease"]:
            raise ValueError("CI action commit does not identify its stable release")
        releases[f"action/{repository}@{tag}"] = dict(published=release["published_at"], url=url)
    uv_version = re.search(r"(?m)^\s+version: '([\d.]+)'", workflow)[1]
    uv_url = f"https://api.github.com/repos/astral-sh/uv/releases/tags/{uv_version}"
    uv = registry.json(uv_url)
    if uv["draft"] or uv["prerelease"]:
        raise ValueError("CI uv must identify a stable release")
    releases[f"uv/{uv_version}"] = dict(published=uv["published_at"], url=uv_url)
    runner = (root / "tools/environments/ascend/run.sh").read_text()
    wheel = re.search(r"wheel_url='([^']+)'", runner)[1]
    docker = (root / "tools/environments/ascend/Dockerfile").read_text()
    digest = re.search(r'hexdigest\(\)=="([a-f0-9]{64})"', docker)[1]
    releases["ascend/tilelang-wheel"] = github_asset(registry, wheel, digest)
    record = dict(
        schema=1, minimum_age_days=7, files=inputs(root), releases=dict(sorted(releases.items()))
    )
    verify(record, inputs(root), datetime.now(UTC))
    (root / RECORD).write_text(json.dumps(record, indent=2) + "\n")
    check(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "refresh"))
    args = parser.parse_args()
    try:
        (check if args.command == "check" else refresh)()
    except (ValueError, KeyError, StopIteration, OSError) as error:
        parser.exit(1, f"Dependency release audit failed: {error}\n")


if __name__ == "__main__":
    main()
