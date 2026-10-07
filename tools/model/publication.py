"""Shared immutable model cloning and validated, atomic publication."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def source_path(base, file):
    path = Path(file)
    if not file or path.is_absolute() or any(p in ('.', '..') for p in path.parts):
        raise ValueError(f'Unsafe artifact path: {file}')
    # Check lexical components too: pathlib normalizes away "." components.
    if any(p in ('', '.', '..') for p in file.split('/')):
        raise ValueError(f'Unsafe artifact path: {file}')
    resolved = (base / path).resolve(strict=True)
    if not resolved.is_relative_to(base):
        raise ValueError(f'Artifact leaves its directory: {file}')
    return resolved

def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def clone_cpu_assets(source, destination):
    """CPU input metadata and immutable tables travel with prepared model data."""
    assets = source / 'cache/cpu'
    if assets.is_dir():
        shutil.copytree(assets, destination / 'cache/cpu',
                        copy_function=lambda src, dst: link_or_copy(src, dst)
                        if Path(src).suffix == '.safetensors' else shutil.copyfile(src, dst))


def load_model(model):
    model = model.resolve(strict=True)
    data = json.loads((model / 'cache/model.json').read_text())
    digest = data['execution_package']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid operator package digest')
    root = Path(os.environ.get('ORINFER_EXECUTION_CACHE', str(Path(os.environ.get(
        'XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'orinfer/packages')))
    origin = root / digest
    if not origin.exists():
        origin = model / 'cache/packages' / digest
    raw = (origin / 'package.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Source operator package digest mismatch')
    return data, origin, json.loads(raw)


def clone_model(source, destination, operator):
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError('Destination must be outside the immutable source model')
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copyfile(path, destination / path.name)
    shutil.copytree(source / 'cache/weights', destination / 'cache/weights',
                    copy_function=lambda src, dst: link_or_copy(src, dst)
                    if Path(src).suffix == '.safetensors' else shutil.copyfile(src, dst))
    clone_cpu_assets(source, destination)
    target = destination / 'cache/packages/.building'
    shutil.copytree(operator, target, copy_function=link_or_copy)
    return target


def commit_package(model, operator, data, package):
    raw = (json.dumps(package, ensure_ascii=False, indent=2) + '\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    # Metadata can be hardlinked to the source. Always replace its inode.
    with tempfile.NamedTemporaryFile(dir=operator, delete=False) as stream:
        temporary = Path(stream.name)
        os.fchmod(stream.fileno(), (operator / 'package.json').stat().st_mode & 0o777)
        stream.write(raw)
    try:
        temporary.replace(operator / 'package.json')
    finally:
        temporary.unlink(missing_ok=True)
    operator.rename(operator.with_name(digest))
    data['execution_package'] = digest
    write_json(model / 'cache/model.json', data)
    return digest


@contextmanager
def staged_directory(destination):
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Destination already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{destination.name}-', dir=destination.parent) as directory:
        staging = Path(directory) / 'model'
        yield staging
        if destination.exists() or destination.is_symlink():
            raise ValueError('Destination appeared during publication')
        staging.rename(destination)


@contextmanager
def atomic_model(destination, engine=None, command='plan-model'):
    with staged_directory(destination) as staging:
        yield staging
        cli = engine if engine is not None else ROOT / 'target/release/orinfer'
        subprocess.run([str(cli.resolve(strict=True)), command, str(staging)],
                       check=True, stdout=subprocess.DEVNULL)
