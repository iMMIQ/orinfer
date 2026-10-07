"""Attach a native execution library to an existing prepared model, atomically.

Only the current schema-1 native execution-package format is supported.
Weight shards and existing cubins are unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from tools.model.package import verify_execution, verify_library
from tools.model.publication import atomic_model, clone_cpu_assets, file_hash, link_or_copy, source_path, write_json


def attach(source, output, library, engine, package_name, version):
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(source):
        raise ValueError('Destination must be outside the immutable source model')
    library = library.resolve(strict=True)
    verify_library(library)
    if not package_name or not version:
        raise ValueError('Execution package identity and version are required')
    descriptor = json.loads((source / 'cache/model.json').read_text())
    if descriptor.get('schema_version') != 1:
        raise ValueError('Expected model data schema 1')
    digest = descriptor['execution_package']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Malformed source package digest')
    bundled = source / 'cache/packages' / digest
    if not bundled.exists():
        default = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'orinfer/packages'
        bundled = Path(os.environ.get('ORINFER_EXECUTION_CACHE', str(default))) / digest
    bundled = bundled.resolve(strict=True)
    raw = (bundled / 'package.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Source package digest mismatch')
    package = json.loads(raw)
    verify_execution(package, bundled)
    for kernel in package['kernels']:
        for field in ('module', 'source', 'host_abi'):
            asset = kernel[field]
            if file_hash(source_path(bundled, asset['file'])) != asset['sha256']:
                raise ValueError('Source kernel asset sha256 mismatch')
    with atomic_model(output, engine) as staging:
        staging.mkdir()
        for path in source.iterdir():
            if path.is_file():
                shutil.copyfile(path, staging / path.name)
        shutil.copytree(source / 'cache/weights', staging / 'cache/weights',
                        copy_function=lambda src, dst: link_or_copy(src, dst)
                        if Path(src).suffix == '.safetensors' else shutil.copyfile(src, dst))
        clone_cpu_assets(source, staging)
        target = staging / 'cache/packages/.building'
        shutil.copytree(bundled, target, copy_function=link_or_copy)
        target_library = target / 'lib/model.so'
        target_library.parent.mkdir(exist_ok=True)
        # A previous package may have been hardlinked: never overwrite its inode.
        target_library.unlink(missing_ok=True)
        shutil.copyfile(library, target_library)
        package['execution'] = dict(abi_version=1,
                                    library=dict(file='lib/model.so', sha256=file_hash(library)),
                                    package=package_name, version=version)
        raw = (json.dumps(package, indent=2, ensure_ascii=False) + '\n').encode()
        (target / 'package.json').unlink()
        (target / 'package.json').write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        target.rename(target.with_name(digest))
        descriptor['execution_package'] = digest
        write_json(staging / 'cache/model.json', descriptor)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--engine', type=Path, default=Path('target/release/orinfer'))
    parser.add_argument('--package', required=True)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    print(attach(args.model, args.output, args.library, args.engine, args.package, args.version))


if __name__ == '__main__':
    main()
