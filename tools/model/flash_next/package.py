"""Publish Flash Next data and its registered Rust execution package (schema 1)."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.model.package import model_library, validate_plan
from tools.model.publication import file_hash, link_or_copy, source_path, write_json

ROOT = Path(__file__).resolve().parents[3]


def publish(directory, engine=None):
    directory = directory.resolve(strict=True)
    cache = directory / 'cache'
    build = json.loads((cache / 'build.json').read_text())
    model = copy.deepcopy(build['metadata'])
    config = json.loads((directory / 'config.json').read_text())
    checked = set()
    for kernel in build['kernels']:
        for field in ('module', 'source', 'host_abi'):
            asset = kernel[field]
            key = (asset['file'], asset['sha256'])
            if key not in checked:
                if file_hash(source_path(cache, asset['file'])) != asset['sha256']:
                    raise ValueError('Flash kernel asset identity mismatch')
                checked.add(key)
    library = model_library()
    package = dict(schema_version=1, runtime_abi=1, target='sm_87', architecture='flash_next',
                   compute_policy='int8_quality',
                   config_signature=dict(text=config['text_config'], quantization=config['quantization_config']),
                   prefill_profiles=build['profiles'], kernels=build['kernels'],
                   buffer_contracts=[{k: v for k, v in b.items() if k != 'data'} for b in model['buffers']],
                   toolchain=model['toolchain'],
                   execution=dict(abi_version=1, library=dict(file='lib/model.so', sha256=file_hash(library)),
                                  package='orinfer-models', version='0.1.1'))
    raw = (json.dumps(package, indent=2, ensure_ascii=False) + '\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    destination = cache / 'packages' / digest
    if not destination.exists():
        destination.mkdir(parents=True)
        (destination / 'package.json').write_bytes(raw)
        (destination / 'lib').mkdir()
        shutil.copyfile(library, destination / 'lib/model.so')
        shutil.copytree(cache / 'kernels', destination / 'kernels', copy_function=link_or_copy)
        for name in ('LICENSE', 'COPYING.LESSER', 'COPYING', 'THIRD_PARTY_NOTICES.md'):
            if (ROOT / name).is_file():
                shutil.copyfile(ROOT / name, destination / name)
    descriptor = dict(schema_version=1, architecture='flash_next', compute_policy='int8_quality',
                      execution_package=digest, buffer_scopes=build['scopes'],
                      frontend_assets={n: file_hash(directory / n) for n in
                                       ('tokenizer.json', 'chat_template.jinja', 'generation_config.json')},
                      metadata=model)
    previous = (cache / 'model.json').read_bytes() if (cache / 'model.json').exists() else None
    write_json(cache / 'model.json', descriptor)
    source = dict(model, kernels=build['kernels'], programs=copy.deepcopy(build['programs']))
    source['programs']['prefill'] = source['programs']['prefill_m512']
    source['programs']['head'] = source['programs']['head_m512']
    try:
        validate_plan(directory, source, {k['name']: k['name'] for k in build['kernels']}, engine or ROOT / 'target/release/orinfer')
    except BaseException:
        if previous is None:
            (cache / 'model.json').unlink()
        else:
            (cache / 'model.json').write_bytes(previous)
        raise
    return dict(execution_package=digest, kernel_bindings=len(build['kernels']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--engine', type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.model, args.engine), indent=2))


if __name__ == '__main__':
    main()
