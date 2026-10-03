"""Package an assembled AOT model as immutable HF sharded safetensors.

This is a container migration: packed weight codes and all initialized buffers
are copied byte-for-byte. It does not quantize, compile, or export a new graph.
Python is used offline; the prepared directory is loaded directly by Rust.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from safetensors import safe_open, serialize_file


DTYPES = {
    'u8': ('U8', 1), 'i8': ('I8', 1), 'f16': ('F16', 2),
    'bf16': ('BF16', 2), 'f32': ('F32', 4), 'u32': ('U32', 4),
    'i32': ('I32', 4), 'u64': ('U64', 8), 'i64': ('I64', 8),
}
RAW_DTYPES = {
    'U8': 'uint8', 'I8': 'int8', 'F16': 'float16', 'BF16': 'bfloat16',
    'F32': 'float32', 'U32': 'uint32', 'I32': 'int32',
    'U64': 'uint64', 'I64': 'int64',
}
CONFIG_FILES = (
    'config.json', 'generation_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'chat_template.jinja', 'preprocessor_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'tokenizer.model',
)
REQUIRED_FILES = ('config.json', 'generation_config.json', 'tokenizer.json',
                  'chat_template.jinja')


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


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def prepare(model, checkpoint, output, shard_bytes=1024**3):
    started = time.monotonic()
    model = model.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f'Output already exists: {output}')
    if shard_bytes <= 0:
        raise ValueError('Shard size must be positive')
    raw = model.read_bytes()
    manifest = json.loads(raw)
    if manifest['schema_version'] != 1 or manifest['target'] != 'sm_87':
        raise ValueError('Expected an assembled schema-1 SM87 AOT model')
    for name in REQUIRED_FILES:
        if not (checkpoint / name).is_file():
            raise ValueError(f'Checkpoint requires {name}')
    config = json.loads((checkpoint / 'config.json').read_text())
    text_config = config.get('text_config', config)
    if text_config.get('vocab_size') != manifest['vocab']:
        raise ValueError('Checkpoint vocabulary differs from the AOT model')
    records = []
    names = set()
    total_size = 0
    for buffer in manifest['buffers']:
        if buffer['name'] in names or not buffer['name']:
            raise ValueError('Empty/duplicate buffer name')
        names.add(buffer['name'])
        if buffer.get('data') is None:
            continue
        dtype, size = DTYPES[buffer['dtype']]
        if not buffer['shape'] or any(type(n) is not int or n <= 0 for n in buffer['shape']):
            raise ValueError(f'Invalid shape: {buffer["name"]}')
        size *= math.prod(buffer['shape'])
        path = source_path(model.parent, buffer['data']['file'])
        if path.stat().st_size != size:
            raise ValueError(f'{buffer["name"]}: byte length mismatch')
        records.append((buffer, path, dtype, size))
        total_size += size
    if not records:
        raise ValueError('No initialized tensors')
    groups = []
    group, group_size = [], 0
    for record in records:
        if group and group_size + record[3] > shard_bytes:
            groups.append(group)
            group, group_size = [], 0
        group.append(record)
        group_size += record[3]
    groups.append(group)
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    if free < total_size + 128 * 1024**2:
        raise ValueError(f'Insufficient space: need {total_size} payload bytes, free {free}')
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        cache = staging / 'cache'
        weights = cache / 'weights'
        kernels = cache / 'kernels'
        weights.mkdir(parents=True)
        kernels.mkdir()
        for name in CONFIG_FILES:
            source = checkpoint / name
            if source.is_file():
                shutil.copyfile(source, staging / name)
        weight_map = {}
        for number, group in enumerate(groups, 1):
            filename = f'model-{number:05d}-of-{len(groups):05d}.safetensors'
            tensors = {}
            metadata = {'orin.cache_format': '1'}
            for buffer, path, dtype, _ in group:
                data = path.read_bytes()
                digest = sha256(data)
                if digest != buffer['data']['sha256']:
                    raise ValueError(f'{buffer["name"]}: source sha256 mismatch')
                name = buffer['name']
                tensors[name] = {'dtype': RAW_DTYPES[dtype], 'shape': buffer['shape'], 'data': data}
                metadata[f'orin.layout.{name}'] = buffer['layout']
                weight_map[name] = filename
                buffer['data'] = {'tensor': name, 'sha256': digest}
            serialize_file(tensors, weights / filename, metadata=metadata)
            # The standard reader validates the written header, shapes and extents.
            with safe_open(weights / filename, framework='np') as reader:
                for buffer, _, dtype, _ in group:
                    view = reader.get_slice(buffer['name'])
                    if view.get_shape() != buffer['shape'] or view.get_dtype() != dtype:
                        raise ValueError('Written safetensors metadata differs')
            del tensors, data
            print(f'prepared shard {number}/{len(groups)}', flush=True)
        write_json(weights / 'model.safetensors.index.json', {
            'metadata': {'total_size': total_size}, 'weight_map': weight_map,
        })
        assets = {}
        for kernel in manifest['kernels']:
            for field in ('module', 'source', 'host_abi'):
                identity = kernel[field]
                key = (identity['file'], identity['sha256'])
                if key not in assets:
                    path = source_path(model.parent, identity['file'])
                    if file_hash(path) != identity['sha256']:
                        raise ValueError(f'{path}: kernel asset sha256 mismatch')
                    filename = identity['sha256'] + path.suffix
                    destination = kernels / filename
                    if not destination.exists():
                        # Copy rather than hardlink: the published cache must stay
                        # immutable even if an intermediate AOT asset is rebuilt.
                        shutil.copyfile(path, destination)
                    assets[key] = {'file': f'kernels/{filename}', 'sha256': identity['sha256']}
                kernel[field] = assets[key].copy()
        manifest['schema_version'] = 2
        manifest['toolchain']['weight_container'] = 'safetensors'
        manifest['toolchain']['source_manifest_sha256'] = sha256(raw)
        write_json(cache / 'manifest.json', manifest)
        # Publish only a complete directory. Existing outputs are never overwritten.
        if output.exists() or output.is_symlink():
            raise ValueError(f'Output appeared during preparation: {output}')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return {'output': str(output), 'tensor_count': len(records), 'shard_count': len(groups),
            'tensor_bytes': total_size, 'weight_bytes': manifest['weight_bytes'],
            'weight_parameters': manifest['weight_parameters'],
            'effective_weight_bits': 8 * manifest['weight_bytes'] / manifest['weight_parameters'],
            'source_manifest_sha256': sha256(raw), 'prepare_s': time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Assembled intermediate AOT model.json')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Matching HF config/tokenizer directory')
    parser.add_argument('--output', type=Path, required=True, help='New prepared model directory')
    parser.add_argument('--shard-mib', type=int, default=1024,
                        help='Target payload size; an individual tensor is never split')
    args = parser.parse_args()
    result = prepare(args.model, args.checkpoint, args.output, args.shard_mib * 1024**2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
