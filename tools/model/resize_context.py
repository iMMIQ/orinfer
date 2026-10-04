"""Resize a prepared Qwen3_5 SM87 cache without touching learned weights.

Run through tools/operators/run.sh. Rebuild capacity-dependent TileLang kernels,
rebind the exported host ABI, and publish a new immutable model directory.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from tools.operators.abi import evaluate, parse_host


def recipe(binding):
    names = {a.get('name') for a in binding['args'] if a['kind'] == 'buffer'}
    rows = int(re.search(r'_m(\d+)', binding['name'])[1]) if '_m' in binding['name'] else 1
    if names & {'FeatureIndex', 'MtpFeatureIndex'}:
        return ('embedding',)
    if 'MRopePositions' in names:
        return ('prepare',)
    if 'MtpTargetHidden' in names:
        return ('gather', rows) if 'MtpCondition' in names else ('capture',)
    if 'Kcontig' in names:
        return ('gather_kv',) if 'Pages' in names else ('prefill', rows)
    if 'Pages' in names:
        if names & {'AttO', 'MtpAttO'}:
            return ('attention', None if binding['name'].startswith('decode/') else rows)
        raise ValueError(f'Unknown context-dependent binding: {binding["name"]}')
    return None


def rebind(binding, old_host, new_host, dimensions):
    """Bind by actual exported expressions; capacity changes can reorder pointers."""
    old = parse_host(old_host)
    new = parse_host(new_host)
    if len(old) != 1 or len(new) != 1:
        raise ValueError('Expected exactly one exported launch')
    if len(old[0]['ordered_arguments']) != len(binding['args']):
        raise ValueError('Stored binding does not match exported ABI')
    arguments = {formal['value']: actual for formal, actual in
                 zip(old[0]['ordered_arguments'], binding['args'])}
    result = copy.deepcopy(binding)
    result['args'] = []
    for formal in new[0]['ordered_arguments']:
        expression = formal['value']
        if expression.endswith('.data_ptr()'):
            actual = arguments[expression]
            if actual['kind'] != 'buffer' or formal['ctype'] != 'ctypes.c_void_p':
                raise ValueError('Pointer ABI mismatch')
        else:
            if formal['ctype'] not in ('ctypes.c_int', 'ctypes.c_int32'):
                raise ValueError('Unsupported scalar ABI')
            actual = dict(kind='i32', value=int(evaluate(expression, dimensions)))
        result['args'].append(actual)
    launch = new[0]['launch_expressions']
    result['symbol'] = new[0]['symbol']
    result['grid'] = [int(evaluate(launch['gridDim' + axis], dimensions)) for axis in 'XYZ']
    result['block'] = [int(evaluate(launch['blockDim' + axis], dimensions)) for axis in 'XYZ']
    result['shared_memory_bytes'] = int(evaluate(launch['sharedMemBytes'], dimensions))
    return result


def trim_prefill(wrapper, package, maximum):
    """Reuse existing smaller profiles and resize their caller-owned scratch."""
    meta = wrapper['metadata']
    previous = meta['chunk_tokens']
    sizes = {p['tokens'] for p in package['prefill_profiles']}
    if maximum not in sizes or maximum > previous or maximum % 64:
        raise ValueError('Maximum prefill must be an existing, 64-token aligned profile')
    removed = sizes - {n for n in sizes if n <= maximum}
    prefixes = tuple(f'{kind}_m{n}/' for n in removed for kind in
                     ('prefill', 'head', 'mtp_capture'))
    package['kernels'] = [k for k in package['kernels'] if not k['name'].startswith(prefixes)]
    package['prefill_profiles'] = [p for p in package['prefill_profiles'] if p['tokens'] <= maximum]
    meta['prefill_plans'] = [p for p in meta['prefill_plans'] if p['chunk_tokens'] <= maximum]
    meta['mtp']['capture_plans'] = [p for p in meta['mtp']['capture_plans'] if p['tokens'] <= maximum]
    chunk_buffers = {'Gcum', 'Bpad', 'System', 'QK', 'Transform', 'W', 'U', 'Rchunk', 'Senter'}
    vision_buffers = {'VPixels', 'VGrid', 'VLength', 'VHidden', 'VNorm', 'VQKV', 'VQ', 'VK', 'VV',
                      'VAttention', 'VProjection', 'VFFN', 'VMerged', 'VOutput', 'Features'}
    for b in meta['buffers']:
        if b['data'] or b['name'] in vision_buffers or b['name'].startswith('Mtp'):
            continue
        if b['name'] in chunk_buffers:
            if b['shape'][1] != previous // 64:
                raise ValueError('Unexpected GDN scratch chunk dimension')
            b['shape'][1] = maximum // 64
        elif wrapper['buffer_scopes'][b['name']] == 'workspace' or b['name'] == meta['input']:
            if b['name'] == 'TemporaryA8':
                if b['shape'][0] % previous:
                    raise ValueError('Unexpected activation workspace')
                b['shape'][0] = b['shape'][0] // previous * maximum
            else:
                b['shape'] = [maximum if n == previous else n for n in b['shape']]
    meta['chunk_tokens'] = maximum


def resize(source, destination, context, output, engine, max_prefill=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tools.operators.common import configure, export_kernel, tensor_sha, write_json
    from kernels.vision.bridge import embedding_features, full_prepare_mrope
    from kernels.model.attention_decode_staged import paged_attention_partials_gqa_staged
    from kernels.model.attention_prefill_staged import attention_prefill_staged
    from kernels.model.speculation import capture_target_hidden, gather_target_hidden

    configure()
    source = source.resolve()
    engine = engine.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Destination must not exist')
    wrapper = json.loads((source / 'cache/model.json').read_text())
    meta = wrapper['metadata']
    if meta.get('kv_cache'):
        raise ValueError('Resize context before optimizing KV; rebuild from the original prepared model')
    config = json.loads((source / 'config.json').read_text())
    if wrapper['architecture'] != 'qwen3_5' or not meta.get('vision') or not meta.get('mtp'):
        raise ValueError('Expected prepared multimodal Qwen3_5 with MTP')
    if context % 128 or not meta['chunk_tokens'] <= context <= config['text_config']['max_position_embeddings']:
        raise ValueError('Context must be page-aligned and within checkpoint capacity')
    old_context = meta['max_context']
    old_package = source / 'cache/operators' / wrapper['operator_package']
    package = json.loads((old_package / 'package.json').read_text())
    if hashlib.sha256((old_package / 'package.json').read_bytes()).hexdigest() != wrapper['operator_package']:
        raise ValueError('Source operator package digest differs')
    if max_prefill is not None:
        trim_prefill(wrapper, package, max_prefill)
    ring = meta['chunk_tokens']
    hidden = next(b['shape'][1] for b in meta['buffers'] if b['name'] == 'Hidden')
    text = config['text_config']
    rope = text['rope_parameters']
    if hidden != 5120 or text['intermediate_size'] != 17408 or text['num_attention_heads'] != 24 \
            or text['num_key_value_heads'] != 4 or text['head_dim'] != 256 \
            or rope['rope_theta'] != 10000000 or rope['partial_rotary_factor'] != 0.25 \
            or rope['rope_type'] != 'default' or rope.get('mrope_interleaved') is not True:
        raise ValueError('Unsupported geometry or RoPE; this resizer targets the 27B operator family')
    pages = context // 128
    staging = destination.with_name(destination.name + '.staging')
    if staging.exists() or staging.is_symlink():
        raise ValueError('Staging directory already exists')
    staging.mkdir(parents=True)
    for item in source.iterdir():
        if item.name != 'cache':
            shutil.copy2(item, staging / item.name)
    cache = staging / 'cache'
    weights = cache / 'weights'
    weights.mkdir(parents=True)
    for item in (source / 'cache/weights').glob('*.safetensors'):
        os.link(item, weights / item.name)
    for b in meta['buffers']:
        if b['name'].endswith(('KPages', 'VPages')):
            b['shape'][0] = pages
        elif b['name'] == 'Pages':
            b['shape'][1] = pages
        elif b['name'] == 'MtpTargetHidden':
            b['shape'][0] = ring
        elif b['name'] in {'Rotary', 'FeatureIndex', 'MtpFeatureIndex', 'MRopePositions', 'Kcontig', 'Vcontig'}:
            b['shape'][0] = context
    freq = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = torch.arange(context, dtype=torch.float32)[:, None] * freq[None, :]
    tables = {'Rotary': torch.cat((angles.cos(), angles.sin()), dim=-1).half(),
              'Pages': torch.arange(pages, dtype=torch.int32).reshape(1, pages)}
    index = json.loads((source / 'cache/weights/model.safetensors.index.json').read_text())
    # Preserve original short-position tables bit for bit, even if an older
    # builder generated them on a different Torch backend.
    with safe_open(source / 'cache/weights' / index['weight_map']['Rotary'], framework='pt') as shard:
        old = shard.get_tensor('Rotary')
        count = min(old.shape[0], context)
        tables['Rotary'][:count].copy_(old[:count])
    layouts = {f'orin.layout.{b["name"]}': b['layout'] for b in meta['buffers'] if b['name'] in tables}
    table_file = f'context-{context}.safetensors'
    if (weights / table_file).exists():
        (weights / table_file).unlink()
    save_file(tables, weights / table_file, metadata=layouts)
    for b in meta['buffers']:
        if b['name'] in tables:
            index['weight_map'][b['name']] = table_file
            b['data'] = dict(tensor=b['name'], sha256=tensor_sha(tables[b['name']]))
    index['metadata']['total_size'] = sum(math.prod(b['shape']) *
        {'i8': 1, 'u8': 1, 'f16': 2, 'f32': 4, 'i32': 4}[b['dtype']]
        for b in meta['buffers'] if b.get('data'))
    write_json(weights / 'model.safetensors.index.json', index)
    package_out = cache / 'operators/staging'
    package_out.mkdir(parents=True)
    shutil.copytree(old_package / 'kernels', package_out / 'kernels', copy_function=os.link)
    for name in ('COPYING', 'LICENSE'):
        if (old_package / name).exists():
            shutil.copy2(old_package / name, package_out / name)
    compiled = {}
    sections = tuple(config['text_config']['rope_parameters']['mrope_section'])
    for i, binding in enumerate(package['kernels']):
        key = recipe(binding)
        if key is None:
            continue
        rows = int(re.search(r'_m(\d+)', binding['name'])[1]) if '_m' in binding['name'] else 1
        dimensions = dict(rows=rows, batch=1, tokens=context, pages=pages, table_width=pages)
        old_host = (old_package / binding['host_abi']['file']).read_text()
        if key == ('gather_kv',):
            package['kernels'][i] = rebind(binding, old_host, old_host, dimensions)
            continue
        if context == old_context and key[0] not in {'capture', 'gather'}:
            # Their capacity and arithmetic did not change; only scratch
            # allocations shrink to the remaining existing profile contracts.
            continue
        if key not in compiled:
            print('compile', key, flush=True)
            kind = key[0]
            if kind == 'embedding':
                kernel = embedding_features(meta['vocab'], hidden, context, meta['vision']['max_features'])
            elif kind == 'prepare':
                kernel = full_prepare_mrope(pages, context, sections, max_position=context)
            elif kind == 'capture':
                kernel = capture_target_hidden(hidden, ring, ring=True)
            elif kind == 'gather':
                kernel = gather_target_hidden(key[1], hidden, ring, ring=True)
            elif kind == 'attention':
                kernel = paged_attention_partials_gqa_staged(pages, pages, queries=key[1])
            elif kind == 'prefill':
                kernel = attention_prefill_staged(1, key[1], context, kv_layout='token_major',
                                                 block_m=32 if key[1] == 512 else 64)
            else:
                raise ValueError(key)
            exported = output / ('-'.join(map(str, key)))
            export_kernel(kernel, exported)
            assets = {}
            for name, file in [('source', 'kernel.cu'), ('host_abi', 'host.txt'), ('module', 'kernel.cubin')]:
                data = (exported / file).read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                relative = 'kernels/' + digest + Path(file).suffix
                if not (package_out / relative).exists():
                    (package_out / relative).write_bytes(data)
                assets[name] = dict(file=relative, sha256=digest)
            compiled[key] = (assets, (exported / 'host.txt').read_text())
        assets, host = compiled[key]
        updated = rebind(binding, old_host, host, dimensions)
        updated.update(assets)
        package['kernels'][i] = updated
    meta['max_context'] = context
    meta['mtp']['hidden_ring'] = 'MtpTargetHidden'
    package['buffer_contracts'] = [{k: v for k, v in b.items() if k != 'data'} for b in meta['buffers']]
    write_json(package_out / 'package.json', package)
    digest = hashlib.sha256((package_out / 'package.json').read_bytes()).hexdigest()
    package_out.rename(package_out.with_name(digest))
    wrapper['operator_package'] = digest
    write_json(cache / 'model.json', wrapper)
    subprocess.run([str(engine), 'validate-model', str(staging)], check=True)
    staging.rename(destination)
    write_json(output / 'resize.json', dict(source=str(source), model=str(destination),
        previous_context=old_context, max_context=context, hidden_ring_tokens=ring,
        compiled_variants=len(compiled), max_prefill_tokens=meta['chunk_tokens'], operator_package=digest))
    print('published', destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--max-context', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', type=Path, default=Path('target/release/orin-llm'))
    parser.add_argument('--max-prefill-tokens', type=int)
    args = parser.parse_args()
    resize(args.model, args.destination, args.max_context, args.output, args.engine, args.max_prefill_tokens)


if __name__ == '__main__':
    main()
