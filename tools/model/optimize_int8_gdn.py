"""Add group-128 INT8 GDN output projections to a quality-priority package.

W4 codes/scales stay unchanged. The optional lossless I8-fragment permutation
uses one packed copy; all readers are rebound together. GDN state remains FP32
and the normalized activation boundary remains FP16. Long prefill retains its
strict temporary-W8 quantization law.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.model.publication import atomic_model, clone_model, commit_package, file_hash, load_model, write_json
from tools.model.upgrade_batching import bind
from tools.operators.abi import parse_host


def repack_weights(source, destination, buffers, layers):
    """Rewrite affected standard safetensors shards; preserve all other tensors."""
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tools.quantization.w4_i8_pack import LAYOUT, pack_array

    index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    changed = {f'L{layer}_Out_P' for layer in layers}
    records = []
    for shard in sorted({index[name] for name in changed}):
        with safe_open(str(source / shard), framework='pt', device='cpu') as reader:
            container_metadata = reader.metadata()
            tensors = {name: reader.get_tensor(name) for name in reader.keys()}
            for name in sorted(changed.intersection(tensors)):
                original = tensors[name].numpy()
                if hashlib.sha256(original.tobytes()).hexdigest() != buffers[name]['data']['sha256']:
                    raise ValueError('Source packed tensor digest mismatch')
                native = pack_array(original, verify=True).view(np.int32)
                digest = hashlib.sha256(native.tobytes()).hexdigest()
                tensors[name] = torch.from_numpy(native)
                buffers[name]['layout'] = LAYOUT
                buffers[name]['data']['sha256'] = digest
                container_metadata['orin.layout.' + name] = LAYOUT
                records.append(dict(name=name, bytes=native.nbytes, packed_roundtrip=True, sha256=digest))
            temporary = destination / (shard + '.new')
            save_file(tensors, str(temporary), metadata=container_metadata)
        temporary.replace(destination / shard)
        print('repacked', shard, flush=True)
        del tensors
    if len(records) != len(changed):
        raise ValueError('Incomplete lossless GDN output permutation')
    return records


def upgrade(model, destination, report, weight_layout):
    from kernels.model.w4a8_decode import w4a8_decode
    from kernels.model.gdn_gated_norm_a8 import gdn_gated_norm_a8
    from kernels.model.w4_i8_to_temporary_w8 import w4_i8_to_temporary_w8
    from tools.operators.common import configure, export_kernel
    import torch

    configure()
    torch.empty(1, device='cuda')
    data, origin, package = load_model(model)
    metadata = data['metadata']
    config = json.loads((model / 'config.json').read_text())
    text = config.get('text_config', config)
    hidden = text['hidden_size']
    width = text['linear_num_value_heads'] * text['linear_value_head_dim']
    if (hidden, width) != (5120, 6144):
        raise ValueError('The grouped gated norm currently supports the 27B geometry')
    buffers = {b['name']: b for b in metadata['buffers']}
    if 'DecodeGdnOutAS' in buffers or not package.get('batch_profiles'):
        raise ValueError('Expected a batch package without INT8 GDN output workspace')
    layers = [i for i, kind in enumerate(text['layer_types']) if kind == 'linear_attention']
    for layer in layers:
        prefix = f'L{layer}_Out'
        if buffers[prefix + '_P']['layout'] != 'u4_warp_n64_k128_mma_f16':
            raise ValueError('Expected the original F16-fragment GDN output weights')
        if buffers[prefix + '_P']['shape'] != [hidden // 64, width // 128, 128, 8]:
            raise ValueError('GDN output projection shape mismatch')
    workspace = dict(name='DecodeGdnOutAS', dtype='f16', shape=[128, width // 128],
                     layout='contiguous', alignment=256, access='read_write', data=None)
    metadata['buffers'].append(workspace)
    data['buffer_scopes'][workspace['name']] = 'workspace'
    cache = destination / 'cache'
    operator = clone_model(model, destination, origin)
    repacked = repack_weights(model / 'cache/weights', destination / 'cache/weights', buffers, layers) if weight_layout == 'i8' else []
    kernels = {k['name']: k for k in package['kernels']}
    exports = {}

    def compile_kernel(name, factory):
        print('compile', name, flush=True)
        out = operator / 'int8-gdn-aot' / name
        export_kernel(factory(), out)
        host = parse_host((out / 'host.txt').read_text())
        if len(host) != 1:
            raise ValueError('Expected one kernel export')
        assets = {field: dict(file=str((out / filename).relative_to(operator)),
                              sha256=file_hash(out / filename)) for field, filename in
                  [('module', 'kernel.cubin'), ('source', 'kernel.cu'), ('host_abi', 'host.txt')]}
        exports[name] = dict(**host[0], **assets)

    compile_kernel('gated-norm-group', lambda: gdn_gated_norm_a8(128, group_activation=True))
    for rows in [1, 2, 4, 8, None]:
        for tile_m in ([16, 32] if rows is None else [16]):
            key = f'out-m{rows or "dynamic"}-tile{tile_m}'
            compile_kernel(key, lambda rows=rows, tile_m=tile_m: w4a8_decode(
                rows, hidden, width, 8, mode='group', TILE_M=tile_m,
                TILE_N=64 if tile_m == 16 else 128, output_dtype='float32',
                activation_group=128, weight_layout=weight_layout))
    if weight_layout == 'i8':
        compile_kernel('out-expand', lambda: w4_i8_to_temporary_w8(hidden, width, BK=512))
    programs = [('decode', 1, 'decode')]
    programs.extend((f'batch_m{rows}', rows, 'batch') for rows in package['batch_profiles'])
    programs.extend((f'prefill_m{p["tokens"]}', p['tokens'], p['kind'])
                    for p in package['prefill_profiles'] if p['kind'] in ('sequence', 'recurrent'))
    replaced = {}
    for program, rows, kind in programs:
        if rows not in (1, 2, 4, 8, 16, 32, 64, 128):
            raise ValueError('Unsupported short-profile row count')
        slot = {'decode': 6, 'batch': 4, 'sequence': 8, 'recurrent': 6}[kind]
        for layer in layers:
            entries = [
                ('gated-norm-group', dict(X='Y', Z='Zout', W=f'L{layer}_GatedWeight',
                                         Q='TemporaryA8', S='DecodeGdnOutAS')),
                (f'out-m{rows if rows <= 8 else "dynamic"}-tile{16 if rows <= 16 else 32}',
                 dict(A='TemporaryA8', PP=f'L{layer}_Out_P', S=f'L{layer}_Out_S',
                      Z=f'L{layer}_Out_Z', WS=f'L{layer}_Out_WS',
                      AS='DecodeGdnOutAS', O='Partial')),
            ]
            for offset, (key, pointers) in enumerate(entries):
                name = f'{program}/layer{layer}/k{slot + offset}'
                if name not in kernels:
                    raise ValueError('Missing registered GDN output slot')
                kernels[name] = bind(exports[key], name, pointers, dict(rows=rows, M=rows))
                replaced[program] = replaced.get(program, 0) + 1
    if weight_layout == 'i8':
        for profile in package['prefill_profiles']:
            if profile['kind'] not in ('chunk_lut4', 'chunk_expanded'):
                continue
            program = f'prefill_m{profile["tokens"]}'
            for layer in layers:
                name = f'{program}/layer{layer}/k13'
                old = kernels[name]
                if not any(a.get('name') == f'L{layer}_Out_P' for a in old['args']):
                    raise ValueError('Registered long-prefill GDN expand slot changed')
                kernels[name] = bind(exports['out-expand'], name,
                    dict(PP=f'L{layer}_Out_P', S=f'L{layer}_Out_S', Z=f'L{layer}_Out_Z',
                         WS=f'L{layer}_Out_WS', W8='TemporaryW8'), dict(rows=profile['tokens']))
                replaced[program] = replaced.get(program, 0) + 1
        # No original F16 reader can remain bound to a permuted tensor.
        rebound = {f'{program}/layer{layer}/k{slot + 1}' for program, _, kind in programs
                   for layer in layers for slot in [{'decode': 6, 'batch': 4, 'sequence': 8, 'recurrent': 6}[kind]]}
        rebound.update(f'prefill_m{p["tokens"]}/layer{layer}/k13' for p in package['prefill_profiles']
                       if p['kind'] in ('chunk_lut4', 'chunk_expanded') for layer in layers)
        names = {f'L{layer}_Out_P' for layer in layers}
        for name, kernel in kernels.items():
            if any(a.get('name') in names for a in kernel['args']) and name not in rebound:
                raise ValueError('Unconverted reader of permuted GDN output weights: ' + name)
    package['kernels'] = list(kernels.values())
    package['buffer_contracts'] = [{k: v for k, v in b.items() if k != 'data'}
                                   for b in metadata['buffers']]
    digest = commit_package(destination, operator, data, package)
    write_json(report / 'upgrade.json', dict(execution_package=digest, replaced=replaced,
               weight_bytes=metadata['weight_bytes'], persistent_weight_bytes_added=0,
               workspace_bytes_added=128 * (width // 128) * 2,
               repacked_weights=repacked,
               weight_representation='lossless I8-fragment W4' if weight_layout == 'i8' else 'unchanged F16-fragment W4; register-only lossless shuffle',
               gdn_output_mode='group-128 activation and original group weight scales'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weight-layout', choices=['i8', 'f16'], default='i8')
    args = parser.parse_args()
    destination = args.model_output.absolute()
    if destination.exists():
        raise FileExistsError(destination)
    args.output.mkdir(parents=True, exist_ok=True)
    with atomic_model(destination, command='validate-model') as staging:
        upgrade(args.model.resolve(strict=True), staging, args.output, args.weight_layout)
    print('INT8 GDN OUTPUT PACKAGE READY', destination, flush=True)


if __name__ == '__main__':
    main()
