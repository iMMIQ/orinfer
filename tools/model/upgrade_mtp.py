"""Offline update of prepared MTP bindings; immutable weights are unchanged.

Reuses the model's existing multimodal embedding and MRoPE implementations,
binding their exported dynamic-row ABI to the draft buffers. Publishes a new
content-addressed operator package and model directory atomically.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.model.publication import atomic_model, clone_model, commit_package, load_model, source_path
from tools.operators.abi import evaluate, parse_host


def bind_rows(kernel, package_dir, names, rows, slot):
    result = copy.deepcopy(kernel)
    exports = parse_host(source_path(package_dir, kernel['host_abi']['file']).read_text())
    if len(exports) != 1 or exports[0]['symbol'] != kernel['symbol']:
        raise ValueError('Expected one exported dynamic-row kernel')
    export = exports[0]
    if len(export['ordered_arguments']) != len(result['args']):
        raise ValueError('Exported argument count differs')
    for arg, abi in zip(result['args'], export['ordered_arguments']):
        if arg['kind'] == 'buffer':
            if abi['ctype'] != 'ctypes.c_void_p':
                raise ValueError('Unexpected buffer ABI')
            arg['name'] = names.get(arg['name'], arg['name'])
        elif arg['kind'] == 'i32' and abi == {'value': 'rows', 'ctype': 'ctypes.c_int32'}:
            arg['value'] = rows
        else:
            raise ValueError('Unexpected scalar ABI')
    dims = export['launch_expressions']
    result['grid'] = [evaluate(dims['gridDim' + axis], {'rows': rows}) for axis in 'XYZ']
    result['block'] = [evaluate(dims['blockDim' + axis], {'rows': rows}) for axis in 'XYZ']
    result['shared_memory_bytes'] = evaluate(dims['sharedMemBytes'], {'rows': rows})
    result['name'] = slot
    return result


def upgrade(model, output, engine):
    model = model.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Output already exists')
    data, package_dir, package = load_model(model)
    metadata = data['metadata']
    spec = metadata['mtp']
    if not spec or not metadata['vision']:
        raise ValueError('Expected a prepared multimodal MTP model')
    kernels = {k['name']: k for k in package['kernels']}
    embedding = kernels['decode/begin/k1']
    prepare = next(k for k in package['kernels'] if k['name'].startswith('prefill_m2/layer')
                   and any(a.get('name') == metadata['vision']['mrope_positions'] for a in k['args']))
    layer_kv = next(a['name'].split('_')[0] for a in prepare['args']
                    if a.get('name', '').endswith('_KPages'))
    embedding_names = {metadata['input']: spec['input'], metadata['position']: spec['position'],
                       metadata['vision']['feature_index']: 'MtpFeatureIndex', 'Hidden': 'MtpEmbedding'}
    prepare_names = dict(FullGate='MtpGate', Positions='MtpPositions', FullQ='MtpFullQ',
                         PrepareStatus='MtpPrepareStatus', FullX='MtpFullX')
    prepare_names.update({layer_kv + '_' + key: 'Mtp' + key
                          for key in ('KPages', 'VPages', 'KWeight', 'QWeight')})
    for plan in spec['warm_plans']:
        for slot, source, mapping in ((2, embedding, embedding_names), (8, prepare, prepare_names)):
            name = f'{plan["program"]}/body/k{slot}'
            if name not in kernels:
                raise ValueError('Missing MTP operator slot')
            kernels[name] = bind_rows(source, package_dir, mapping, plan['tokens'], name)
    package['kernels'] = [kernels[k['name']] for k in package['kernels']]
    index = dict(name='MtpFeatureIndex', dtype='i32', shape=[metadata['max_context']],
                 layout='contiguous', alignment=256, access='read_write', data=None)
    if any(b['name'] == index['name'] for b in metadata['buffers']):
        raise ValueError('Model already has shifted MTP inputs')
    metadata['buffers'].append(index)
    package['buffer_contracts'].append({k: v for k, v in index.items() if k != 'data'})
    data['buffer_scopes'][index['name']] = 'sequence'
    spec.update(draft_logits='MtpLogits', verification_logits='SequenceLogits', feature_index=index['name'])
    with atomic_model(output, engine, command='validate-model') as staging:
        operator = clone_model(model, staging, package_dir)
        new_digest = commit_package(staging, operator, data, package)
    return dict(model=str(output), operator_package=new_digest, weights_unchanged=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--engine', type=Path, default=Path('target/release/orinfer'))
    args = parser.parse_args()
    print(json.dumps(upgrade(args.model, args.output, args.engine), indent=2))
