"""Offline Flash decode batching: shared projections/MoE, private causal mixers.

Reuses immutable weights and single-sequence state kernels. Compilation consumes
shape-only meta tensors, so no second model or expanded expert bank is loaded.
"""
import argparse
import ast
import copy
import json
import re
from pathlib import Path

from tools.model.publication import atomic_model, clone_model, commit_package, load_model, source_path, file_hash, write_json

PROFILES = (2, 4, 8, 16, 32, 64, 128)
PRIVATE = ('gdn-conv-', 'gdn-sequence-', 'ple-conv-', 'ple-history-',
           'qsa-prepare-', 'qsa-kv-store-', 'index-query-', 'index-compress-',
           'index-pending-', 'index-scores-', 'index-hist-', 'index-choose-',
           'index-counts-', 'index-offsets-', 'index-scatter-', 'qsa-sparse-', 'qsa-merge-')
INTERFACES = {'gdn-conv': (0,), 'gdn-sequence': (3, 4, 7),
              'ple-conv': (0, 1, 4), 'ple-history': (0,),
              'qsa-prepare': (0, 1, 2), 'index-query': (0,),
              'index-compress': (0,), 'index-pending': (0,), 'qsa-merge': (4,)}


def kind(label):
    label = label.removeprefix('vision-')
    return next((prefix[:-1] for prefix in PRIVATE if label.startswith(prefix)), None)


def upgrade(model, destination, output):
    import torch
    from tools.model.flash_next.native import Model
    from tools.model.flash_next.prepare import Publisher, DTYPES
    from tools.operators.common import configure
    from tools.operators.abi import parse_host
    configure()
    data, origin, package = load_model(model)
    if data['architecture'] != 'flash_next' or data['metadata'].get('batch_profiles'):
        raise ValueError('Expected an unbatched native Flash model')
    operator = clone_model(model, destination, origin)
    metadata = data['metadata']
    buffers = {b['name']: b for b in metadata['buffers']}
    original = {k['name']: k for k in package['kernels']}
    exports = {}
    def api(kernel):
        path = source_path(operator.resolve(), kernel['host_abi']['file'])
        if path not in exports:
            node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'call')
            launch, = parse_host(path.read_text())
            exports[path] = ([a.arg for a in node.args.args if a.arg not in ('kernels', 'stream')], launch)
        return exports[path]
    def variable(kernel, index):
        names, launch = api(kernel)
        name = names[index] + '.data_ptr()'
        values = [arg for arg, abi in zip(kernel['args'], launch['ordered_arguments']) if abi['value'] == name]
        if len(values) != 1 or values[0]['kind'] != 'buffer':
            raise ValueError('Expected a complete interface buffer')
        return values[0]['name']
    by_layer = {}
    for k in original.values():
        if not k['name'].startswith('decode/layer'):
            continue
        layer = int(k['name'].split('/')[1][5:])
        factory = Path(k['module']['file']).parent.name
        if kind(factory):
            by_layer.setdefault(layer, []).append((kind(factory), factory, k))
    initialize = original[f'decode/begin/k{int(metadata.get("vision") is not None)}']
    residual_role = variable(initialize, 1)

    class ShapeModel(Model):
        def kernel(self, key, build):
            if kind(key):
                return None
            return super().kernel(key, build)
    planner = ShapeModel.__new__(ShapeModel)
    planner.capacity, planner.position, planner.output = metadata['max_context'], metadata['max_context'] - 128, output
    planner.is_mtp, planner.draft_vocab, planner.decode_w8_optimized = False, None, True
    planner.layer_ids = tuple(range(48))
    planner.weights, planner.states, planner.plans, planner.workspaces, planner.kernels = {}, {}, {}, {}, {}
    planner.prefill_workspace_rows = 0
    logical = {}
    weight_tensors = {}
    types = {v[0]: getattr(torch, k) for k, v in DTYPES.items()}
    for name, b in buffers.items():
        if not name.startswith('W_'):
            continue
        match = re.fullmatch(r'W_(.+)_(short_table|packed|table|signs|value|weight|scale)', name)
        if not match:
            raise ValueError(f'Unknown Flash weight role: {name}')
        tensor = torch.empty(b['shape'], dtype=types[b['dtype']], device='meta')
        logical[tensor.untyped_storage()._cdata] = name
        weight_tensors[name] = tensor
        role, suffix = match.groups()
        if suffix == 'value':
            planner.weights[role] = tensor
        else:
            planner.weights.setdefault(role, {})[suffix] = tensor
    # Identical E8P books are physically shared across layers in native data.
    # Recover those aliases from the pinned argument bindings, without copying.
    for kernel in original.values():
        packed = [a['name'] for a in kernel['args'] if a.get('name', '').endswith('_packed')]
        if len(packed) != 1 or not packed[0].startswith('W_'):
            continue
        role = packed[0][2:-len('_packed')]
        for arg in kernel['args']:
            name = arg.get('name', '')
            suffix = 'short_table' if name.endswith('_short_table') else 'table' if name.endswith('_table') else None
            if suffix and name in weight_tensors:
                planner.weights[role][suffix] = weight_tensors[name]
    for name, value in list(planner.weights.items()):
        if isinstance(value, dict) and set(value) == {'weight', 'scale'}:
            planner.weights[name] = (value['weight'], value['scale'])
    for layer in range(48):
        if layer % 4 != 3:
            shapes = {'conv': (1, 3, 10240), 'gdn': (48, 128, 128)}
        else:
            shapes = {'key': (planner.capacity, 2, 256), 'value': (planner.capacity, 2, 256),
                      'key_scale': (planner.capacity, 2, 4), 'value_scale': (planner.capacity, 2, 4),
                      'index': ((planner.capacity + 3)//4, 128), 'pending': (4, 128)}
        for suffix, shape in shapes.items():
            name = f'State_{layer}_{suffix}'
            tensor = torch.empty(shape, device='meta', dtype=types[buffers[name]['dtype']])
            planner.states[f'{layer}:{suffix}'] = tensor
            logical[tensor.untyped_storage()._cdata] = name
    for attr, name in [('position_gpu', 'Position'), ('lengths', 'Length'), ('position_out', 'PositionOut')]:
        tensor = torch.empty(1, device='meta', dtype=torch.int32)
        setattr(planner, attr, tensor);logical[tensor.untyped_storage()._cdata] = name
    planner.states['ple'] = torch.empty((9, 10240), device='meta', dtype=torch.float16)
    logical[planner.states['ple'].untyped_storage()._cdata] = 'State_ple'

    class BatchPublisher(Publisher):
        def bind(self, *args):
            op = super().bind(*args)
            # New builds may differ from pinned verification cubins even when
            # their factory label matches. Keep both immutable identities.
            for field in ('module', 'source', 'host_abi'):
                asset = self.kernels[-1][field]
                if not asset['file'].startswith('flash-batch/'):
                    asset['file'] = 'flash-batch/' + asset['file']
            return op

        def buffer(self, tensor, name=None, scope='workspace'):
            pointer = tensor.untyped_storage()._cdata
            offset = tensor.storage_offset()*tensor.element_size()
            if pointer in logical:
                return logical[pointer], offset
            if pointer not in self.storage:
                name = f'BatchM{self.rows}_' + self.interfaces.get(pointer, f'Scratch{self.number}')
                self.number += 1
                self.storage[pointer] = name
                size = tensor.untyped_storage().nbytes()
                b = dict(name=name, dtype=DTYPES[str(tensor.dtype).removeprefix('torch.')][0], shape=[size//tensor.element_size()],
                         layout='native_contiguous', alignment=256, access='read_write')
                buffers[name] = b;data['buffer_scopes'][name] = 'workspace'
            return self.storage[pointer], offset
    publish = BatchPublisher(planner, destination)
    # Publisher paths are inside the independently cloned operator package.
    publish.cache = operator/'flash-batch'
    (publish.cache/'kernels').mkdir(parents=True)
    publish.rows = 0
    strides = {}
    programs = {}
    counts = {}
    for rows in PROFILES:
        publish.rows = rows;publish.storage = {};publish.interfaces = {}
        planner.position = planner.capacity - rows
        plan = planner.plan(rows, all_logits=True)
        publish.interfaces[plan['embedding'].untyped_storage()._cdata] = 'M1_Embedding'
        publish.interfaces[plan['ple_embedding'].untyped_storage()._cdata] = 'M1_Ple'
        publish.interfaces[plan['residual'].untyped_storage()._cdata] = residual_role
        publish.interfaces[plan['output'].untyped_storage()._cdata] = 'Logits'
        publish.interfaces[plan['token'].untyped_storage()._cdata] = 'Selected'
        # Reconstruct layer boundaries from immutable parameter identities.
        layers = [];layer = 0
        entries = list(zip(plan['ops'], plan['labels']))
        for (kernel, args), label in entries:
            weight_layers = [int(match[1]) for a in args if isinstance(a, torch.Tensor)
                             and not logical.get(a.untyped_storage()._cdata, '').endswith('_table')
                             and (match := re.match(r'W_blk\.(\d+)\.', logical.get(a.untyped_storage()._cdata, '')))]
            if weight_layers:layer = min(weight_layers)
            if label.startswith('dense-quant-') and args[0] is plan['ple_embedding']:layer = 1
            layers.append(layer)
        private = {}
        for index, (((kernel, args), label), layer) in enumerate(zip(entries, layers)):
            knd = kind(label)
            if knd not in INTERFACES or index >= plan['body_count']:
                continue
            candidates = [k for name, _, k in by_layer[layer] if name == knd]
            if len(candidates) != 1:raise ValueError(f'Ambiguous private {layer}/{knd}')
            old = candidates[0]
            for arg_index in INTERFACES[knd]:
                tensor = args[arg_index];role = variable(old, arg_index)
                pointer = tensor.untyped_storage()._cdata
                if pointer in publish.interfaces and publish.interfaces[pointer] != role:
                    raise ValueError('Batch interface alias mismatch')
                publish.interfaces[pointer] = role
                size = tensor.numel()*tensor.element_size()//rows
                if role in strides and strides[role] != size:raise ValueError('Batch interface stride mismatch')
                strides[role] = size
        for role, tensor in [(residual_role, plan['residual']), ('M1_Embedding', plan['embedding']),
                             ('M1_Ple', plan['ple_embedding']), ('Logits', plan['output']), ('Selected', plan['token'])]:
            strides[role] = tensor.numel()*tensor.element_size()//rows
            publish.buffer(tensor)
        shared = {}
        stage = 'pre'
        for index, (((kernel, args), label), layer) in enumerate(zip(entries, layers)):
            if index >= plan['body_count']:
                section = 'head'
            elif index == 0:
                section = 'begin'
            else:
                if index == 1 or layers[index-1] != layer:stage = 'ple' if layer == 1 else 'pre'
                knd = kind(label)
                if knd:
                    stage = 'pre' if knd.startswith('ple-') else 'post'
                    continue
                if label == 'gdn-history-copy':continue
                if label.startswith('index-qk:'):
                    section = f'layer{layer}/pre'
                else:section = f'layer{layer}/{stage}'
            op = publish.bind(f'flash_batch_m{rows}', section, label, args, rows)
            if any(data['buffer_scopes'].get(a.get('name')) == 'sequence' for a in publish.kernels[-1]['args']):
                raise ValueError(f'Shared Flash stage reads private request state: {section}/{label}')
            publish.groups.setdefault(f'flash_batch_m{rows}', {}).setdefault(section, []).append(op)
            shared.setdefault(section, []).append(op)
        for kernel, args in plan['greedy_ops']:
            label = f'greedy-partials-{rows}-{planner.V}' if args[0] is plan['output'] else f'greedy-merge-{rows}-{planner.V}'
            op = publish.bind(f'flash_batch_m{rows}', 'head', label, args, rows)
            publish.groups[f'flash_batch_m{rows}'].setdefault('head', []).append(op)
            shared['head'].append(op)
        for layer in range(48):
            for knd, factory, old in by_layer[layer]:
                private.setdefault(layer, []).append(old)
        # Clone validated M1 mixers and redirect only the row-major boundaries.
        for layer, old_kernels in private.items():
            for old in old_kernels:
                k = copy.deepcopy(old)
                section = f'layer{layer}/private'
                slot = len(shared.get(section, []))
                k['name'] = f'flash_batch_m{rows}/{section}/k{slot}'
                for arg in k['args']:
                    if arg.get('name') in strides:
                        arg['name'] = f'BatchM{rows}_{arg["name"]}'
                publish.kernels.append(k)
                shared.setdefault(section, []).append(dict(kind='kernel', name=k['name']))
        for section, ops in shared.items():
            name = f'flash_batch_m{rows}/{section}'
            programs[name] = ops
        counts[rows] = {section: len(ops) for section, ops in shared.items()}
        print('BATCH STAGES', rows, {k:v for k,v in counts[rows].items() if k in ('begin','head','layer0/pre','layer0/post','layer0/private','layer1/ple','layer1/pre','layer1/post','layer1/private','layer3/pre','layer3/post','layer3/private')}, flush=True)
    metadata['batch_profiles'] = list(PROFILES)
    package['batch_profiles'] = list(PROFILES)
    metadata['batch_layout'] = dict(layers=['full_attention' if i%4==3 else 'linear_attention' for i in range(48)],
                                    row_strides=strides, profiles={}, hidden=2560, small_mixed_shapes=[])
    # CPU lookup operands and target logits cannot alias across active requests.
    for name in ('M1_Embedding', 'M1_Ple', metadata['logits']):data['buffer_scopes'][name] = 'sequence'
    metadata['buffers'] = list(buffers.values())
    package['kernels'].extend(publish.kernels)
    package['buffer_contracts'] = [{k:v for k,v in b.items() if k != 'data'} for b in metadata['buffers']]
    from tools.model.package import model_library
    library = model_library();path = operator/'lib/model.so'
    path.unlink();__import__('shutil').copyfile(library, path)
    package['execution']['library']['sha256'] = file_hash(path)
    commit_package(destination, operator, data, package)
    write_json(output/'stages.json', counts)
    # The native library derives every stage and private-state invocation.
    import subprocess
    subprocess.run([str(Path('target/release/orinfer').resolve()), 'plan-model', str(destination)],
                   check=True, stdout=(output/'registered-plan.json').open('w'))
    registered = json.loads((output/'registered-plan.json').read_text())['manifest']['programs']
    for name, ops in programs.items():
        if registered.get(name) != ops:
            raise ValueError(f'Registered Flash batch stage differs from export: {name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--model-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compile-cache', type=Path)
    a = p.parse_args()
    if a.compile_cache:
        from tools.model.publication import seed_compile_cache
        seed_compile_cache(a.compile_cache, a.output/'cache/0.1.15')
    with atomic_model(a.model_output) as staging:upgrade(a.model.resolve(), staging, a.output)


if __name__ == '__main__':main()
