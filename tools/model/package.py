"""Split a prepared safetensors cache into model data and an operator package.

Only an offline schema-2 cache is imported here. The online loader builds all
execution order from registered Rust architecture recipes. No weights are
requantized; existing immutable shards are hardlinked, or copied across devices.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.model.prepare import file_hash, source_path, write_json


TEXT_KEYS = (
    'hidden_size', 'intermediate_size', 'num_hidden_layers', 'vocab_size',
    'layer_types', 'num_attention_heads', 'num_key_value_heads', 'head_dim',
    'linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim',
    'linear_value_head_dim', 'linear_conv_kernel_dim', 'hidden_act', 'rms_norm_eps',
    'rope_parameters', 'partial_rotary_factor', 'attn_output_gate', 'output_gate_type',
    'attention_bias', 'tie_word_embeddings', 'mtp_num_hidden_layers', 'mtp_use_dedicated_embeddings',
)


def config_signature(config):
    text = config.get('text_config', config)
    return {'text': {k: text[k] for k in TEXT_KEYS if k in text},
            'vision': config.get('vision_config')}


def section_ops(model, config):
    """Assign stable implementation slots; never publish source control flow."""
    kernels = {k['name']: k for k in model['kernels']}
    text = config.get('text_config', config)
    sections = {}
    profiles = []

    def layer_start(op):
        if op['kind'] != 'kernel':
            return None
        return next((int(match[1]) for arg in kernels[op['name']]['args']
                     if (match := re.fullmatch(r'L(\d+)_PreWeight', arg.get('name', '')))), None)

    for profile in model['prefill_plans'] + [{'chunk_tokens': 1, 'prefill_program': 'decode'}]:
        tokens = profile['chunk_tokens']
        program = profile['prefill_program']
        ops = model['programs'][program]
        starts = [(i, layer_start(op)) for i, op in enumerate(ops) if layer_start(op) is not None]
        if [layer for _, layer in starts] != list(range(text['num_hidden_layers'])):
            raise ValueError(f'{program}: layer bindings differ from config')
        end = next((i for i in range(starts[-1][0], len(ops))
                    if ops[i]['kind'] == 'kernel' and '_advance_' in ops[i]['name']), None)
        if end is None:
            raise ValueError(f'{program}: missing advance binding')
        sections[program] = {'begin': ops[:starts[0][0]], 'end': ops[end:]}
        for j, (index, layer) in enumerate(starts):
            stop = starts[j + 1][0] if j + 1 < len(starts) else end
            sections[program][f'layer{layer}'] = ops[index:stop]
        if program != 'decode':
            # Recipe selection comes from the actual implementation, not token count.
            names = [op['name'] for op in ops if op['kind'] == 'kernel']
            kind = ('sequence' if any('_sequence_' in f'_{n}' for n in names) else
                    'chunk_lut4' if any('_lut4_native' in n for n in names) else 'chunk_expanded')
            profiles.append({'tokens': tokens, 'kind': kind})
            head = profile['head_program']
            sections[head] = {'body': model['programs'][head]}
    if model.get('vision'):
        depth = config['vision_config']['depth']
        for plan in model['vision']['plans']:
            program = plan['program']
            ops = model['programs'][program]
            if len(ops) != 2 + depth * 10 + 3:
                raise ValueError('Unsupported vision operator recipe')
            sections[program] = {'begin': ops[:2], 'end': ops[-3:]}
            for layer in range(depth):
                sections[program][f'layer{layer}'] = ops[2 + layer * 10:2 + (layer + 1) * 10]
    if model.get('mtp'):
        mtp = model['mtp']
        for plan in mtp['capture_plans']:
            sections[plan['program']] = {'body': model['programs'][plan['program']]}
        for plan in mtp['warm_plans']:
            for field in ('program', 'head_program'):
                sections[plan[field]] = {'body': model['programs'][plan[field]]}
        gdn = [i for i, kind in enumerate(text['layer_types']) if kind == 'linear_attention']
        for plan in mtp['verification_plans']:
            verify = model['programs'][plan['program']]
            prefill = model['programs'][f'prefill_m{plan["tokens"]}']
            if verify[:-4] != prefill or any(op['kind'] != 'kernel' for op in verify[-4:]):
                raise ValueError('Unsupported MTP verification head recipe')
            sections[plan['program']] = {'head': verify[-4:]}
            ops = model['programs'][plan['restore_program']]
            if len(ops) != len(gdn) * 2:
                raise ValueError('Unsupported MTP restore recipe')
            sections[plan['restore_program']] = {f'layer{layer}': ops[j * 2:j * 2 + 2]
                                                 for j, layer in enumerate(gdn)}
    return sections, profiles


def scope(buffer, model):
    if buffer['access'] == 'read':
        return 'weights'
    name = buffer['name']
    state = set(model['reset_buffers']) | {model[k] for k in ('input', 'token', 'status', 'position')}
    state.update(('Features', 'FeatureIndex', 'MtpFeatureIndex', 'MRopePositions', 'MtpCondition',
                  'MtpTargetHidden', 'MtpInput', 'MtpToken', 'MtpStatus', 'MtpSeqLength', 'MtpPositions',
                  'SeqLength', 'SequenceTokens', 'SequenceStatus', 'AcceptedInputs'))
    if name in state or re.match(r'L\d+_(Sequence|Saved|State|History|KPages|VPages)', name):
        return 'sequence'
    return 'workspace'


def validate_plan(directory, source, mapping, engine):
    """Require exact graph, ABI and launch equivalence before publishing."""
    completed = subprocess.run([str(engine.resolve(strict=True)), 'plan-model', str(directory)],
                               check=True, capture_output=True, text=True)
    plan = json.loads(completed.stdout)['manifest']
    if set(plan['programs']) != set(source['programs']):
        raise ValueError('Registered architecture does not cover all source programs')
    before = {k['name']: k for k in source['kernels']}
    after = {k['name']: k for k in plan['kernels']}
    for program, original_ops in source['programs'].items():
        generated_ops = plan['programs'][program]
        if len(original_ops) != len(generated_ops):
            raise ValueError(f'{program}: registered operation count differs')
        for index, (left, right) in enumerate(zip(original_ops, generated_ops)):
            if left['kind'] != 'kernel':
                equal = left == right
            else:
                equal = (right['kind'] == 'kernel' and mapping[right['name']] == left['name']
                         and {k: v for k, v in before[left['name']].items() if k != 'name'}
                         == {k: v for k, v in after[right['name']].items() if k != 'name'})
            if not equal:
                raise ValueError(f'{program}/{index}: registered operation or binding differs')


def publish(directory, engine=None):
    """Replace a staging-only schema-2 plan with data/implementation contracts."""
    cache = directory / 'cache'
    model = json.loads((cache / 'manifest.json').read_text())
    config = json.loads((directory / 'config.json').read_text())
    if model['schema_version'] != 2 or config.get('model_type') not in ('qwen3_5', 'qwen3_5_text'):
        raise ValueError('Expected a prepared Qwen3_5 safetensors model')
    sections, profiles = section_ops(model, config)
    original = {k['name']: k for k in model['kernels']}
    checked = set()
    for kernel in original.values():
        for field in ('module', 'source', 'host_abi'):
            asset = kernel[field]
            key = (asset['file'], asset['sha256'])
            if key not in checked:
                if file_hash(source_path(cache.resolve(), asset['file'])) != asset['sha256']:
                    raise ValueError('Source operator asset sha256 mismatch')
                checked.add(key)
    bindings = []
    binding_map = {}
    for program, groups in sections.items():
        for group, ops in groups.items():
            slot = 0
            for op in ops:
                if op['kind'] != 'kernel':
                    continue
                item = copy.deepcopy(original[op['name']])
                item['name'] = f'{program}/{group}/k{slot}'
                bindings.append(item)
                binding_map[item['name']] = op['name']
                slot += 1
    package = dict(schema_version=1, runtime_abi=1, target='sm_87', architecture='qwen3_5',
                   compute_policy='int8_quality', config_signature=config_signature(config),
                   prefill_profiles=profiles, kernels=bindings,
                   buffer_contracts=[{k: v for k, v in b.items() if k != 'data'} for b in model['buffers']],
                   toolchain={k: model['toolchain'][k] for k in ('torch', 'tilelang', 'cuda')
                              if k in model['toolchain']})
    raw = (json.dumps(package, indent=2, ensure_ascii=False) + '\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    destination = cache / 'operators' / digest
    destination.mkdir(parents=True)
    (destination / 'package.json').write_bytes(raw)
    (cache / 'kernels').rename(destination / 'kernels')
    # Include project licensing in independently distributed packages.
    repo = Path(__file__).resolve().parents[2]
    for name in ('LICENSE', 'COPYING.LESSER', 'COPYING', 'THIRD_PARTY_NOTICES.md'):
        if (repo / name).is_file():
            shutil.copyfile(repo / name, destination / name)
    metadata = {k: v for k, v in model.items() if k not in ('kernels', 'programs')}
    write_json(cache / 'model.json', dict(schema_version=1, architecture='qwen3_5',
               compute_policy='int8_quality', operator_package=digest,
               buffer_scopes={b['name']: scope(b, model) for b in model['buffers']}, metadata=metadata))
    validate_plan(directory, model, binding_map, engine or repo / 'target/release/orin-llm')
    (cache / 'manifest.json').unlink()
    return {'operator_package': digest, 'kernel_bindings': len(bindings), 'binding_map': binding_map}


def split(model, output, engine=None):
    model = model.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        def link_or_copy(source, destination):
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
        # Only our immutable, prepared directory is accepted. Config/metadata are
        # copied independently; large immutable tensors share storage on this host.
        for file in model.iterdir():
            if file.is_file():
                shutil.copyfile(file, staging / file.name)
        cache = staging / 'cache'
        cache.mkdir()
        shutil.copyfile(model / 'cache/manifest.json', cache / 'manifest.json')
        for subdir in ('weights', 'kernels'):
            shutil.copytree(model / 'cache' / subdir, cache / subdir, copy_function=link_or_copy)
        report = publish(staging, engine)
        if output.exists() or output.is_symlink():
            raise ValueError('Output appeared during packaging')
        staging.rename(output)
        return report
    except BaseException:
        shutil.rmtree(staging)
        raise


def archive(package, output):
    package = package.resolve(strict=True)
    if output.exists():
        raise ValueError('Archive already exists')
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        with tarfile.open(temporary, 'w:gz') as stream:
            stream.add(package, arcname=package.name)
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def install(archive_path, cache):
    cache.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.install-', dir=cache))
    try:
        with tarfile.open(archive_path, 'r:gz') as stream:
            members = stream.getmembers()
            roots = {Path(m.name).parts[0] for m in members if Path(m.name).parts}
            if len(roots) != 1 or any(not (m.isfile() or m.isdir()) for m in members):
                raise ValueError('Expected one package directory containing regular files')
            digest = roots.pop()
            if not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('Malformed operator package digest')
            if (cache / digest).exists():
                raise ValueError('Operator package already installed')
            for member in members:
                p = Path(member.name)
                if p.is_absolute() or '..' in p.parts:
                    raise ValueError('Unsafe archive path')
            stream.extractall(staging)
        package = staging / digest
        if file_hash(package / 'package.json') != digest:
            raise ValueError('Operator package digest mismatch')
        manifest = json.loads((package / 'package.json').read_text())
        checked = set()
        for kernel in manifest['kernels']:
            for field in ('module', 'source', 'host_abi'):
                asset = kernel[field]
                key = (asset['file'], asset['sha256'])
                if key not in checked:
                    if file_hash(source_path(package.resolve(), asset['file'])) != asset['sha256']:
                        raise ValueError('Operator asset sha256 mismatch')
                    checked.add(key)
        package.rename(cache / digest)
        return digest
    finally:
        shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('split')
    build.add_argument('--model', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    pack = sub.add_parser('archive')
    pack.add_argument('package', type=Path)
    pack.add_argument('output', type=Path)
    add = sub.add_parser('install')
    add.add_argument('archive', type=Path)
    add.add_argument('cache', type=Path)
    args = parser.parse_args()
    if args.command == 'split':
        report = split(args.model, args.output)
        report.pop('binding_map')
        print(json.dumps(report, indent=2))
    elif args.command == 'archive':
        archive(args.package, args.output)
    else:
        print(install(args.archive, args.cache))


if __name__ == '__main__':
    main()
