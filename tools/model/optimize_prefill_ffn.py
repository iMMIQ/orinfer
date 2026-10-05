"""Publish validated prefill FFN tiles without changing weights or precision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from tools.model.prepare import file_hash, write_json
from tools.model.optimize_kv import link_or_copy
from tools.model.screen_prefill_ffn import model_identity
from tools.model.upgrade_batching import bind
from tools.operators.abi import parse_host


def publish(model, screens, destination, engine):
    model = model.resolve(strict=True)
    destination = destination.absolute()
    source_raw = (model / 'cache/model.json').read_bytes()
    fingerprint = model_identity(model)
    if isinstance(screens, Path):
        screens = [screens]
    selected = {}
    for screen in screens:
        report = json.loads((screen / 'result.json').read_text())
        rows = report['rows']
        expanded = report.get('kind') == 'expanded_int8_gemm'
        if report['status'] != 'passed' or rows not in (512, 2048) or report['fingerprint'] != fingerprint:
            raise ValueError('Screen must validate this exact model and supported prefill shapes')
        if expanded != (rows == 2048):
            raise ValueError('Expected LUT4 at 512 rows or temporary-W8 GEMM at 2048 rows')
        if rows in selected:
            raise ValueError('Duplicate prefill shape in screens')
        if expanded:
            mappings = {(v['rows'], v['grid_order']) for v in report.get('mapping_checks', [])
                        if v['exact'] and v['graph_last_row'] and v['guard']}
            expected = {(m, order) for m in [321, 513] for order in
                        ['grouped4', 'grouped8', 'n1m2', 'n1m4', 'n1m8', 'n2m2', 'n2m4']}
            if mappings != expected:
                raise ValueError('Incomplete grouped launch tail verification')
        families = {}
        for family in ['GateUp', 'Down']:
            candidates = [r for r in report['cases'] if r['family'] == family
                          and r['integer_oracle_equal'] and r['tail_guard'] and r['graph_zero_restore']
                          and r.get('nonaligned_tail_rows') == rows + 1]
            if expanded:
                candidates = [r for r in candidates if r.get('selected')
                              and r.get('decoded_weight_equal') and r.get('finalist_rechecked')]
            elif report.get('kind') == 'lut4_ffn':
                candidates = [r for r in candidates if r.get('selected') and r.get('finalist_rechecked')]
            if not candidates:
                raise ValueError(f'No validated {family} candidates')
            families[family] = dict(choice=min(candidates, key=lambda r: r['timing']['median_ms']),
                                    screen=screen)
        selected[rows] = families
    if not selected:
        raise ValueError('No prefill screens supplied')
    data = json.loads(source_raw)
    meta = data['metadata']
    cache_root = Path(os.environ.get('ORIN_OPERATOR_CACHE', str(Path(os.environ.get(
        'XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'orin-llm/operators')))
    origin = cache_root / data['operator_package']
    if not origin.exists():
        origin = model / 'cache/operators' / data['operator_package']
    if file_hash(origin / 'package.json') != data['operator_package']:
        raise ValueError('Package digest mismatch')
    package = json.loads((origin / 'package.json').read_text())
    buffers = {b['name']: b for b in meta['buffers']}
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name('.' + destination.name + '.building-' + uuid.uuid4().hex)
    try:
        staging.mkdir(parents=True)
        for file in model.iterdir():
            if file.is_file():
                shutil.copyfile(file, staging / file.name)
        shutil.copytree(model / 'cache/weights', staging / 'cache/weights', copy_function=link_or_copy)
        target = staging / 'cache/operators/building'
        shutil.copytree(origin, target, copy_function=link_or_copy)
        (target / 'kernels').mkdir(exist_ok=True)
        exports = {}
        for rows, families in selected.items():
            for family, selection in families.items():
                directory = selection['screen'] / selection['choice']['export']
                host = parse_host((directory / 'host.txt').read_text())
                if len(host) != 1:
                    raise ValueError('Expected a single projection export')
                assets = {}
                for field, filename in [('module', 'kernel.cubin'), ('source', 'kernel.cu'), ('host_abi', 'host.txt')]:
                    raw = (directory / filename).read_bytes()
                    digest = hashlib.sha256(raw).hexdigest()
                    asset = 'kernels/' + digest + Path(filename).suffix
                    if not (target / asset).exists():
                        (target / asset).write_bytes(raw)
                    elif file_hash(target / asset) != digest:
                        raise ValueError('Existing asset digest mismatch')
                    assets[field] = dict(file=asset, sha256=digest)
                exports[rows, family] = dict(**host[0], **assets)
        replaced = {f'prefill_m{rows}/{family}': 0 for rows in selected for family in selected[rows]}
        for i, kernel in enumerate(package['kernels']):
            for rows, families in selected.items():
                if not kernel['name'].startswith(f'prefill_m{rows}/layer'):
                    continue
                expanded = rows == 2048
                names = [arg['name'] for arg in kernel['args'] if arg['kind'] == 'buffer']
                for family, selection in families.items():
                    choice = selection['choice']
                    suffix = '_' + family + ('_WS' if expanded else '_P')
                    weight = next((n for n in names if n.endswith(suffix)), None)
                    if weight is None or 'TemporaryA8' not in names:
                        continue
                    if expanded and 'TemporaryW8' not in names:
                        continue
                    prefix = weight[:-3] if expanded else weight[:-2]
                    if buffers[prefix + '_S']['shape'] != [choice['n'], choice['k'] // 128]:
                        raise ValueError('Projection shape differs from its validated export')
                    old = parse_host((origin / kernel['host_abi']['file']).read_text())[0]
                    pointers = {formal['value'].removesuffix('.data_ptr()'): arg['name']
                                for formal, arg in zip(old['ordered_arguments'], kernel['args'])
                                if arg['kind'] == 'buffer'}
                    package['kernels'][i] = bind(exports[rows, family], kernel['name'], pointers,
                                                dict(rows=rows, M=rows, batch=1))
                    replaced[f'prefill_m{rows}/{family}'] += 1
        config = json.loads((model / 'config.json').read_text())
        layers = len(config.get('text_config', config)['layer_types'])
        if any(count != layers for count in replaced.values()):
            raise ValueError('Incomplete FFN projection replacement')
        raw = (json.dumps(package, ensure_ascii=False, indent=2) + '\n').encode()
        digest = hashlib.sha256(raw).hexdigest()
        # Never truncate a hardlink shared with the source package.
        (target / 'package.new.json').write_bytes(raw)
        (target / 'package.new.json').replace(target / 'package.json')
        target.rename(target.with_name(digest))
        data['operator_package'] = digest
        write_json(staging / 'cache/model.json', data)
        subprocess.run([str(engine.resolve()), 'validate-model', str(staging)],
                       check=True, stdout=subprocess.DEVNULL)
        staging.rename(destination)
        return dict(operator_package=digest,
                    selected={str(rows): {family: v['choice'] for family, v in families.items()}
                              for rows, families in selected.items()}, replaced=replaced,
                    weight_bytes=meta['weight_bytes'], persistent_weight_bytes_added=0)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--screen', type=Path, nargs='+', required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=Path('target/release/orin-llm'))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = publish(args.model, args.screen, args.destination, args.engine)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)


if __name__ == '__main__':
    main()
