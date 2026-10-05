"""Publish symbolic-row launch contracts from the verified exported host ABI.

No weights or cubins change. Rust specializes launch/graph plans for exact
nonstandard rows; common optimized batch profiles retain their existing ABI.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from tools.model.prepare import file_hash, write_json
from tools.operators.abi import parse_host, evaluate


def expression(source):
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 0xffffffff:
            return dict(op='constant', value=node.value)
        if isinstance(node, ast.Name) and node.id in ('M', 'rows', 'batch'):
            return dict(op='rows')
        operators = {ast.Add:'add', ast.Sub:'subtract', ast.Mult:'multiply',
                     ast.FloorDiv:'divide', ast.Mod:'remainder'}
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return dict(op=operators[type(node.op)], lhs=visit(node.left), rhs=visit(node.right))
        raise ValueError(f'Unsupported row expression: {source}')
    return visit(ast.parse(source, mode='eval').body)


def contract(kernel, host):
    dimensions = dict(M=128, rows=128, batch=128)
    launch = host['launch_expressions']
    if kernel['grid'] != [evaluate(launch['gridDim'+axis], dimensions) for axis in 'XYZ']:
        raise ValueError('Host grid differs from capacity binding')
    if kernel['block'] != [evaluate(launch['blockDim'+axis], dimensions) for axis in 'XYZ']:
        raise ValueError('Host block differs from capacity binding')
    if kernel['shared_memory_bytes'] != evaluate(launch['sharedMemBytes'], dimensions):
        raise ValueError('Host shared memory differs from capacity binding')
    if len(kernel['args']) != len(host['ordered_arguments']):
        raise ValueError('Host argument count differs')
    args = []
    for index, (bound, exported) in enumerate(zip(kernel['args'], host['ordered_arguments'])):
        if exported['ctype'] in ('ctypes.c_int32', 'c_int32'):
            if bound != dict(kind='i32', value=evaluate(exported['value'], dimensions)):
                raise ValueError('Host scalar differs from capacity binding')
            args.append(dict(index=index, value=expression(exported['value'])))
        elif exported['ctype'] not in ('ctypes.c_void_p', 'c_void_p') or bound['kind'] != 'buffer':
            raise ValueError('Unsupported dynamic host ABI')
    grid = [expression(launch['gridDim'+axis]) for axis in 'XYZ']
    if kernel['name'].startswith('batch_gdn_m128/'):
        # These row-independent CTA bodies address one private arena per b.
        # The fixed capacity export changes only the number of batch CTAs.
        slot = kernel['name'].rsplit('/k', 1)[1]
        axis = {'0':1, '1':2}.get(slot)
        if axis is None or kernel['grid'][axis] != 128 or args:
            raise ValueError('Unsupported batched GDN launch')
        grid[axis] = dict(op='rows')
    return dict(name=kernel['name'], grid=grid, arguments=args)


def upgrade(model, destination):
    model = model.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise ValueError('Destination already exists')
    data = json.loads((model/'cache/model.json').read_text())
    operator_cache = Path(os.environ.get('ORIN_OPERATOR_CACHE',
        str(Path(os.environ.get('XDG_CACHE_HOME',str(Path.home()/'.cache')))/'orin-llm/operators')))
    origin = operator_cache/data['operator_package']
    if not origin.exists(): origin = model/'cache/operators'/data['operator_package']
    if file_hash(origin/'package.json') != data['operator_package']:
        raise ValueError('Source package digest mismatch')
    package = json.loads((origin/'package.json').read_text())
    if 128 not in package.get('batch_profiles', []) or package.get('dynamic_batch_kernels'):
        raise ValueError('Requires capacity128 without dynamic contracts')
    contracts, parsed = [], {}
    for kernel in package['kernels']:
        if not kernel['name'].startswith(('batch_m128/', 'batch_gdn_m128/')): continue
        identity = kernel['host_abi']
        host_path = (origin/identity['file']).resolve(strict=True)
        if not host_path.is_relative_to(origin.resolve()) or file_hash(host_path) != identity['sha256']:
            raise ValueError('Host ABI identity mismatch')
        if host_path not in parsed: parsed[host_path] = parse_host(host_path.read_text())
        hosts = parsed[host_path]
        if len(hosts) != 1 or hosts[0]['symbol'] != kernel['symbol']:
            raise ValueError('Expected one matching host export')
        contracts.append(contract(kernel, hosts[0]))
    if not contracts: raise ValueError('No dynamic capacity kernels')
    package['dynamic_batch_kernels'] = contracts
    raw = (json.dumps(package,ensure_ascii=False,indent=2)+'\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{destination.name}.', dir=destination.parent) as temporary_root:
        staged = Path(temporary_root)/'model'
        staged.mkdir()
        for path in model.iterdir():
            if path.is_file(): shutil.copyfile(path,staged/path.name)
        cache = staged/'cache';cache.mkdir()
        shutil.copytree(model/'cache/weights',cache/'weights',copy_function=os.link)
        operator = cache/'operators'/digest
        shutil.copytree(origin,operator,copy_function=os.link)
        # Replacement leaves the source's hardlinked metadata untouched.
        temporary = operator/'package.json.tmp';temporary.write_bytes(raw)
        temporary.replace(operator/'package.json')
        data['operator_package'] = digest
        write_json(cache/'model.json',data)
        os.rename(staged,destination)
    return dict(operator_package=digest, dynamic_templates=len(contracts),
                weight_bytes=data['metadata']['weight_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    if args.report.exists(): parser.error('Report must be a fresh path')
    write_json(args.report,upgrade(args.model,args.output))


if __name__ == '__main__': main()
