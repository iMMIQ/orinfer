"""Replace measured prefill AOT kernels while retaining immutable weights.

Only state/output and norm outputs with verified absent consumers change.
Actual old/new exported ABI determines argument order, bounds and resources.
Decode kernels, programs, buffers and weight budgets must remain identical.
"""
import argparse
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil

from tools.operators.abi import evaluate, parse_host


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--gdn-exports', type=Path, required=True)
    ap.add_argument('--norm-exports', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), args.output
    gdn_report = json.loads((args.gdn_exports/'result.json').read_text())
    norm_report = json.loads((args.norm_exports/'result.json').read_text())
    assert gdn_report['status'] == norm_report['status'] == 'passed'
    assert gdn_report['policy'] == 'high' and gdn_report['flags'] == [False]*4
    assert {r['T'] for r in gdn_report['cases']} >= {512,513,2048,8192}
    assert {r['rows'] for r in norm_report['cases']} >= {512,513,2048,8192}
    source = args.model.parent
    original = json.loads(args.model.read_text())
    model = copy.deepcopy(original)
    assert {p['chunk_tokens'] for p in model['prefill_plans']} == {512,2048,8192}
    old_kernels = {k['name']: k for k in original['kernels']}
    args.output.mkdir(parents=True)
    copied = {}
    hash_cache = {}

    def hash_once(path):
        if path not in hash_cache:
            hash_cache[path] = digest(path)
        return hash_cache[path]

    def preserve(ref):
        src, dst = source/ref['file'], args.output/ref['file']
        assert hash_once(src) == ref['sha256'], src
        if ref['file'] in copied:
            assert copied[ref['file']] == ref['sha256']
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError as error:
            if error.errno not in (errno.EPERM, errno.EACCES, errno.EXDEV):
                raise
            shutil.copyfile(src, dst)
            assert digest(dst) == ref['sha256']
        copied[ref['file']] = ref['sha256']

    # Materialize all data before the final manifest, retaining source identity.
    for buffer in model['buffers']:
        if buffer.get('data'):
            preserve(buffer['data'])
    exports = {}
    for family, root in [('state',args.gdn_exports/'state'),
                         ('output',args.gdn_exports/'output'),
                         ('norm-no-y',args.norm_exports/'no-y')]:
        rel = Path('aot-gdn-high')/family
        dest = args.output/rel
        dest.mkdir(parents=True)
        refs = {}
        for field, filename in [('module','kernel.cubin'),('source','kernel.cu'),('host_abi','host.txt')]:
            shutil.copyfile(root/filename, dest/filename)
            refs[field] = dict(file=str(rel/filename), sha256=digest(dest/filename))
        launches = parse_host((root/'host.txt').read_text())
        assert len(launches)==1
        exports[family] = (refs, launches[0])

    added = {}
    counts = {}
    for phase, program in model['programs'].items():
        if not phase.startswith('prefill'):
            continue
        norms, consumers = [], []
        for index, op in enumerate(program):
            if op['kind'] != 'kernel':
                continue
            kernel = old_kernels[op['name']]
            family = Path(kernel['module']['file']).parent.name
            if any(a.get('name')=='Norm' for a in kernel['args']):
                if family.startswith('norm_a8'):
                    norms.append(index)
                else:
                    assert family.startswith('prefill_ab'), family
                    consumers.append(index)
        used_norms = {max(n for n in norms if n<c) for c in consumers}
        assert len(norms)==128 and len(used_norms)==48
        dead_norms = set(norms)-used_norms
        counts[phase] = dict(state=0, output=0, **{'norm-no-y':0})
        for index, op in enumerate(program):
            if op['kind'] != 'kernel':
                continue
            old = old_kernels[op['name']]
            old_family = Path(old['module']['file']).parent.name
            family = ('state' if old_family=='state_factored' else
                      'output' if old_family=='output_factored' else
                      'norm-no-y' if index in dead_norms else None)
            if family is None:
                continue
            old_launches = parse_host((source/old['host_abi']['file']).read_text())
            assert len(old_launches)==1
            old_order = old_launches[0]['ordered_arguments']
            assert len(old_order)==len(old['args'])
            bindings = {a['value']: bound for a,bound in zip(old_order,old['args'])}
            variables = {a['value']: bound['value'] for a,bound in zip(old_order,old['args'])
                         if bound['kind']=='i32'}
            refs, launch = exports[family]
            expr = launch['launch_expressions']
            name = old['name']+'_'+family.replace('-','_')+'_high'
            new = dict(copy.deepcopy(refs), name=name, symbol=launch['symbol'], cooperative=False,
                       grid=[evaluate(expr['gridDim'+axis],variables) for axis in 'XYZ'],
                       block=[evaluate(expr['blockDim'+axis],variables) for axis in 'XYZ'],
                       shared_memory_bytes=evaluate(expr['sharedMemBytes'],variables),
                       args=[copy.deepcopy(bindings[a['value']]) for a in launch['ordered_arguments']])
            assert new['grid']==old['grid'] and new['block']==old['block']
            if family=='norm-no-y':
                assert 'Y.data_ptr()' not in {a['value'] for a in launch['ordered_arguments']}
                assert len(new['args'])==len(old['args'])-1
            else:
                assert new['args']==old['args']
            if name in added:
                assert added[name]==new
            else:
                added[name]=new
            program[index]=dict(kind='kernel',name=name)
            counts[phase][family]+=1
        assert counts[phase]==dict(state=48,output=48,**{'norm-no-y':80})
    used={op['name'] for ops in model['programs'].values() for op in ops if op['kind']=='kernel'}
    model['kernels']=[k for k in model['kernels'] if k['name'] in used]+list(added.values())
    for kernel in model['kernels']:
        for field in ('module','source','host_abi'):
            if not kernel[field]['file'].startswith('aot-gdn-high/'):
                preserve(kernel[field])
    assert model['buffers']==original['buffers']
    assert model['programs']['decode']==original['programs']['decode']
    selected = {op['name'] for op in original['programs']['decode'] if op['kind']=='kernel'}
    assert {k['name']:k for k in model['kernels'] if k['name'] in selected} == {
        k['name']:k for k in original['kernels'] if k['name'] in selected}
    assert model['weight_bytes']==original['weight_bytes']
    manifest = args.output/'model.json'
    manifest.write_text(json.dumps(model,indent=2)+'\n')
    record = dict(status='assembled', source_model=str(args.model),
                  source_manifest_sha256=digest(args.model), manifest_sha256=digest(manifest),
                  gdn_report=identity(args.gdn_exports/'result.json'),
                  norm_report=identity(args.norm_exports/'result.json'), changes=counts,
                  weight_bytes=model['weight_bytes'], buffers_identical=True,
                  decode_program_and_kernels_identical=True,
                  scope='Prefill-only FP16 product policy with FP32 state, '
                        '80 norm outputs removed per plan; model performance/quality pending')
    (args.output/'assembly.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copyfile(__file__,args.output/Path(__file__).name)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
