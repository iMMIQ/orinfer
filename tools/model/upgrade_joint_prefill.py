"""Add dense shared prefill projections; causal request plans remain in Rust.

The package declares row capacities and bindings, never a model execution plan.
All resident tensors are reused unchanged. Shared FFN kernels retain the
existing 512-token LUT4 codebook so merging requests does not change weights.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from tools.model.publication import atomic_model, clone_model, commit_package, load_model, write_json
from tools.model.upgrade_batching import bind
from tools.operators.abi import parse_host


def upgrade(model, destination, output):
    import torch
    from kernels.model.w4a8 import gdn_qkvz_int8
    from kernels.model.w4a8_lut4 import w4a8_lut4
    from kernels.projections.candidates import int8_gemm
    from kernels.operators.op07_gdn_ab import gdn_ab_tensorcore
    from tools.operators.common import configure, export_kernel, error, benchmark

    configure()
    torch.empty(1, device='cuda')
    data, origin, package = load_model(model)
    meta = data['metadata']
    config = json.loads((model / 'config.json').read_text())
    text = config.get('text_config', config)
    if (text['hidden_size'], text['intermediate_size']) != (5120, 17408):
        raise ValueError('The projection factories require the 27B geometry')
    if meta['chunk_tokens'] != 2048 or not meta['kv_cache'].get('prefill_workspace'):
        raise ValueError('Requires 2048-token workspace and staged INT8 KV')
    if not package.get('batch_profiles') or package.get('prefill_batch_profiles'):
        raise ValueError('Requires batching without joint prefill profiles')
    target = clone_model(model, destination, origin)
    kernels = {k['name']: k for k in package['kernels']}
    exports = {}
    checks = []

    def compile_export(key, kernel):
        print('compile', key, flush=True)
        directory = output / key
        export_kernel(kernel, directory)
        host = parse_host((directory / 'host.txt').read_text())
        if len(host) != 1:
            raise ValueError('Expected one host export')
        assets = {}
        for field, filename in [('module', 'kernel.cubin'), ('source', 'kernel.cu'), ('host_abi', 'host.txt')]:
            raw = (directory / filename).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            path = 'kernels/' + digest + Path(filename).suffix
            if not (target / path).exists():
                (target / path).write_bytes(raw)
            assets[field] = dict(file=path, sha256=digest)
        exports[key] = dict(**host[0], **assets)

    # Every new static-row GEMM gets an independent INT32-dot oracle and a
    # changed-input graph replay, including a guard beyond the real output.
    def validate_gemm(key, kernel, n, k, qkv=False):
        a = torch.randint(-8, 9, (1024, k), dtype=torch.int8, device='cuda')
        b = torch.randint(-8, 9, (n, k), dtype=torch.int8, device='cuda')
        sa = torch.full((1024,), .125, dtype=torch.float16, device='cuda')
        sb = torch.full((n,), .125, dtype=torch.float16, device='cuda')
        y = torch.full((1025, n), 91., dtype=torch.float16, device='cuda')
        if qkv:
            q = torch.full((1025, 10240), 91., dtype=torch.float16, device='cuda')
            z = torch.full((1025, 6144), 91., dtype=torch.float16, device='cuda')
            run = lambda: kernel(a, b, sa, sb, q[:1024], z[:1024])
        else:
            run = lambda: kernel(a, b, sa, sb, y[:1024])
        timing, graph = benchmark(run, repetitions=2)
        reference = ((a.float() @ b.float().T) * .015625).half()
        actual = torch.cat((q[:1024], z[:1024]), 1) if qkv else y[:1024]
        exact = bool(torch.equal(actual, reference))
        a.zero_(); graph.replay(); torch.cuda.synchronize()
        changed = bool(torch.count_nonzero(q[:1024]) == 0 and torch.count_nonzero(z[:1024]) == 0) if qkv else bool(torch.count_nonzero(y[:1024]) == 0)
        guard = bool((q[1024] == 91).all() and (z[1024] == 91).all()) if qkv else bool((y[1024] == 91).all())
        checks.append(dict(kernel=key, integer_oracle_equal=exact, changed_graph_replay=changed, guard=guard, timing=timing))
        if not (exact and changed and guard):
            raise ValueError('New joint projection validation failed: ' + key)

    factories = {
        'qkvz': (lambda: gdn_qkvz_int8(1024), 16384, 5120, True),
        'out': (lambda: int8_gemm(1024, 5120, 6144, 256, 128, 128, 2, 256), 5120, 6144, False),
        'full': (lambda: int8_gemm(1024, 14336, 5120, 256, 128, 128, 2, 256), 14336, 5120, False),
    }
    for key, (factory, n, k, qkv) in factories.items():
        kernel = factory()
        validate_gemm(key, kernel, n, k, qkv)
        compile_export(key, kernel)
    ab = gdn_ab_tensorcore(1024, BM=32, output_dtype='float16')
    x = torch.randn((1024, 5120), dtype=torch.float16, device='cuda') * .1
    w = torch.randn((96, 5120), dtype=torch.float16, device='cuda') * .1
    y = torch.full((1025, 96), 91., dtype=torch.float16, device='cuda')
    _, graph = benchmark(lambda: ab(x, w, y[:1024]), repetitions=2)
    metric = error(y[:1024], (x.float() @ w.float().T).half())
    x.zero_(); graph.replay(); torch.cuda.synchronize()
    valid = metric['finite'] and metric['relative_l2'] < .002 and bool((y[:1024] == 0).all()) and bool((y[1024] == 91).all())
    checks.append(dict(kernel='ab', error=metric, changed_graph_replay=valid))
    if not valid:
        raise ValueError('New AB projection validation failed')
    compile_export('ab', ab)

    from safetensors import safe_open
    from tools.model.screen_prefill_ffn import decode_w8
    index = json.loads((model / 'cache/weights/model.safetensors.index.json').read_text())['weight_map']
    def tensor(name):
        with safe_open(str(model / 'cache/weights' / index[name]), framework='pt', device='cpu') as f:
            return f.get_tensor(name).cuda()
    for family, n, k in [('GateUp',34816,5120), ('Down',5120,17408)]:
        pp, scales, step, coef, ws = [tensor('L0_' + family + suffix)
                                    for suffix in ['_P','_S','_Step','_LUT4','_WS']]
        decoded = torch.from_numpy(decode_w8(pp,step,coef,n,k)).cuda()
        for rows in [1024,2048]:
            key = f'{family}-m{rows}'
            kernel = w4a8_lut4(rows,n,k,BM=256,BN=64,stages=2,coalesced_epilogue=True)
            a = torch.randint(-4,5,(rows,k),dtype=torch.int8,device='cuda')
            sa = torch.full((rows,),.125,dtype=torch.float16,device='cuda')
            y = torch.full((rows+1,n),91.,dtype=torch.float16,device='cuda')
            timing, graph = benchmark(lambda:kernel(a.view(torch.uint32),pp,scales,step,coef,sa,ws,y[:rows]),repetitions=2)
            reference = ((a.float() @ decoded.float().T) * sa.float()[:,None] * ws.float()[None,:]).half()
            exact = bool(torch.equal(y[:rows],reference))
            a.zero_();graph.replay();torch.cuda.synchronize()
            changed = bool((y[:rows]==0).all());guard=bool((y[rows]==91).all())
            checks.append(dict(kernel=key,integer_oracle_equal=exact,changed_graph_replay=changed,guard=guard,timing=timing))
            if not (exact and changed and guard):
                raise ValueError('LUT4 joint projection validation failed: '+key)
            compile_export(key,kernel)

    for rows in [512, 1024, 2048]:
        template_rows = 512 if rows == 512 else 2048
        for layer, kind in enumerate(text['layer_types']):
            gdn = kind == 'linear_attention'
            slots = list(range(5)) + list(range(12,19)) if gdn else list(range(3)) + list(range(6,13))
            replacements = {2:'qkvz',3:'ab',14:'out'} if gdn else {2:'full',8:'out'}
            for slot in slots:
                ffn = slot >= (16 if gdn else 10)
                template = kernels[f'prefill_m{512 if ffn else template_rows}/layer{layer}/k{slot}']
                name = f'prefill_batch_m{rows}/layer{layer}/k{slot}'
                if rows == 512 or (rows == 2048 and not ffn):
                    kernels[name] = dict(copy.deepcopy(template), name=name)
                    continue
                host = parse_host((origin / template['host_abi']['file']).read_text())[0]
                pointers = {formal['value'].removesuffix('.data_ptr()'): arg['name']
                            for formal, arg in zip(host['ordered_arguments'], template['args']) if arg['kind'] == 'buffer'}
                family = 'GateUp' if slot == (16 if gdn else 10) else 'Down' if slot == (18 if gdn else 12) else None
                export = exports[f'{family}-m{rows}'] if family else exports[replacements[slot]] if slot in replacements else dict(
                    **host, **{f: template[f] for f in ('module', 'source', 'host_abi')})
                kernels[name] = bind(export, name, pointers, dict(rows=rows, M=rows, batch=1))
    package['prefill_batch_profiles'] = [dict(tokens=n, kind='chunk_lut4') for n in [512,1024,2048]]
    package['kernels'] = list(kernels.values())
    digest = commit_package(destination, target, data, package)
    write_json(output / 'upgrade.json', dict(operator_package=digest, checks=checks,
        persistent_weight_bytes_added=0, workspace_bytes_added=0, profiles=package['prefill_batch_profiles']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--model-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=Path('target/release/orin-llm'))
    a = p.parse_args()
    destination = a.model_output.absolute()
    if destination.exists():
        raise FileExistsError(destination)
    a.output.mkdir(parents=True, exist_ok=True)
    with atomic_model(destination) as staging:
        upgrade(a.model.resolve(strict=True), staging, a.output)


if __name__ == '__main__':
    main()
