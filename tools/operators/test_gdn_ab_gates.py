"""GPU integration check: op07 packed output -> op09 without a repack."""
import argparse
from pathlib import Path
import shutil
import torch
import tilelang.language as T
from safetensors import safe_open
from common import configure, benchmark, error, identity, export_kernel, write_json, environment
from abi import parse_host
from op07_gdn_ab import input_rows, MODEL
from kernels.operators.op07_gdn_ab import gdn_ab_simt, gdn_ab_tensorcore
from kernels.operators.op09_gdn_gates import gdn_gates, launch

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    out = Path(parser.parse_args().output)
    out.mkdir(parents=True, exist_ok=True)
    configure()
    report = {'environment': environment(), 'cases': [], 'exports': [],
              'scope': 'Packed interface integration, not a model-quality evaluation'}
    frozen = out / 'dependencies'
    frozen.mkdir()
    report['sources'] = []
    for name in ('kernels/operators/op07_gdn_ab.py', 'kernels/operators/op09_gdn_gates.py',
                 'tools/operators/op07_gdn_ab.py', 'tools/operators/abi.py'):
        dest = frozen / Path(name).name
        shutil.copyfile(ROOT / name, dest)
        report['sources'].append(identity(dest))
    with safe_open(str(MODEL), framework='pt', device='cpu') as model:
        prefix = 'model.language_model.layers.0.linear_attn.'
        weight = torch.cat([model.get_tensor(prefix + f'in_proj_{part}.weight')
                            for part in ('a', 'b')]).half().cuda()
        al = model.get_tensor(prefix + 'A_log').float().cuda()
        dt = model.get_tensor(prefix + 'dt_bias').float().cuda()
    for dtype in ('float16', 'float32'):
        packed = gdn_gates(dtype=dtype, packed_ab=True)
        separate = gdn_gates(dtype=dtype)
        for name, kernel in (('packed', packed), ('separate', separate)):
            dest = out / (dtype + '-' + name)
            exported = export_kernel(kernel, dest)
            report['exports'].append({'name': dest.name, 'artifacts': exported,
                'abi': parse_host((dest / 'host.txt').read_text())})
        for rows in (1, 3, 8, 513):
            ab = torch.randn((rows, 96), device='cuda', dtype=getattr(torch, dtype))
            a, b = ab[:, :48].contiguous(), ab[:, 48:].contiguous()
            g, beta = [torch.empty((rows, 48), device='cuda') for _ in range(2)]
            rg, rb = torch.empty_like(g), torch.empty_like(beta)
            def run():
                launch(packed, ab, ab, al, dt, g, beta,
                       stream=torch.cuda.current_stream().cuda_stream)
            launch(separate, a, b, al, dt, rg, rb,
                   stream=torch.cuda.current_stream().cuda_stream)
            run()
            assert torch.equal(g, rg) and torch.equal(beta, rb)
            refg = -al.exp() * torch.nn.functional.softplus(ab[:, :48].float() + dt)
            refb = torch.sigmoid(ab[:, 48:].float())
            e = [error(g, refg), error(beta, refb)]
            assert all(item['finite'] and item['relative_l2'] < .0001 for item in e)
            timing, graph = benchmark(run, repetitions=5, calls_per_replay=16)
            saved = ab.clone()
            ab.mul_(.5).add_(.25)
            g.fill_(float('nan')); beta.fill_(float('nan'))
            graph.replay(); torch.cuda.synchronize()
            changed = [error(g, -al.exp() * torch.nn.functional.softplus(ab[:, :48].float() + dt)),
                       error(beta, torch.sigmoid(ab[:, 48:].float()))]
            assert all(item['finite'] and item['relative_l2'] < .0001 for item in changed)
            ab.copy_(saved); g.fill_(float('nan')); beta.fill_(float('nan'))
            graph.replay(); torch.cuda.synchronize()
            assert torch.equal(g, rg) and torch.equal(beta, rb)
            report['cases'].append({'dtype': dtype, 'rows': rows, 'packed_equals_separate': True,
                'math_errors': e, 'graph_changed_errors': changed, 'graph_restored': True, 'timing': timing})
    for rows in (1, 8, 512):
        cpu, origin = input_rows(0, rows)
        x = cpu.cuda(); ab = torch.empty((rows, 96), device='cuda')
        project = (gdn_ab_simt if rows <= 8 else gdn_ab_tensorcore)(
            T.dynamic('M'), output_dtype='float32')
        gates = gdn_gates(dtype='float32', packed_ab=True)
        g, beta = [torch.empty((rows, 48), device='cuda') for _ in range(2)]
        def run_chain():
            stream = torch.cuda.current_stream().cuda_stream
            project(x, weight, ab, stream=stream)
            launch(gates, ab, ab, al, dt, g, beta, stream=stream)
        run_chain()
        ref = x.float() @ weight.float().T
        e = [error(g, -al.exp() * torch.nn.functional.softplus(ref[:, :48] + dt)),
             error(beta, torch.sigmoid(ref[:, 48:]))]
        assert all(item['finite'] and item['relative_l2'] < .0001 for item in e)
        timing, _ = benchmark(run_chain, repetitions=5, calls_per_replay=4)
        report['cases'].append({'scope': 'real layer0 projection -> packed gates', 'rows': rows,
                               'input': origin, 'errors': e, 'timing': timing})
    report['status'] = 'passed'
    write_json(out / 'results.json', report)


if __name__ == '__main__':
    main()
