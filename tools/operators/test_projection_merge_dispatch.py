"""Public down/mixer plans execute the delivered op32 reducer, not legacy reduce."""
import argparse
import shutil
from pathlib import Path
import torch
import tilelang.language as T
from common import ROOT, benchmark, configure, environment, error, export_kernel, identity, write_json
from abi import parse_host
from tools.projections.decode_common import raw_weights, logical, dequant, inputs
from kernels.operators.op05_ffn_down import build_ffn_down
from kernels.operators.op18_mixer_out import build_mixer_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--repetitions', type=int, default=15)
    args = parser.parse_args()
    configure()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    deps = [ROOT / p for p in ('kernels/operators/op05_ffn_down.py',
        'kernels/operators/op18_mixer_out.py', 'kernels/operators/op32_split_k_merge.py',
        'kernels/operators/op03_ffn_gate_up.py', 'kernels/projections/candidates.py',
        'tools/projections/decode_common.py', 'tools/operators/common.py')]
    frozen = [identity(p) for p in deps]
    for src in deps:
        dest = out / 'measurement-source' / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    result = {'status': 'in_progress', 'environment': environment(), 'sources': frozen,
              'cases': [], 'exports': [], 'scope': 'Real layer0 projection inputs; no model/concurrency verdict'}
    for case, build in (('down', build_ffn_down), ('gdn_out', build_mixer_out)):
        raw, names, _ = raw_weights(case)
        pc, sc, zc, qcodes, _ = logical(raw)
        p, s, z = pc.cuda(), sc.cuda(), zc.cuda()
        w = dequant(qcodes, sc, zc)
        plan = build(T.dynamic('M'))
        assert plan.route == 'splitk_register'
        for name, kernel in (('projection', plan.projection), ('merge', plan.merge)):
            dest = out / 'aot' / case / name
            files = export_kernel(kernel, dest)
            result['exports'].append({'case': case, 'stage': name, 'files': files,
                                     'actual_abi': parse_host((dest / 'host.txt').read_text())})
        for m, cpu_x, provenance in inputs(case):
            if m not in (1, 3):
                continue
            x = cpu_x.cuda()
            y = torch.empty((m, 5120), device='cuda', dtype=torch.float16)
            partial = torch.empty(plan.workspace_shape(m), device='cuda')
            def run():
                plan(x, p, s, z, y, partial, stream=torch.cuda.current_stream().cuda_stream)
            def check():
                observed = error(y, x.float() @ w.float().T)
                assert observed['finite'] and observed['relative_l2'] < .002, observed
                return observed
            run()
            torch.cuda.synchronize()
            observed = check()
            saved_x, saved_y = x.clone(), y.clone()
            hot, graph = benchmark(run, repetitions=args.repetitions)
            x.zero_()
            y.fill_(float('nan'))
            partial.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.count_nonzero(y) == 0
            x.copy_(saved_x)
            y.fill_(float('nan'))
            partial.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(y, saved_y)
            result['cases'].append({'case': case, 'M': m, 'error': observed, 'hot': hot,
                'activation': provenance, 'weights': names, 'graph_zero_and_restore_exact': True,
                'partial_bytes': partial.numel() * 4, 'selected_public_route': plan.route})
            write_json(out / 'results.json', result)
            print(f'public {case} M{m}: {hot["median_ms"]:.6f} ms, L2 {observed["relative_l2"]:.6g}', flush=True)
    assert frozen == [identity(p) for p in deps], 'Sources changed during test'
    result['status'] = 'passed'
    write_json(out / 'results.json', result)


if __name__ == '__main__':
    main()
