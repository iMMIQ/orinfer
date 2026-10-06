"""Check complete gated residual reads/writes against explicit Torch semantics."""
import argparse
import gc
from pathlib import Path

import torch

from kernels.model.hyperconnection import hc_norm, hc_silu, hc_mix, hc_combine, hc_projection
from tools.operators.common import configure, benchmark, error, export_kernel, environment, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only-first', action='store_true')
    args = parser.parse_args()
    configure()
    report = {'environment': environment(), 'cases': [], 'complete': False,
              'scope': 'Complete HC read and residual write, synthetic weights/activations; '
                       'no full-model quality/TPS.'}
    for dtype_name in ('bfloat16', 'float16'):
        dtype = getattr(torch, dtype_name)
        for m, h, rank in ((1, 2560, 320), (3, 2560, 320), (17, 64, 64), (512, 2560, 320)):
            c = 4
            residual = torch.randn((m, c, h), device='cuda', dtype=dtype)
            residual[0, 0] = 0
            weight = torch.randn((c, h), device='cuda') * .1
            wd = (torch.randn((rank, c*h), device='cuda') / (c*h)**.5).bfloat16()
            wu = (torch.randn((c*h, rank), device='cuda') / rank**.5).bfloat16()
            wi = (torch.randn((c, c*h), device='cuda') / (c*h)**.5).bfloat16()
            normed = torch.empty_like(residual)
            down = torch.empty((m, rank), device='cuda', dtype=dtype)
            activated = torch.empty_like(down)
            up = torch.empty_like(residual)
            mixed = torch.empty((m, h), device='cuda', dtype=dtype)
            block = torch.randn_like(mixed)
            inject = torch.empty((m, c), device='cuda', dtype=dtype)
            out = torch.empty_like(residual)
            bm = 64 if m >= 64 else 16
            norm = hc_norm(m, h, c, dtype=dtype_name)
            project_down = hc_projection(m, rank, c*h, bm, dtype_name)
            silu = hc_silu(m, rank, c, dtype_name)
            project_up = hc_projection(m, c*h, rank, bm, dtype_name)
            mix = hc_mix(m, h, c, dtype_name)
            project_inject = hc_projection(m, c, c*h, bm, dtype_name)
            combine = hc_combine(m, h, c, dtype_name)
            if args.only_first:
                for name, kernel in [('norm', norm), ('down', project_down), ('silu', silu),
                                     ('up', project_up), ('mix', mix), ('inject', project_inject),
                                     ('combine', combine)]:
                    export_kernel(kernel, args.output / name)

            debug_sync = args.only_first

            def sync_stage(name):
                if debug_sync:
                    torch.cuda.synchronize()
                    print(f'HC {name} complete', flush=True)

            def run():
                norm(residual, weight, normed)
                sync_stage('norm')
                project_down(normed.view(m, c*h), wd, down)
                sync_stage('down')
                silu(down, activated)
                sync_stage('silu')
                project_up(activated, wu, up.view(m, c*h))
                sync_stage('up')
                mix(normed, up, mixed)
                sync_stage('mix')
                project_inject(normed.view(m, c*h), wi, inject)
                sync_stage('inject')
                combine(block, residual, inject, out)
                sync_stage('combine')

            def validate():
                xf = residual.float()
                expected_norm = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True)+1e-6) *
                                 (1+weight)).to(dtype)
                # Both tensor-core operands are BF16, including optional FP16 streams.
                expected_down = (normed.flatten(1).bfloat16().float() @ wd.float().T).to(dtype)
                expected_act = torch.nn.functional.silu((down/c).to(dtype).float()).to(dtype)
                expected_up = (activated.bfloat16().float() @ wu.float().T).to(dtype).reshape(m,c,h)
                expected_mix = (torch.sigmoid(up.float()).to(dtype) * normed).mean(1)
                expected_inject = (normed.flatten(1).bfloat16().float() @ wi.float().T).to(dtype)
                gate = (2*torch.sigmoid((inject/c).to(dtype).float()).to(dtype)).to(dtype)
                expected_out = residual + (block[:,None,:]*gate[...,None]).to(dtype)
                pairs = [('norm', normed, expected_norm), ('down', down, expected_down),
                         ('silu', activated, expected_act), ('up', up, expected_up),
                         ('mix', mixed, expected_mix), ('inject', inject, expected_inject),
                         ('combine', out, expected_out)]
                metrics = {name: error(actual, expected) for name, actual, expected in pairs}
                for name, metric in metrics.items():
                    assert metric['finite'] and metric['relative_l2'] < .004, (name,metric)
                return metrics

            run()
            torch.cuda.synchronize()
            debug_sync = False
            metrics = validate()
            timing, graph = benchmark(run, repetitions=8)
            saved_mix, saved_out = mixed.clone(), out.clone()
            normed.fill_(float('nan'))
            mixed.fill_(float('nan'))
            out.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(mixed, saved_mix) and torch.equal(out, saved_out)
            original = residual.clone()
            residual.copy_(torch.randn_like(residual))
            graph.replay()
            torch.cuda.synchronize()
            changed = validate()
            assert not torch.equal(out, saved_out)
            residual.copy_(original)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(mixed, saved_mix) and torch.equal(out, saved_out)
            # In-place residual writes retain one owner per element.
            combine(block, residual, inject, residual)
            torch.cuda.synchronize()
            assert torch.equal(residual, saved_out)
            residual.copy_(original)
            folder = args.output / f'{dtype_name}-M{m}-H{h}'
            for name, kernel in [('norm',norm), ('down',project_down), ('silu',silu),
                                 ('up',project_up), ('mix',mix), ('inject',project_inject),
                                 ('combine',combine)]:
                export_kernel(kernel, folder/name)
            report['cases'].append({'dtype':dtype_name, 'rows':m, 'hidden':h, 'rank':rank,
                                    'timing':timing,'errors':metrics,'changed_errors':changed,
                                    'graph_checks':['poison','changed_input','restore'],
                                    'inplace_combine_verified':True})
            write_json(args.output / 'results.json',report)
            print(dtype_name,m,h,timing['median_ms'],flush=True)
            del graph,saved_mix,saved_out,original
            gc.collect()
            if args.only_first:
                report['complete'] = True
                write_json(args.output / 'results.json', report)
                return
    report['complete']=True
    write_json(args.output / 'results.json',report)


if __name__ == '__main__':
    main()
