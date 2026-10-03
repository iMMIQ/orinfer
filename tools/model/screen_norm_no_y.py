"""Remove unused normalized-half stores; retain exact FP16/A8/R boundaries."""
import argparse
import json
from pathlib import Path
import shutil

import torch
from common import benchmark, configure, environment, export_kernel, identity, write_json
from kernels.model.residual_norm_a8 import residual_norm_a8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--validation-only', action='store_true')
    args = ap.parse_args()
    configure()
    source = Path('kernels/model/residual_norm_a8.py')
    shutil.copyfile(source, args.output/source.name)
    report = dict(status='running', source=identity(source), environment=environment(), cases=[],
                  scope='Paired synthetic full norm/quantization; exact retained outputs, '
                        'not whole-model TPS; use only where Norm has no consumer')
    candidate = residual_norm_a8(write_normalized=False)
    baseline = residual_norm_a8()
    export_kernel(candidate, args.output/'no-y')
    export_kernel(baseline, args.output/'paired-current')
    for rows, mode in [(512,'random'),(2048,'random'),(8192,'random'),(513,'random'),
                       (3,'zero'),(3,'tiny'),(3,'cancellation'),(3,'large')]:
        x = torch.randn((rows,5120), device='cuda', dtype=torch.float16)
        r = torch.randn((rows,5120), device='cuda', dtype=torch.float32)*3
        w = torch.randn(5120, device='cuda', dtype=torch.float16)*.1
        if mode=='zero':
            x.zero_(); r.zero_()
        elif mode=='tiny':
            x.mul_(2**-20); r.mul_(2**-20)
        elif mode=='cancellation':
            r.copy_(-x.float()); r[:,0].add_(2**-20)
        elif mode=='large':
            x.mul_(1000); r.mul_(1000)
        untouched = torch.full_like(x, 73.)
        ro = torch.empty_like(r)
        q = torch.empty_like(x, dtype=torch.int8)
        scale = torch.empty(rows, device='cuda', dtype=torch.float16)
        ref_y = torch.empty_like(x)
        ref_ro, ref_q, ref_scale = torch.empty_like(ro), torch.empty_like(q), torch.empty_like(scale)
        def run():
            candidate.adapter.func(x,r,w,untouched,ro,q,scale,
                                   stream=torch.cuda.current_stream().cuda_stream)
        def paired_run():
            baseline.adapter.func(x,r,w,ref_y,ref_ro,ref_q,ref_scale,
                                  stream=torch.cuda.current_stream().cuda_stream)
        paired_run(); run(); torch.cuda.synchronize()
        assert all(torch.equal(a,b) for a,b in zip((ro,q,scale),(ref_ro,ref_q,ref_scale)))
        assert bool((untouched==73.).all())
        if args.validation_only:
            timing = paired_timing = None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
        else:
            timing,graph = benchmark(run,repetitions=8)
            paired_timing,paired_graph = benchmark(paired_run,repetitions=8)
            del paired_graph
        backup_x, backup_r = x.clone(), r.clone()
        saved = [t.clone() for t in (ro,q,scale)]
        x.zero_(); r.zero_(); ro.fill_(float('nan')); q.fill_(42); scale.zero_()
        graph.replay(); torch.cuda.synchronize()
        assert bool((ro==0).all() and (q==0).all() and (scale==1).all())
        assert bool((untouched==73.).all())
        x.copy_(backup_x); r.copy_(backup_r)
        graph.replay(); torch.cuda.synchronize()
        assert all(torch.equal(a,b) for a,b in zip((ro,q,scale),saved))
        rec = dict(rows=rows,mode=mode,retained_outputs_bitwise_equal=True,
                   normalized_output_untouched=True,graph_zero_restore=True,
                   timing=timing,paired_current_timing=paired_timing)
        report['cases'].append(rec)
        write_json(args.output/'result.json',report)
        print(json.dumps(rec),flush=True)
        del x,r,w,untouched,ro,q,scale,ref_y,ref_ro,ref_q,ref_scale,backup_x,backup_r,saved,graph
    report['status']='passed'
    write_json(args.output/'result.json',report)


if __name__=='__main__':
    main()
