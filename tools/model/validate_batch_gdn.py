"""Validate private-arena M1 mixers, padding and changed-table Graph replay."""
import argparse
import json
from pathlib import Path
import torch
from tools.operators.common import configure, benchmark, error
from kernels.model.gdn_batch import batch_gdn_conv, batch_gdn_recurrent
from kernels.model.gdn_recurrent_inplace import gdn_recurrent_inplace
from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--skip-benchmark', action='store_true', help='Validate memory/state without timing instrumented replays')
    a = p.parse_args()
    configure()
    # Create the Torch primary context before TileLang's driver ABI probe.
    context_owner = torch.empty(1, device='cuda')
    rows = []
    for batch in (2, 4, 8, 16, 32, 64, 128):
        conv, recurrent = batch_gdn_conv(batch), batch_gdn_recurrent(batch)
        reference_conv = gdn_conv_prep(B=1, tokens=1, tile_tokens=1)
        reference_gdn = gdn_recurrent_inplace(q_scale=128**-.5)
        state = torch.randn(batch,48,128,128,device='cuda')*.01
        initial = state.clone()
        reference = initial.clone()
        history = torch.randn(batch,3,10240,device='cuda',dtype=torch.float16)
        history_initial = history.clone()
        ref_history = history.clone()
        steps = torch.arange(batch,device='cuda',dtype=torch.int32) % 5
        pointers = torch.zeros(128,3,device='cuda',dtype=torch.uint64)
        def table(order):
            values = [[state[i].data_ptr(),history[i].data_ptr(),steps[i:].data_ptr()] for i in order]
            pointers.zero_()
            pointers[:len(order)].copy_(torch.tensor(values,device='cuda',dtype=torch.uint64))
        x = torch.randn(batch,10240,device='cuda',dtype=torch.float16)*.1
        w = torch.randn(10240,4,device='cuda',dtype=torch.float16)*.1
        q,k,v = [torch.empty(batch,h,128,device='cuda',dtype=torch.float16) for h in (16,16,48)]
        g = -torch.rand(batch,48,device='cuda')*.1
        beta = torch.rand(batch,48,device='cuda')
        out = torch.empty_like(v)
        qr,kr,vr = [torch.empty_like(t) for t in (q,k,v)]
        ho = torch.empty_like(history[:1]); po = torch.empty(1,device='cuda',dtype=torch.int32)
        length = torch.ones(1,device='cuda',dtype=torch.int32)
        ref_out = torch.empty_like(out)
        def run_reference(order):
            for lane,i in enumerate(order):
                reference_conv(x[lane:lane+1],w,ref_history[i:i+1],length,steps[i:i+1],
                               qr[lane:lane+1],kr[lane:lane+1],vr[lane:lane+1],ho,po)
                ref_history[i:i+1].copy_(ho)
                reference_gdn(qr[lane:lane+1],kr[lane:lane+1],vr[lane:lane+1],
                              g[lane:lane+1],beta[lane:lane+1],reference[i:i+1],ref_out[lane:lane+1])
        def run():
            conv(pointers,x,w,q,k,v)
            recurrent(pointers,q,k,v,g,beta,out)
        table(list(range(batch)))
        run(); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        metrics = []
        for order in (list(range(batch)), list(reversed(range(batch-1)))):
            state.copy_(initial); history.copy_(history_initial)
            reference.copy_(initial); ref_history.copy_(history_initial)
            x.normal_(0,.1); table(order)
            graph.replay(); run_reference(order); torch.cuda.synchronize()
            for actual,expected in [(q[:len(order)],qr[:len(order)]),(k[:len(order)],kr[:len(order)]),
                                    (v[:len(order)],vr[:len(order)]),(out[:len(order)],ref_out[:len(order)]),
                                    (state,reference),(history,ref_history)]:
                e=error(actual,expected); metrics.append(e)
                assert e['finite'] and e['max_abs'] < 2e-5, e
        # Includes identical reset copies on both sides; comparative only.
        table(list(range(batch)))
        def candidate():
            state.copy_(initial);history.copy_(history_initial);run()
        def old():
            reference.copy_(initial);ref_history.copy_(history_initial)
            run_reference(list(range(batch)))
        timings = {} if a.skip_benchmark else dict(candidate=benchmark(candidate)[0],reference=benchmark(old)[0])
        rows.append(dict(batch=batch,errors=metrics,**timings))
        print('validated',batch,flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'result.json').write_text(json.dumps(dict(seed=20261002,rows=rows,
        changed_table_graph_replay=True,padded_request_isolation=True),indent=2)+'\n')
    del context_owner


if __name__ == '__main__':
    main()
