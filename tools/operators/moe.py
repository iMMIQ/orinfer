"""Independent numerical and graph checks for MoE routing and reduction."""
import argparse
from pathlib import Path

import torch

from kernels.model.moe import router_topk, moe_combine
from tools.operators.common import configure, benchmark, error, export_kernel, environment, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure()
    report = {'environment': environment(), 'cases': [], 'complete': False}
    for m, e, k in ((1, 512, 10), (8, 512, 10), (17, 37, 10), (512, 512, 10), (3, 5, 5)):
        logits = torch.randn((m, e), device='cuda', dtype=torch.float32) * 5
        logits[0] = 0  # Exact ties must choose lower expert IDs.
        if m > 1:
            logits[1] = -10000
            logits[1, -1] = 10000
        ids = torch.empty((m, k), device='cuda', dtype=torch.int32)
        prob = torch.empty((m, k), device='cuda')
        for renormalize in (False, True):
            kernel = router_topk(m, e, k, renormalize)

            def run():
                kernel(logits, ids, prob)

            def reference():
                expected_ids = torch.argsort(logits, descending=True, stable=True)[:, :k]
                # Selection by logits avoids softmax underflow collapsing ties.
                expected = torch.softmax(logits, dim=-1).gather(1, expected_ids)
                if renormalize:
                    expected /= expected.sum(-1, keepdim=True)
                return expected_ids, expected

            run()
            expected_ids, expected = reference()
            assert torch.equal(ids.long(), expected_ids)
            metric = error(prob, expected)
            assert metric['finite'] and metric['max_abs'] < 2e-6, metric
            timing, graph = benchmark(run, repetitions=8)
            original = logits.clone()
            prob.fill_(float('nan'))
            ids.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(ids.long(), expected_ids)
            torch.testing.assert_close(prob, expected, atol=2e-6, rtol=2e-5)
            logits.copy_(torch.randn_like(logits))
            graph.replay()
            torch.cuda.synchronize()
            expected_ids, expected = reference()
            assert torch.equal(ids.long(), expected_ids)
            torch.testing.assert_close(prob, expected, atol=2e-6, rtol=2e-5)
            logits.copy_(original)
            graph.replay()
            torch.cuda.synchronize()
            expected_ids, expected = reference()
            assert torch.equal(ids.long(), expected_ids)
            export_kernel(kernel, args.output / f'router-M{m}-E{e}-norm{int(renormalize)}')
            report['cases'].append({'kind': 'router', 'rows': m, 'experts': e, 'top_k': k,
                                    'renormalize': renormalize, 'timing': timing, 'error': metric})
            write_json(args.output / 'results.json', report)
            print('router', m, e, renormalize, timing['median_ms'], flush=True)

    for m, h, slots in ((1, 2560, 10), (8, 2560, 80), (17, 257, 37), (512, 2560, 5120)):
        k = 10
        experts = torch.randn((slots, h), device='cuda', dtype=torch.float16)
        slot_map = torch.randint(0, slots, (m, k), device='cuda', dtype=torch.int32)
        slot_map[0, :2] = torch.tensor([-1, slots], device='cuda')
        prob = torch.softmax(torch.randn((m, k), device='cuda'), dim=-1)
        shared = torch.randn((m, h), device='cuda', dtype=torch.float16)
        gate = torch.randn(m, device='cuda', dtype=torch.float16)
        out = torch.empty_like(shared)
        kernel = moe_combine(m, h, slots, k)

        def run():
            kernel(experts, slot_map, prob, shared, gate, out)

        def reference():
            values = experts[slot_map.clamp(0, slots-1).long()].float()
            values[(slot_map < 0) | (slot_map >= slots)] = 0
            return ((values * prob[..., None]).sum(1) +
                    torch.sigmoid(gate.float())[:, None] * shared.float()).half()

        run()
        expected = reference()
        metric = error(out, expected)
        assert metric['finite'] and metric['relative_l2'] < .001, metric
        timing, graph = benchmark(run, repetitions=8)
        saved = out.clone()
        out.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, saved)
        original_map = slot_map.clone()
        slot_map.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, reference(), atol=.001, rtol=.001)
        slot_map.copy_(original_map)
        original_gate = gate.clone()
        gate.fill_(10)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, reference(), atol=.001, rtol=.001)
        gate.copy_(original_gate)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, saved)
        export_kernel(kernel, args.output / f'combine-M{m}-H{h}')
        report['cases'].append({'kind': 'combine', 'rows': m, 'hidden': h,
                                'slots': slots, 'timing': timing, 'error': metric})
        write_json(args.output / 'results.json', report)
        print('combine', m, h, timing['median_ms'], flush=True)
    report['complete'] = True
    write_json(args.output / 'results.json', report)


if __name__ == '__main__':
    main()
