"""Check GPU expert dispatch and a native grouped Q2A8 routed FFN chain.

This uses synthetic activations/router logits. It is not a model quality/TPS
test. Optional checkpoint weights are the actual complete layer-0 expert bank.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from kernels.model.moe import (
    router_topk, expert_histogram, expert_offsets, expert_tiles, expert_dispatch, moe_combine,
)
from kernels.model.q2a8 import q2a8_grouped, q2a8_repack
from kernels.operators.op30_activation_quantization import (
    activation_quantization, swiglu_activation_quantization, launch,
)
from tools.operators.q2a8 import synthesize, decode, reference_quant
from tools.operators.common import configure, benchmark, error, export_kernel, environment, write_json


def checkpoint_bank(directory, partial):
    """Offline probe only: a partial shard must have durable verified chunks.

    Full original-source SHA verification belongs to the downloader. Until it
    completes, report the bank as a range-verified experimental input.
    """
    names = [f'blk.0.ffn_{name}_exps.weight' for name in ('gate', 'up', 'down')]
    path = directory / 'model-00001-of-00002.safetensors'
    if not path.exists():
        if not partial:
            raise ValueError('Complete checkpoint shard required')
        path = directory / 'model-00001-of-00002.part'
        state = json.loads((directory / 'model-00001-of-00002.safetensors.download.json').read_text())
        with path.open('rb') as reader:
            length = int.from_bytes(reader.read(8), 'little')
            header = json.loads(reader.read(length))
            prefix = 8 + length
            for name in names:
                begin, end = header[name]['data_offsets']
                for chunk in range(begin // (32*1024**2), (end-1) // (32*1024**2) + 1):
                    expected = state['completed'].get(str(chunk))
                    if expected is None:
                        raise ValueError(f'Expert bank chunk {chunk} is still downloading')
                    reader.seek(prefix + chunk * 32*1024**2)
                    data = reader.read(min(32*1024**2, path.stat().st_size-prefix-chunk*32*1024**2))
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError(f'Expert bank chunk {chunk} hash mismatch')
    with safe_open(path, framework='np') as reader:
        values = [reader.get_tensor(name) for name in names]
    sources = [{'file': str(path), 'tensor': name, 'bytes': x.nbytes,
                'sha256': hashlib.sha256(memoryview(x)).hexdigest(),
                'full_shard_verified': path.suffix == '.safetensors'} for name, x in zip(names, values)]
    return values, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--allow-verified-partial', action='store_true')
    parser.add_argument('--implementation', choices=('shared', 'register'), default='shared')
    parser.add_argument('--layout', choices=('row_major', 'group_major'), default='row_major')
    args = parser.parse_args()
    configure()
    report = {'environment': environment(), 'cases': [], 'sources': [], 'complete': False,
              'scope': 'Native routed/grouped FFN, synthetic activations/router logits; '
                       'zero shared expert. No complete-model quality/TPS.', 'global_w8_bytes': 0,
              'implementation': args.implementation, 'layout': args.layout}
    banks = [('synthetic', (synthesize(37, 64, 128), synthesize(37, 64, 128),
                             synthesize(37, 128, 64)), (1, 3, 17, 512))]
    if args.checkpoint_dir:
        values, sources = checkpoint_bank(args.checkpoint_dir, args.allow_verified_partial)
        report['sources'] = sources
        banks.append(('real_bank512', values, (1, 8, 512)))
    for label, (gp, up, dp), sizes in banks:
        b, f, packed_k = gp.shape
        h = packed_k // 18 * 64
        assert up.shape == gp.shape and dp.shape == (b, h, f//64*18)
        def storage(x):
            if args.layout == 'row_major':
                return torch.from_numpy(x).cuda()
            e, n, pk = x.shape
            blocks = x.reshape(e, n, pk//18, 18)
            source = torch.from_numpy(x).cuda()
            result = torch.empty((e,pk//18,n,18), device='cuda', dtype=torch.uint8)
            repack = q2a8_repack(e,n,pk//18*64)
            repack(source,result)
            assert np.array_equal(result.cpu().numpy().transpose(0,2,1,3),blocks)
            # Actual graph replay must respond to a source change, then restore.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                repack(source,result)
            source.bitwise_xor_(255)
            graph.replay()
            assert np.array_equal(result.cpu().numpy().transpose(0,2,1,3),np.bitwise_xor(blocks,255))
            source.bitwise_xor_(255)
            graph.replay()
            assert np.array_equal(result.cpu().numpy().transpose(0,2,1,3),blocks)
            export_kernel(repack,args.output/f'repack-{label}-N{n}-K{pk//18*64}')
            return result
        g, u, d = (storage(x) for x in (gp, up, dp))
        for m in sizes:
            k = 10
            assignments = m*k
            capacity = (assignments+15)//16 + min(b, assignments)
            x = (torch.randn((m, h), device='cuda') * .2).half()
            logits = torch.randn((m, b), device='cuda')
            ids = torch.empty((m, k), device='cuda', dtype=torch.int32)
            prob = torch.empty((m, k), device='cuda')
            counts = torch.empty(b, device='cuda', dtype=torch.int32)
            relative = torch.empty_like(ids)
            offsets = torch.empty_like(counts)
            tile_offsets = torch.empty_like(counts)
            tile_count = torch.empty(1, device='cuda', dtype=torch.int32)
            tile_expert = torch.empty(capacity, device='cuda', dtype=torch.int32)
            tile_row = torch.empty_like(tile_expert)
            aq = torch.empty_like(x, dtype=torch.int8)
            sa = torch.empty((m, h//64), device='cuda', dtype=torch.float16)
            dispatch = torch.empty((assignments, h), device='cuda', dtype=torch.int8)
            ds = torch.empty((assignments, h//64), device='cuda', dtype=torch.float16)
            slot_map = torch.empty_like(ids)
            gu = torch.empty((assignments, 2*f), device='cuda', dtype=torch.float16)
            fq = torch.empty((assignments, f), device='cuda', dtype=torch.int8)
            fs = torch.empty((assignments, f//64), device='cuda', dtype=torch.float16)
            y = torch.empty((assignments, h), device='cuda', dtype=torch.float16)
            shared = torch.zeros_like(x)
            gate = torch.zeros(m, device='cuda', dtype=torch.float16)
            out = torch.empty_like(x)
            mask_a = torch.zeros(h, device='cuda', dtype=torch.uint8)
            mask_f = torch.zeros(f, device='cuda', dtype=torch.uint8)
            route = router_topk(m, b, k)
            hist = expert_histogram(m, b, k)
            prefix = expert_offsets(b)
            tiles = expert_tiles(m, b, k)
            scatter = expert_dispatch(m, h, b, k)
            quant = activation_quantization(h, 64, threads=128)
            gate_up = q2a8_grouped(assignments, b, capacity, 2*f, h, True, args.implementation, args.layout)
            fused = swiglu_activation_quantization(f, 64, threads=128)
            down = q2a8_grouped(assignments, b, capacity, h, f, implementation=args.implementation, layout=args.layout)
            combine = moe_combine(m, h, assignments, k)

            def run():
                stream = torch.cuda.current_stream().cuda_stream
                route(logits, ids, prob)
                hist(ids, counts, relative)
                prefix(counts, offsets, tile_offsets, tile_count)
                tiles(counts, tile_offsets, tile_expert, tile_row)
                launch(quant, x, mask_a, aq, sa, stream=stream)
                scatter(aq, sa, ids, relative, offsets, dispatch, ds, slot_map)
                gate_up(dispatch, g, u, ds, counts, offsets, tile_expert, tile_row, tile_count, gu)
                launch(fused, gu, mask_f, fq, fs, stream=stream)
                down(fq, d, d, fs, counts, offsets, tile_expert, tile_row, tile_count, y)
                combine(y, slot_map, prob, shared, gate, out)

            def validate():
                expected_ids = torch.argsort(logits, descending=True, stable=True)[:, :k]
                assert torch.equal(ids.long(), expected_ids)
                expected_counts = torch.bincount(ids.flatten().long(), minlength=b).int()
                assert torch.equal(counts, expected_counts)
                expected_offsets = expected_counts.cumsum(0).int() - expected_counts
                assert torch.equal(offsets, expected_offsets)
                assert int(tile_count.item()) == int(((expected_counts+15)//16).sum())
                assert int(tile_count.item()) <= capacity
                assert torch.equal(torch.sort(slot_map.flatten()).values,
                                   torch.arange(assignments, device='cuda', dtype=torch.int32))
                # Every compact row must retain its original activation/scales.
                assert torch.equal(dispatch[slot_map.long()], aq[:, None, :].expand(m, k, h))
                assert torch.equal(ds[slot_map.long()], sa[:, None, :].expand(m, k, h//64))
                expected_gu = torch.empty_like(gu)
                expected_y = torch.empty_like(y)
                for expert in range(b):
                    count, offset = int(counts[expert]), int(offsets[expert])
                    if not count:
                        continue
                    ar = (dispatch[offset:offset+count].float().reshape(count, h//64, 64) *
                          ds[offset:offset+count].float()[..., None]).reshape(count, h)
                    wg = torch.cat((decode(gp[expert:expert+1], h)[0],
                                    decode(up[expert:expert+1], h)[0]), dim=0)
                    expected_gu[offset:offset+count] = (ar @ wg.T).half()
                    fr = (fq[offset:offset+count].float().reshape(count, f//64, 64) *
                          fs[offset:offset+count].float()[..., None]).reshape(count, f)
                    wd = decode(dp[expert:expert+1], f)[0]
                    expected_y[offset:offset+count] = (fr @ wd.T).half()
                eg, ed = error(gu, expected_gu), error(y, expected_y)
                assert eg['finite'] and ed['finite'] and eg['relative_l2'] < .002 and ed['relative_l2'] < .002, (eg, ed)
                activated = (torch.nn.functional.silu(gu[:, :f].float()) * gu[:, f:].float()).half()
                _, _, ref = reference_quant(activated)
                actual = (fq.float().reshape(assignments, f//64, 64) * fs.float()[..., None]).reshape(assignments, f)
                ef = error(actual, ref)
                assert ef['relative_l2'] < .002, ef
                expected_out = (expected_y[slot_map.long()].float() * prob[..., None]).sum(1).half()
                ec = error(out, expected_out)
                assert ec['finite'] and ec['relative_l2'] < .002, ec
                return {'gate_up': eg, 'down': ed, 'fused_swiglu': ef, 'mixture': ec}

            run()
            torch.cuda.synchronize()
            metrics = validate()
            timing, graph = benchmark(run, repetitions=4)
            saved = out.clone()
            out.fill_(float('nan'))
            gu.fill_(float('nan'))
            y.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, saved)
            original_logits = logits.clone()
            # Strongly skewed routing switches most experts off, changes all tile maps.
            logits.fill_(-20)
            logits[:, :k] = torch.arange(k, device='cuda').float()
            graph.replay()
            torch.cuda.synchronize()
            changed = validate()
            assert not torch.equal(out, saved)
            skewed_timing, _ = benchmark(run, repetitions=4)
            logits.copy_(original_logits)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, saved)
            original_x = x.clone()
            x.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert bool((out == 0).all())
            x.copy_(original_x)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, saved)
            folder = args.output / f'{label}-M{m}'
            for name, kernel in [('router', route), ('histogram', hist), ('offsets', prefix),
                                 ('tiles', tiles), ('dispatch', scatter), ('gate_up', gate_up),
                                 ('down', down), ('combine', combine)]:
                export_kernel(kernel, folder / name)
            report['cases'].append({'bank': label, 'rows': m, 'experts': b, 'timing': timing,
                                    'skewed_timing': skewed_timing, 'errors': metrics,
                                    'changed_routing_errors': changed,
                                    'graph_checks': ['poison', 'changed_routing', 'changed_activation', 'restore']})
            write_json(args.output / 'results.json', report)
            print(label, m, timing['median_ms'], skewed_timing['median_ms'], flush=True)
            del graph, saved
            gc.collect()
        del g, u, d
        gc.collect()
    report['complete'] = True
    write_json(args.output / 'results.json', report)


if __name__ == '__main__':
    main()
