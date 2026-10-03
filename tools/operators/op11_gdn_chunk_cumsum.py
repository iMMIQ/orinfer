"""Offline op11 reference, ragged/graph verification, ABI export and timing."""
import argparse
import json
import time
from pathlib import Path

import torch

from abi import parse_host
from common import benchmark, configure, environment, error, export_kernel, identity, write_json
from kernels.operators.op11_gdn_chunk_cumsum import (
    HEADS, gdn_chunk_cumsum, gdn_pack_chunk_cumsum, launch, launch_pack, validate_lengths)

ROOT = Path(__file__).resolve().parents[2]


def reference(g, beta, lengths, bt):
    b, tokens, _ = g.shape
    c = (tokens + bt - 1) // bt
    mask = torch.arange(tokens, device=g.device)[None, :, None] < lengths[:, None, None]
    clean_g = torch.where(mask, g, 0.)
    clean_beta = torch.where(mask, beta.float(), 0.)
    pad = c * bt - tokens
    clean_g = torch.nn.functional.pad(clean_g, (0, 0, 0, pad))
    clean_beta = torch.nn.functional.pad(clean_beta, (0, 0, 0, pad))
    head_g = clean_g.permute(0, 2, 1).contiguous().reshape(b, HEADS, c, bt)
    head_beta = clean_beta.permute(0, 2, 1).contiguous().reshape(b, HEADS, c, bt)
    return head_g.cumsum(-1), head_beta, head_g


def check(got_g, got_beta, expected):
    metrics = {'G': error(got_g, expected[0]), 'beta': error(got_beta, expected[1]),
               'nonpositive': bool((got_g <= 0).all()),
               'monotone_within_chunk': bool((got_g[..., 1:] <= got_g[..., :-1]).all()),
               'beta_bitwise': bool(torch.equal(got_beta, expected[1]))}
    assert torch.allclose(got_g, expected[0], atol=2e-5, rtol=3e-6), metrics
    assert metrics['G']['finite'] and metrics['beta']['finite'], metrics
    assert metrics['nonpositive'] and metrics['monotone_within_chunk'] and metrics['beta_bitwise'], metrics
    return metrics


def host_checks():
    assert validate_lengths([0, 1, 64], batch=3, tokens=64) == (0, 1, 64)
    bad = [([-1], 1, 64), ([65], 1, 64), ([1.5], 1, 64), ([True], 1, 64),
           ([1, 2], 1, 64), ([0], 1, 0), ([0], 1, 2**31), ([0], 2**31, 64)]
    for lengths, batch, tokens in bad:
        try:
            validate_lengths(lengths, batch=batch, tokens=tokens)
        except ValueError:
            continue
        raise AssertionError('invalid host lengths accepted')
    return {'valid_case': True, 'invalid_rejected_cases': len(bad)}


def check_tail(gc, lens, bt):
    for b, length in enumerate(lens):
        for c in range(gc.shape[2]):
            valid = max(0, min(bt, length-c*bt))
            if valid == 0:
                assert bool((gc[b, :, c] == 0).all())
            elif valid < bt:
                assert torch.equal(gc[b, :, c, valid:],
                                   gc[b, :, c, valid-1:valid].expand(-1, bt-valid))
    return {'tail_exact_last_valid': True, 'all_invalid_chunks_exact_zero': True}


def export(kernel, dest, mode, bt, tile, beta_dtype, env):
    artifacts = export_kernel(kernel, dest)
    actual_abi = parse_host((dest / 'host.txt').read_text())
    write_json(dest / 'abi.json', {
        'operator': 'op11_gdn_chunk_cumsum', 'mode': mode, 'sm': 87,
        'BT': bt, 'heads': HEADS, 'heads_tile': tile, 'actual_generated_launches': actual_abi,
        'logical_parameters': (['G_fp32[B,T,48]', f'Beta_{beta_dtype}[B,T,48]', 'Lengths_i32[B]',
                                'CumulativeG_fp32[B,48,ceildiv(T,BT),BT]', 'PaddedBeta_fp32[same]']
                               if mode == 'pack' else ['G_fp32[B,48,C,BT]', 'CumulativeG_fp32[same]']),
        'layout': 'contiguous row-major', 'cooperative_launch': False,
        'workspace_bytes': 0, 'parameter_bytes': 0, 'input_output_alias': 'all allocations disjoint',
        'stream': 'explicit caller stream, read current stream on every runner invocation',
        'length_policy': 'host reject invalid length before upload; GPU defensive clamp to [0,T]',
        'state_policy': 'stateless, every chunk starts at zero, no accumulation across requests or chunks',
        'rounding': 'FP32 warp inclusive scan; operation order differs from Torch cumsum; beta FP16 input exactly promoted',
        'toolchain': env, 'artifacts': artifacts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--quick', action='store_true', help='compile/verify one production case only')
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    configure()
    env = environment()
    report = {'environment': env, 'host_checks': host_checks(), 'tuning': [], 'cases': [],
              'source': identity(ROOT / 'kernels/operators/op11_gdn_chunk_cumsum.py'),
              'workspace_bytes': 0, 'resident_parameter_bytes': 0,
              'budget_ms': {'B1_T512': .025, 'B1_T2048': .1, 'B1_T8192': .4},
              'reference': 'Torch FP32 cumsum only in offline runner; no model execution',
              'failures': []}
    kernels = {}
    for tile in ((4,) if args.quick else (4, 8, 16)):
        started = time.perf_counter()
        kernel = gdn_pack_chunk_cumsum(64, tile)
        kernels['pack', 64, tile, 'float32'] = kernel
        g = -torch.rand((1, 512, HEADS), device='cuda')
        beta = torch.rand_like(g)
        lens = torch.tensor([512], device='cuda', dtype=torch.int32)
        gc = torch.empty((1, HEADS, 8, 64), device='cuda')
        pb = torch.empty_like(gc)
        run = lambda: launch_pack(kernel, g, beta, lens, gc, pb,
                                 stream=torch.cuda.current_stream().cuda_stream)
        prep = time.perf_counter() - started
        started = time.perf_counter(); run(); torch.cuda.synchronize()
        first = 1000 * (time.perf_counter() - started)
        metrics = check(gc, pb, reference(g, beta, lens, 64))
        timing, graph = benchmark(run, repetitions=32, calls_per_replay=32)
        report['tuning'].append({'heads_tile': tile, 'prepare_s': prep, 'first_use_ms': first,
                                 'error': metrics, 'timing': timing})
        del graph
        write_json(out / 'progress.json', report)
    tile = min(report['tuning'], key=lambda t: t['timing']['median_ms'])['heads_tile']
    report['selected_heads_tile'] = tile
    specs = [(1, 512, 64, 'full', 'float32')]
    if not args.quick:
        specs = [(b, t, 64, mode, 'float32') for b in (1, 2, 3, 4, 5, 7, 8)
                 for t in (511, 512, 513, 2048, 8192) for mode in ('full', 'ragged')]
        specs += [(b, t, bt, 'boundary', dtype) for bt in (16, 32, 64)
                  for b in (1, 2, 3, 4, 5, 7, 8) for t in (bt-1, bt, bt+1, 2*bt+1)
                  for dtype in ('float32',)]
        specs += [(3, 513, bt, 'ragged', 'float16') for bt in (16, 32, 64)]
    for b, tokens, bt, mode, dtype in specs:
        key = 'pack', bt, tile, dtype
        if key not in kernels:
            started = time.perf_counter(); kernels[key] = gdn_pack_chunk_cumsum(bt, tile, dtype)
            report.setdefault('compilation', []).append({'key': list(key), 'prepare_s': time.perf_counter()-started})
        kernel = kernels[key]
        lengths = ([tokens] * b if mode == 'full' else
                   [min(tokens, x) for x in ([0, 1, bt-1, bt, bt+1, tokens-1, tokens] * 2)[:b]])
        lengths = validate_lengths(lengths, batch=b, tokens=tokens)
        lens = torch.tensor(lengths, device='cuda', dtype=torch.int32)
        g = -torch.rand((b, tokens, HEADS), device='cuda') * 2
        # Unique scales make every request/head distinguishable.
        g *= (torch.arange(b, device='cuda')[:, None, None] + 1.)
        g *= (torch.arange(HEADS, device='cuda')[None, None, :] + 1.) / HEADS
        raw_beta = torch.rand_like(g)
        beta = raw_beta.to(getattr(torch, dtype))
        upstream_beta_cast = error(beta.float(), raw_beta)
        mask = torch.arange(tokens, device='cuda')[None, :, None] >= lens[:, None, None]
        g.masked_fill_(mask, float('nan')); beta.masked_fill_(mask, float('nan'))
        original = (g.clone(), beta.clone(), lens.clone())
        c = (tokens + bt - 1) // bt
        gc = torch.empty((b, HEADS, c, bt), device='cuda'); pb = torch.empty_like(gc)
        expected = reference(g, beta, lens, bt)
        run = lambda: launch_pack(kernel, g, beta, lens, gc, pb,
                                 stream=torch.cuda.current_stream().cuda_stream)
        run(); torch.cuda.synchronize()
        case = {'B': b, 'T': tokens, 'BT': bt, 'mode': mode, 'beta_dtype': dtype,
                'lengths': list(lengths), 'error': check(gc, pb, expected),
                'tail': check_tail(gc, lengths, bt),
                'upstream_beta_cast_to_input_dtype': upstream_beta_cast,
                'input_bytes': g.numel()*4 + beta.numel()*beta.element_size() + b*4,
                'output_bytes': 8*gc.numel()}
        case['timing'], graph = benchmark(run, repetitions=20, calls_per_replay=16)
        # Mutate all three inputs and poison both outputs, replay, then restore.
        updated = validate_lengths([max(0, tokens-(i+1)) for i in range(b)], batch=b, tokens=tokens)
        lens.copy_(torch.tensor(updated, device='cuda', dtype=torch.int32))
        g.fill_(-.125); beta.fill_(.375)
        gc.fill_(float('nan')); pb.fill_(float('nan'))
        graph.replay(); torch.cuda.synchronize()
        case['graph_changed'] = check(gc, pb, reference(g, beta, lens, bt))
        g.copy_(original[0]); beta.copy_(original[1]); lens.copy_(original[2])
        gc.fill_(float('nan')); pb.fill_(float('nan'))
        graph.replay(); torch.cuda.synchronize()
        case['graph_restored'] = check(gc, pb, expected)
        del graph
        # Headmajor entry checks direct no-pack path against the same reference.
        hmkey = 'headmajor', bt, tile, 'float32'
        if hmkey not in kernels:
            kernels[hmkey] = gdn_chunk_cumsum(bt, tile)
        head = expected[2]; head_gc = torch.empty_like(head)
        run_head = lambda: launch(kernels[hmkey], head, head_gc,
                                  stream=torch.cuda.current_stream().cuda_stream)
        run_head(); torch.cuda.synchronize()
        case['headmajor'] = check(head_gc, pb, expected)
        if b == 1 and tokens in (511, 512, 513, 2048, 8192) and mode == 'full':
            case['headmajor_timing'], graph = benchmark(run_head, repetitions=32, calls_per_replay=32)
            old_head = head.clone()
            head.fill_(-.25); head_gc.fill_(float('nan'))
            graph.replay(); torch.cuda.synchronize()
            assert torch.allclose(head_gc, head.cumsum(-1), atol=2e-5, rtol=3e-6)
            head.copy_(old_head); head_gc.fill_(float('nan'))
            graph.replay(); torch.cuda.synchronize()
            assert torch.allclose(head_gc, expected[0], atol=2e-5, rtol=3e-6)
            case['headmajor_graph_changed_restored'] = True
            del graph
            case['single_node_timing'], graph = benchmark(run, repetitions=100, calls_per_replay=1)
            del graph
        # Resume on a chunk boundary: direct slice views with contiguous per-request
        # storage; compare each independently launched chunk, no state carry.
        if b == 1 and tokens == 513 and bt == 64 and mode == 'full':
            pieces = []
            for chunk in range(c):
                start, stop = chunk*bt, min(tokens, (chunk+1)*bt)
                gg, bb = g[:, start:stop], beta[:, start:stop]
                ll = torch.tensor([stop-start], device='cuda', dtype=torch.int32)
                one_gc = torch.empty((1, HEADS, 1, bt), device='cuda'); one_pb = torch.empty_like(one_gc)
                launch_pack(kernel, gg, bb, ll, one_gc, one_pb, stream=torch.cuda.current_stream().cuda_stream)
                pieces.append(one_gc)
            torch.cuda.synchronize()
            case['chunk_boundary_resume'] = check(torch.cat(pieces, dim=2), pb, expected)
        report['cases'].append(case)
        write_json(out / 'progress.json', report)
    for (mode, bt, ht, dtype), kernel in kernels.items():
        if ht == tile:
            export(kernel, out / f'{mode}-bt{bt}-{dtype}', mode, bt, ht, dtype, env)
    report['native_precision'] = {
        'g': 'op09 and native log-decay are FP32; op11 does not recompute the gate function',
        'beta': 'op09 FP32 output is retained by default. Explicit float16 input exactly promotes already-rounded native beta; difference is upstream gate rounding.',
        'scan': 'FP32 warp prefix scan reorders additions vs Torch FP32 cumsum; no extra FP16 g rounding'}
    report['peak_torch_validation_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['status'] = 'passed'
    write_json(out / 'results.json', report)
    print(json.dumps({'status': 'passed', 'cases': len(report['cases']), 'selected_heads_tile': tile,
                      'tuning': report['tuning']}, indent=2))


if __name__ == '__main__':
    main()
