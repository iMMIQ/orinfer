"""Offline synthetic legal-state reference/graph/ABI validation for op16."""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from abi import parse_host
from common import benchmark, configure, environment, error, export_kernel, identity, write_json
from gdn_reference import chunk_matrices, triangle_transform, wy, chunk_scan, chunk_output
from kernels.operators.op16_gdn_chunk_output import HK, HV, DK, DV, gdn_chunk_output, launch

ROOT = Path(__file__).resolve().parents[2]
SCALE = DK**-.5


def host_checks():
    try:
        gdn_chunk_output()
    except TypeError:
        pass
    else:
        raise AssertionError('q_scale must be required')
    invalid = ({'q_scale': 0}, {'q_scale': float('nan')}, {'q_scale': True},
               {'q_scale': SCALE, 'bt': 63}, {'q_scale': SCALE, 'q_dtype': 'bfloat16'},
               {'q_scale': SCALE, 'output_dtype': 'bfloat16'},
               {'q_scale': SCALE, 'output_layout': 'unknown'})
    for params in invalid:
        try:
            gdn_chunk_output(**params)
        except ValueError:
            continue
        raise AssertionError('invalid specialization accepted')
    return {'q_scale_required': True, 'invalid_rejected': len(invalid)}


def inputs(b, tokens, bt, dtype, mode, scaled):
    c = (tokens + bt - 1) // bt
    q = torch.nn.functional.normalize(torch.randn((b, HK, c, bt, DK), device='cuda'), dim=-1)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1).half()
    if scaled:
        q *= SCALE
    q = q.to(getattr(torch, dtype))
    v = torch.randn((b, HV, c, bt, DV), device='cuda') * .2
    g = -torch.rand((b, HV, c, bt), device='cuda') * .03
    beta = torch.rand_like(g)
    if mode == 'g0':
        g.zero_()
    elif mode == 'beta0':
        beta.zero_()
    elif mode == 'beta1':
        beta.fill_(1.)
    elif mode == 'strongdecay':
        g.fill_(-1000.)
    if tokens < c * bt:
        valid = tokens - (c - 1) * bt
        q[:, :, -1, valid:].zero_(); k[:, :, -1, valid:].zero_()
        v[:, :, -1, valid:].zero_(); g[:, :, -1, valid:].zero_(); beta[:, :, -1, valid:].zero_()
    gc = g.cumsum(-1)
    scale = 1. if scaled else SCALE
    system, qk = chunk_matrices(q, k, gc, beta, q_scale=scale)
    transform = triangle_transform(system).contiguous()
    w, u = wy(transform, k, v, gc, beta)
    # Nonzero request-private initial states exercise the between-chunk term.
    initial = torch.randn((b, HV, DK, DV), device='cuda') * .025
    states, r, _ = chunk_scan(k, gc, w, u, initial)
    return (q.contiguous(), gc.contiguous(), qk.contiguous(),
            states.contiguous(), r.contiguous())


def output_shape(tensors, layout):
    b, _, c, bt, _ = tensors[0].shape
    return (b, HV, c, bt, DV) if layout == 'headmajor' else (b, c * bt, HV, DV)


def headmajor(tensor, inputs, layout):
    if layout == 'headmajor':
        return tensor
    b, _, c, bt, _ = inputs[0].shape
    return tensor.view(b, c, bt, HV, DV).permute(0, 3, 1, 2, 4).contiguous()


def check(actual, expected, tensors, tokens, layout):
    actual = headmajor(actual, tensors, layout)
    metrics = {'FP32_mathematical': error(actual, expected),
               'same_final_output_cast': error(actual, expected.to(actual.dtype))}
    assert metrics['FP32_mathematical']['finite']
    assert metrics['FP32_mathematical']['relative_l2'] < .002, metrics
    tol = 2e-5 if actual.dtype == torch.float32 else 2e-4
    assert torch.allclose(actual, expected.to(actual.dtype), atol=tol, rtol=.002), metrics
    bt = tensors[0].shape[3]
    if tokens % bt:
        assert bool((actual[:, :, -1, tokens % bt:] == 0).all()), 'invalid rows must be exactly zero'
        metrics['invalid_rows_exact_zero'] = True
    return metrics


def fp64_subset(tensors, scale, actual, layout):
    q, g, qk, state, r = tensors
    heads = (2, 3, 47)
    q64 = q[0, [h // 3 for h in heads], :1].double()
    g64 = g[0, list(heads), :1].double()
    s64 = state[0, list(heads), :1].double()
    r64 = r[0, list(heads), :1].double()
    qk64 = qk[0, list(heads), :1].double()
    expected = ((q64 * scale) * g64.exp()[..., None]) @ s64 + qk64 @ r64
    subset = headmajor(actual, tensors, layout)[0, list(heads), :1]
    # common.error intentionally reports FP32 metrics; double discrepancy is
    # computed before casting, preserving the FP64 oracle comparison.
    delta = subset.double() - expected
    return {'heads': list(heads), 'relative_l2': float(delta.norm() / expected.norm().clamp_min(1e-30)),
            'max_abs': float(delta.abs().max()), 'reference_dtype': 'float64'}


def graph_checks(kernel, tensors, y, tokens, layout, scale, graph):
    originals = tuple(x.clone() for x in tensors)
    result = []
    for name, tensor, factor in zip(('Q', 'G', 'QK', 'Senter', 'R'), tensors, (.5, 1.3, -.7, .25, -.5)):
        tensor.mul_(factor); y.fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
        expected = chunk_output(*tensors, q_scale=scale)
        result.append({'input': name, 'check': check(y, expected, tensors, tokens, layout)})
        for target, original in zip(tensors, originals):
            target.copy_(original)
    y.fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
    restored = check(y, chunk_output(*tensors, q_scale=scale), tensors, tokens, layout)
    # One request / one shared query head changes only its three value heads.
    baseline = headmajor(y, tensors, layout).clone()
    tensors[0][1, 3].mul_(-.5)
    y.fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
    changed = headmajor(y, tensors, layout)
    unaffected = torch.ones((y.shape[0], HV), device='cuda', dtype=torch.bool)
    unaffected[1, 9:12] = False
    assert torch.equal(changed[unaffected], baseline[unaffected])
    check(y, chunk_output(*tensors, q_scale=scale), tensors, tokens, layout)
    assert bool((changed[1, 9:12] != baseline[1, 9:12]).any())
    tensors[0].copy_(originals[0]); y.fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
    assert torch.equal(headmajor(y, tensors, layout), baseline)
    return {'mutations': result, 'restored': restored,
            'request_shared_query_head_isolation': 'B1 kh3 changes only B1 vh9/10/11; restoration bitwise'}


def export(kernel, dest, spec, env):
    artifacts = export_kernel(kernel, dest)
    bt, dtype, y_dtype, scaled, layout, tile = spec
    write_json(dest / 'abi.json', {
        'operator': 'op16_gdn_chunk_output', 'sm': 87, 'BT': bt, 'HK': HK, 'HV': HV,
        'DK': DK, 'DV': DV, 'q_scale': 1. if scaled else SCALE, 'q_dtype': dtype,
        'output_dtype': y_dtype, 'output_layout': layout, 'output_tile': list(tile),
        'logical_parameters': [f'Q_{dtype}[B,16,C,BT,128]', 'G_fp32[B,48,C,BT]',
                               'QK_fp32[B,48,C,BT,BT]', 'Senter_fp32[B,48,C,128,128]',
                               'R_fp32[B,48,C,BT,128]',
                               f'Y_{y_dtype}' + ('[B,48,C,BT,128]' if layout == 'headmajor' else '[B,C*BT,48,128]')],
        'actual_generated_launches': parse_host((dest / 'host.txt').read_text()),
        'layout': 'contiguous row-major; Senter [K,V]; kh=vh//3; tokenmajor directly stored by TileLang',
        'workspace_bytes': 0, 'resident_parameter_bytes': 0, 'cooperative_launch': False,
        'stream': 'explicit caller stream; current stream resolved per runner invocation',
        'alias_policy': 'disjoint CUDA contiguous buffers; inputs read-only; output overwritten',
        'math': 'FP32 SIMT FMA accumulations, exp/scale FP32, independent sums then add; optional FP16 final store only; no TF32/TC',
        'formula': '(Q*q_scale*exp(G))@Senter + QK@R; QK already includes scale',
        'toolchain': env, 'artifacts': artifacts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--additional', action='store_true', help='FP16 Q / FP32 Y and direct layout equivalence')
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    configure(); env = environment()
    freeze = out / 'source-freeze'; frozen = []
    for path in ('kernels/operators/op16_gdn_chunk_output.py', 'tools/operators/op16_gdn_chunk_output.py',
                 'tools/operators/gdn_reference.py', 'tools/operators/common.py',
                 'tools/operators/abi.py'):
        dest = freeze / path; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest); frozen.append(identity(dest))
    report = {'environment': env, 'source_freeze': frozen, 'host_checks': host_checks(),
              'reference': 'shared FP32 chunk_output, legal states/residuals generated by reference op12..15; synthetic inputs, not model trace',
              'TF32': False, 'cases': [], 'compilation': [], 'failures': [],
              'workspace_bytes': 0, 'resident_parameter_bytes': 0, 'budget_ms_B1_T512': .4,
              'debug_relative_l2_line': .002, 'model_quality_claim': False}
    specs = [(1, 512, 64, 'float16', 'float16', 'random', False, 'headmajor')]
    additions = [(3, bt+1, bt, 'float16', y_dtype, 'random', False, layout)
                 for bt in (16, 32, 64) for y_dtype in ('float16', 'float32')
                 for layout in ('headmajor', 'tokenmajor')]
    if not args.quick and not args.tune:
        specs = [(b, bt+1, bt, 'float16', 'float16', 'random', False, 'headmajor')
                 for bt in (16, 32, 64) for b in (1, 2, 3, 4, 5, 7, 8)]
        specs += [(1, t, bt, 'float16', 'float16', 'random', False, 'headmajor')
                  for bt in (16, 32, 64) for t in (511, 512, 513, 2048, 8192)]
        specs += [(3, bt+1, bt, 'float16', 'float16', mode, False, 'headmajor')
                  for bt in (16, 32, 64) for mode in ('g0', 'beta0', 'beta1', 'strongdecay')]
        specs += [(3, bt+1, bt, 'float32', 'float32', 'random', scaled, layout)
                  for bt in (16, 32, 64) for scaled in (False, True) for layout in ('headmajor', 'tokenmajor')]
        specs += [(3, bt+1, bt, 'float16', 'float16', 'random', False, 'tokenmajor') for bt in (16, 32, 64)]
        specs += [(1, t, 64, 'float16', 'float16', 'random', False, 'tokenmajor') for t in (512, 2048, 8192)]
        specs += [spec for spec in additions if spec[4] == 'float32']
        specs += [(3, 7296, 16, 'float16', 'float16', 'random', False, 'headmajor')]
    if args.additional:
        specs = additions
    kernels = {}
    for b, tokens, bt, dtype, y_dtype, mode, scaled, layout in specs:
        tile_options = ((4, 32), (8, 16), (16, 8)) if args.tune else ((8, 16),)
        started = time.perf_counter()
        tensors = inputs(b, tokens, bt, dtype, mode, scaled)
        torch.cuda.synchronize(); prepare_s = time.perf_counter()-started
        expected = chunk_output(*tensors, q_scale=1. if scaled else SCALE)
        for tile in tile_options:
            key = (bt, dtype, y_dtype, scaled, layout, tile)
            if key not in kernels:
                started = time.perf_counter()
                kernels[key] = gdn_chunk_output(q_scale=1. if scaled else SCALE, bt=bt,
                                               q_dtype=dtype, output_dtype=y_dtype,
                                               output_layout=layout, token_tile=tile[0], value_tile=tile[1])
                report['compilation'].append({'specialization': key, 'compile_prepare_s': time.perf_counter()-started})
            kernel = kernels[key]
            y = torch.empty(output_shape(tensors, layout), device='cuda', dtype=getattr(torch, y_dtype))
            run = lambda: launch(kernel, *tensors, y, stream=torch.cuda.current_stream().cuda_stream)
            started = time.perf_counter(); run(); torch.cuda.synchronize()
            first_ms = (time.perf_counter()-started)*1000
            case = {'B': b, 'T': tokens, 'BT': bt, 'C': tensors[0].shape[2], 'mode': mode,
                    'Q_dtype': dtype, 'Y_dtype': y_dtype, 'Q_already_scaled': scaled,
                    'q_scale': 1. if scaled else SCALE, 'output_layout': layout, 'output_tile': tile,
                    'prepare_inputs_reference_s': prepare_s, 'first_use_ms': first_ms,
                    'validation': check(y, expected, tensors, tokens, layout),
                    'FP64_subset': fp64_subset(tensors, 1. if scaled else SCALE, y, layout),
                    'input_bytes': sum(x.numel()*x.element_size() for x in tensors),
                    'output_bytes': y.numel()*y.element_size(), 'output_stride': list(y.stride()), 'workspace_bytes': 0}
            case['timing'], graph = benchmark(run, repetitions=8, calls_per_replay=4)
            if layout == 'tokenmajor':
                head_key = (bt, dtype, y_dtype, scaled, 'headmajor', tile)
                head_kernel = kernels[head_key]
                other_y = torch.empty(output_shape(tensors, 'headmajor'), device='cuda', dtype=y.dtype)
                launch(head_kernel, *tensors, other_y, stream=torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                assert torch.equal(other_y, headmajor(y, tensors, layout)), 'same-input layout results differ'
                case['same_input_headmajor_tokenmajor_bitwise_equal'] = True
                del other_y
            if b == 3 and tokens <= bt+1 and mode == 'random':
                case['graph'] = graph_checks(kernel, tensors, y, tokens, layout, 1. if scaled else SCALE, graph)
                broken = list(tensors); broken[3] = broken[3].transpose(-1, -2)
                try:
                    launch(kernel, *broken, y, stream=torch.cuda.current_stream().cuda_stream)
                except AssertionError:
                    case['noncontiguous_state_rejected'] = True
                else:
                    raise AssertionError('noncontiguous square state accepted')
            del graph, y
            report['cases'].append(case)
            write_json(out / 'progress.json', report)
            print(json.dumps({'case': len(report['cases']), 'B': b, 'T': tokens, 'BT': bt,
                              'layout': layout, 'tile': tile, 'ms': case['timing']['median_ms'],
                              'l2': case['validation']['FP32_mathematical']['relative_l2']}), flush=True)
        del tensors, expected
    for key, kernel in kernels.items():
        bt, dtype, y_dtype, scaled, layout, tile = key
        dest = out / f'bt{bt}-{dtype}-y{y_dtype}-scaled{int(scaled)}-{layout}-tile{tile[0]}x{tile[1]}'
        export(kernel, dest, key, env)
    report['peak_torch_validation_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['status'] = 'passed'
    write_json(out / 'results.json', report)
    print(json.dumps({'status': 'passed', 'cases': len(report['cases'])}))


if __name__ == '__main__':
    main()
