"""Real norm weights, synthetic GDN-core outputs: numerical/graph/AOT checks."""
import argparse
import ast
import gc
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open

from abi import parse_host
from common import (ROOT, benchmark, configure, environment, error, export_kernel,
                    identity, tensor_sha, write_json)
from kernels.operators.op17_gdn_gated_norm import gdn_gated_norm, launch

MODEL = Path('/home/nvidia/model/vllm-comparison-20260930/awq-http')
SOURCE = ROOT / 'artifacts/operators/op17_gdn_gated_norm/reference-source'
ROWS = (1, 2, 3, 4, 5, 7, 8, 511, 512, 513, 2048, 8192)


def reference(x, z, w):
    xf = x.float()
    return ((xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6))
            * w.float()) * F.silu(z.float())


def check(actual, expected):
    e = error(actual, expected)
    assert e['finite'] and e['relative_l2'] <= .001, e
    return e


def native_reference():
    path = SOURCE / 'layernorm_guard.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'layer_norm_fwd_kernel')
    # All HAS_* constants are passed explicitly; retain only the actual jit.
    node.decorator_list = [ast.Attribute(value=ast.Name(id='triton', ctx=ast.Load()),
                                        attr='jit', ctx=ast.Load())]
    scope = {'triton': triton, 'tl': tl, '__name__': 'op17_frozen_native'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(path), 'exec'), scope)
    kernel = scope['layer_norm_fwd_kernel']

    def run(x, z, w, y, rstd):
        heads = x.shape[0] * 48
        rpb = min(triton.next_power_of_2(triton.cdiv(heads, 32)), 4)
        kernel[(triton.cdiv(heads, rpb), 1)](
            x, y, w, None, z, None, rstd, 128, 128, 128, heads, 128, 1e-6,
            BLOCK_N=128, ROWS_PER_BLOCK=rpb, HAS_BIAS=False, HAS_Z=True,
            NORM_BEFORE_GATE=True, IS_RMS_NORM=True, ACTIVATION='silu', num_warps=1)
    return run


def binding(output):
    started = time.perf_counter()
    lock_path = ROOT / 'artifacts/reference/reference-lock.json'
    locked = json.loads(lock_path.read_text())
    checkpoint = next(f for f in locked['files'] if f['name'] == 'model.safetensors')
    assert (MODEL / 'model.safetensors').stat().st_size == checkpoint['bytes']
    config = json.loads((MODEL / 'config.json').read_text())['text_config']
    assert config['rms_norm_eps'] == 1e-6
    assert config['linear_num_value_heads'] == 48 and config['linear_value_head_dim'] == 128
    assert config['output_gate_type'] in ('silu', 'swish')
    weights, tensors = {}, []
    with safe_open(str(MODEL / 'model.safetensors'), framework='pt', device='cpu') as f:
        names = sorted(n for n in f.keys() if n.endswith('.linear_attn.norm.weight'))
        assert len(names) == 48
        for name in names:
            raw = f.get_tensor(name)
            assert list(raw.shape) == [128], (name, raw.shape)
            converted = raw.half().contiguous()
            assert torch.equal(raw, converted.to(raw.dtype)), name
            weights[name] = converted.cuda()
            tensors.append({'name': name, 'shape': list(raw.shape), 'source_dtype': str(raw.dtype),
                'source_sha256': tensor_sha(raw), 'runtime_sha256': tensor_sha(converted),
                'runtime_dtype': 'float16', 'conversion': 'exact BF16 to FP16'})
    info = {'checkpoint': checkpoint, 'path': str(MODEL / 'model.safetensors'),
        'identity_policy': 'reuse locked full-file SHA256; verify size; read all 48 norm tensors only',
        'lock': identity(lock_path), 'image': locked['config']['image'],
        'config': identity(MODEL / 'config.json'), 'weights': tensors,
        'sources': [identity(p) for p in sorted(SOURCE.glob('*.py'))] + [identity(
            ROOT / 'artifacts/operators/op02_residual_norm/reference-source/model_executor/layers/layernorm.py')],
        'weight_layout': '128 dimensions shared across 48 value heads; independent layer weights',
        'epsilon': 1e-6, 'norm_before_gate': True, 'ordinary_weight': True,
        'input_origin': 'synthetic X/Z, never claimed to be unexported GDN pre-norm trace',
        'load_slice_and_gpu_transfer_s': time.perf_counter() - started}
    write_json(output / 'binding.json', info)
    return weights, info


def export(kernel, directory, dtype, hpb, threads, M=None):
    exported = export_kernel(kernel, directory)
    cuda = (directory / 'kernel.cu').read_text()
    entries = re.findall(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)', cuda)
    abi = {'schema_version': 1, 'operator': 'op17_gdn_gated_norm',
        'tensor_api_order': ['X', 'Z', 'W', 'Y'],
        'shape': {'X': ['M', 48, 128], 'Z': ['M', 48, 128], 'W': [128], 'Y': ['M', 48, 128]},
        'dtypes': {'X': dtype, 'Z': 'float16', 'W': 'float16', 'Y': 'float16'},
        'layout': 'contiguous row-major; shared ordinary W[128]',
        'M': M or 'runtime rows', 'sm': 87, 'heads_per_block': hpb, 'threads': threads,
        'actual_host_launches': parse_host((directory / 'host.txt').read_text()),
        'actual_cuda_entries': [{'symbol': symbol, 'parameters_verbatim': args}
                               for symbol, args in entries if symbol in exported['symbols']],
        'static_shared_memory_bytes': 0, 'cooperative_launch': False,
        'workspace_bytes': 0, 'persistent_weight_bytes_per_layer': 256,
        'alias_contract': 'X/Z/W immutable, Y distinct; graph uses stable addresses',
        'rounding': 'FP32 mean-square/reduction/rstd/norm/ordinary weight/SiLU/gate; final FP16 store',
        'toolchain': environment(), **exported}
    write_json(directory / 'abi.json', abi)
    return abi


def checked_graph(kernel, x, z, w, y, repetitions):
    def run():
        launch(kernel, x, z, w, y, stream=torch.cuda.current_stream().cuda_stream)
    start = time.perf_counter(); run(); torch.cuda.synchronize()
    first_s = time.perf_counter() - start
    initial = check(y, reference(x, z, w))
    timing, graph = benchmark(run, repetitions=repetitions, calls_per_replay=16)
    saved_x, saved_z, expected = x.clone(), z.clone(), y.clone()
    y.fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
    assert torch.equal(y, expected), 'poisoned output not restored'
    x.mul_(-.375); y.fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
    changed_x = check(y, reference(x, z, w)); assert not torch.equal(y, expected)
    x_only = y.clone()
    z.mul_(-.25); z.add_(.5); y.fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
    changed_z = check(y, reference(x, z, w)); assert not torch.equal(y, x_only)
    x.copy_(saved_x); z.copy_(saved_z); y.fill_(float('nan'))
    graph.replay(); torch.cuda.synchronize(); assert torch.equal(y, expected)
    return {'error_fp32_math': initial, 'first_launch_s': first_s, 'timing': timing,
        'graph': {'poison_replay': True, 'changed_x': changed_x, 'changed_z': changed_z,
                  'restore_replay_bit_exact': True}}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    p.add_argument('--repetitions', type=int, default=10)
    args = p.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    configure(); weights, bind = binding(out); native = native_reference()
    w = weights['model.language_model.layers.0.linear_attn.norm.weight']
    report = {'environment': environment(), 'binding': bind, 'cases': [], 'exports': [],
        'tuning': [], 'boundary_cases': [], 'status': 'in_progress',
        'input_scope': 'seeded synthetic GDN outputs/gates; genuine checkpoint norm weights',
        'workspace_bytes': 0}
    for dtype in ('float16', 'float32'):
        started = time.perf_counter(); kernel = gdn_gated_norm(x_dtype=dtype)
        report['exports'].append({'dtype': dtype, 'prepare_compile_s': time.perf_counter() - started,
            'abi': export(kernel, out / 'aot' / dtype, dtype, 4, 128)})
        for m in ROWS:
            x = torch.randn((m, 48, 128), device='cuda', dtype=getattr(torch, dtype))
            z = torch.randn((m, 48, 128), device='cuda', dtype=torch.float16)
            y = torch.empty_like(z)
            result = checked_graph(kernel, x, z, w, y, args.repetitions)
            native_y = torch.empty_like(x); rstd = torch.empty((m * 48,), device='cuda')
            native(x, z, w, native_y, rstd); torch.cuda.synchronize()
            result['error_frozen_native'] = check(y, native_y.half())
            native_timing, _ = benchmark(lambda: native(x, z, w, native_y, rstd),
                repetitions=args.repetitions, calls_per_replay=16)
            budget = {1: .004, 512: .055, 2048: .22, 8192: .88}.get(m)
            elements = m * 48 * 128
            traffic = elements * (x.element_size() + 4)
            case = {'dtype': dtype, 'M': m, 'input_origin': 'synthetic',
                'input_x_sha256': tensor_sha(x), 'input_z_sha256': tensor_sha(z),
                'io_bytes': traffic, 'weight_bytes': 256, 'workspace_bytes': 0,
                'minimum_unique_bytes': traffic + 256,
                'logical_weight_read_bytes': m * 48 * 256,
                'effective_io_GB_s': traffic / result['timing']['median_ms'] / 1e6,
                'frozen_native_timing': native_timing, 'frozen_native_rstd_workspace_bytes': m * 48 * 4,
                'frozen_native_output_contract': f'{dtype}; cast FP32 native output to FP16 for API comparison',
                'budget_ms': budget, **result}
            if budget is not None:
                case['budget_met'] = result['timing']['median_ms'] <= budget
                case['latency_over_budget_ratio'] = result['timing']['median_ms'] / budget
            report['cases'].append(case); write_json(out / 'results.json', report)
            print(f'{dtype} M{m} l2={result["error_fp32_math"]["relative_l2"]:.3g} ms={result["timing"]["median_ms"]:.6f}', flush=True)
            del x, z, y, native_y, rstd; gc.collect()
        # All actual layer weights: extreme gates/values, zero, subnormal, full scale.
        for name, layer_w in weights.items():
            x = torch.randn((3, 48, 128), device='cuda', dtype=getattr(torch, dtype))
            z = torch.randn((3, 48, 128), device='cuda', dtype=torch.float16)
            x[0, 0].zero_(); x[0, 1].fill_(2**-24); x[0, 2].fill_(65504)
            x[0, 3].zero_(); x[0, 3, 0] = 65504
            z[0, 4].fill_(-65504); z[0, 5].fill_(-100); z[0, 6].fill_(100)
            # Keep extreme-gate output within FP16 range, so finite acceptance is meaningful.
            z[0, 7].fill_(65504); x[0, 7].zero_()
            if dtype == 'float32':
                x[0, 8].fill_(1e15); x[0, 9].fill_(1e-30)
            y = torch.empty_like(z)
            run = lambda: launch(kernel, x, z, layer_w, y, stream=torch.cuda.current_stream().cuda_stream)
            run(); torch.cuda.synchronize()
            e = check(y, reference(x, z, layer_w)); assert bool((y[0, 0] == 0).all())
            per_head = [check(y[0, h], reference(x, z, layer_w)[0, h]) for h in range(10)]
            expected = y.clone(); x[1, 17].mul_(-3); z[1, 17].add_(1)
            run(); torch.cuda.synchronize()
            mask = torch.ones((3, 48), device='cuda', dtype=torch.bool); mask[1, 17] = False
            assert torch.equal(y[mask], expected[mask]), 'head/token isolation broken'
            assert not torch.equal(y[1, 17], expected[1, 17])
            report['boundary_cases'].append({'weight': name, 'dtype': dtype, 'error': e,
                'first_ten_head_errors': per_head, 'zero_exact': True, 'head_token_isolation': True})
        write_json(out / 'results.json', report)
    # Alternatives preserve semantics; export each actual tested specialization.
    for m in (1, 512):
        for hpb, threads in ((1, 32), (2, 64), (4, 128), (8, 256)):
            started = time.perf_counter(); kernel = gdn_gated_norm(M=m, x_dtype='float32', heads_per_block=hpb, threads=threads)
            compile_s = time.perf_counter() - started
            x = torch.randn((m, 48, 128), device='cuda'); z = torch.randn_like(x).half(); y = torch.empty_like(z)
            result = checked_graph(kernel, x, z, w, y, args.repetitions)
            abi = export(kernel, out / 'aot' / f'static_m{m}_h{hpb}', 'float32', hpb, threads, M=m)
            report['tuning'].append({'M': m, 'heads_per_block': hpb, 'threads': threads,
                'prepare_compile_s': compile_s, 'abi': abi, **result})
            print(f'tuning M{m} h{hpb} ms={result["timing"]["median_ms"]:.6f}', flush=True)
    torch.cuda.synchronize()
    report['status'] = 'all numerical, boundary, isolation and graph checks passed; budgets evaluated individually'
    report['memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    report['implementation_identity'] = [identity(Path(__file__)), identity(
        ROOT / 'kernels/operators/op17_gdn_gated_norm.py')]
    write_json(out / 'results.json', report)
    print('op17 complete; wrapper cleanup releases GPU lock', flush=True)


if __name__ == '__main__':
    main()
