"""Offline op09 compile, real-parameter verification and graph timing."""
import argparse
import ast
import json
import re
import shutil
import statistics
import time
from pathlib import Path
from tools.reference import CHECKPOINT, SOURCE

import torch
import triton
import triton.language as tl
from safetensors import safe_open

from common import configure, environment, error, export_kernel, identity, tensor_sha, write_json
from kernels.operators.op09_gdn_gates import HEADS, gdn_gates, launch

ROOT = Path(__file__).resolve().parents[2]
MODEL = CHECKPOINT / 'model.safetensors'
LOCK = ROOT / 'artifacts/reference/reference-lock.json'
NATIVE = (SOURCE / 'model_executor/layers/mamba/gdn_linear_attn.py')
ROWS = [1, 2, 3, 4, 5, 7, 8, 511, 512, 513, 2048, 8192]


def reference(a, b, a_log, dt):
    return (-a_log.float().exp() * torch.nn.functional.softplus(a.float() + dt.float()),
            torch.sigmoid(b.float()))


def check(g, beta, ref):
    def stable_error(actual, expected):
        # FP64 diagnostics avoid norm overflow on otherwise finite FP32 1e30 cases.
        d = actual.double() - expected.double()
        norm = float(expected.double().norm())
        return {'finite': bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
                'relative_l2': float(d.norm()) / max(norm, 1e-300), 'reference_l2': norm,
                'max_abs': float(d.abs().max()), 'rms_abs': float(d.square().mean().sqrt())}
    result = {"g": stable_error(g, ref[0]), "beta": stable_error(beta, ref[1])}
    result['g_nonpositive'] = bool((g <= 0).all())
    result['beta_in_unit_interval'] = bool(((beta >= 0) & (beta <= 1)).all())
    for name, actual, expected in (("g", g, ref[0]), ("beta", beta, ref[1])):
        near = expected.abs() < 1e-5
        result[name]['near_zero_count'] = int(near.sum())
        result[name]['near_zero_max_abs'] = float((actual - expected).abs()[near].max()) if near.any() else 0.
        result[name]['mixed_check'] = bool(((actual - expected).abs() <= 1e-6 + 1e-4 * expected.abs()).all())
        assert result[name]['finite'] and result[name]['relative_l2'] <= 1e-4 and result[name]['mixed_check'], result
    assert result['g_nonpositive'] and result['beta_in_unit_interval'], result
    return result


def measure(run, launches=128, replays=5):
    """Many nodes per capture amortize Python replay gaps, no reset is needed."""
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            run()
    capture_s = time.perf_counter() - started
    samples = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / (launches * replays))
    # One replay is reported separately, including its entire graph launch overhead.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); graph.replay(); end.record(); end.synchronize()
    return {'median_ms': statistics.median(samples), 'trials_ms': samples,
            'launches_per_graph': launches, 'replays_per_trial': replays,
            'graph_capture_s': capture_s, 'whole_graph_ms': start.elapsed_time(end),
            'timing_scope': 'CUDA events around complete repeated graph replay / executed gate launches; no work removed'}, graph


def load_parameters(out):
    locked = json.loads(LOCK.read_text())
    file_identity = next(x for x in locked['files'] if x['name'] == 'model.safetensors')
    assert MODEL.stat().st_size == file_identity['bytes']
    params, tensors = {}, []
    with safe_open(str(MODEL), framework='pt', device='cpu') as f:
        layers = sorted({int(k.split('.layers.')[1].split('.')[0]) for k in f.keys() if k.endswith('linear_attn.A_log')})
        assert len(layers) == 48
        for layer in layers:
            pair = []
            for suffix in ('A_log', 'dt_bias'):
                name = f'model.language_model.layers.{layer}.linear_attn.{suffix}'
                raw = f.get_tensor(name)
                assert list(raw.shape) == [HEADS]
                pair.append(raw.float().cuda())
                tensors.append({'name': name, 'shape': list(raw.shape), 'storage_dtype': str(raw.dtype),
                                'storage_sha256': tensor_sha(raw), 'fp32_sha256': tensor_sha(raw.float())})
            params[layer] = tuple(pair)
    shutil.copyfile(LOCK, out / 'reference-lock.json')
    shutil.copyfile(NATIVE, out / 'native-gdn-source.py')
    return params, {'model': str(MODEL), 'locked_file_identity': file_identity,
                    'full_hash_policy': 'Reuse previously locked whole-file SHA256; size checked, only 96 gate tensors read',
                    'lock_source': identity(LOCK), 'native_source': identity(NATIVE), 'tensors': tensors}


def native_function(out):
    # Execute only the frozen @triton.jit gate definition, never import/start vLLM.
    source = out / 'native-gdn-source.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'fused_gdn_gating_kernel')
    module = ast.Module(body=[node], type_ignores=[])
    scope = {'triton': triton, 'tl': tl, '__name__': 'op09_native_frozen'}
    exec(compile(module, str(source), 'exec'), scope)
    return scope['fused_gdn_gating_kernel']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    configure()
    params, parameter_identity = load_parameters(out)
    native = native_function(out)
    stream = torch.cuda.current_stream().cuda_stream
    report = {'environment': environment(), 'parameters': parameter_identity, 'cases': [], 'tuning': [],
              'all_layer_cases': [], 'native_comparisons': [], 'workspace_bytes': 0,
              'resident_parameter_bytes_per_layer': 384, 'optional_retained_raw_plus_expA_bytes': 576,
              'source': identity(ROOT / 'kernels/operators/op09_gdn_gates.py'),
              'input_scope': 'Synthetic a/b only; all 48 actual checkpoint A_log/dt_bias loaded; no full model service',
              'rounding': 'BF16 checkpoint params -> exact FP32; inputs -> FP32, all arithmetic/output FP32',
              'failures': []}
    kernels, selected = {}, {}
    for dtype in ('float16', 'float32'):
        for mode in ('a_log', 'exp_a'):
            for block, threads in ((128, 128), (256, 128), (512, 128)):
                started = time.perf_counter()
                kernel = gdn_gates(dtype, block, threads, mode)
                kernels[dtype, mode, block, threads] = kernel
                trial = {'dtype': dtype, 'parameter_mode': mode, 'block': block, 'threads': threads,
                         'prepare_s': time.perf_counter() - started, 'timings': {}}
                al, dt = params[0]
                parameter = al if mode == 'a_log' else al.exp()
                for m in (1, 512):
                    a = torch.randn((m, HEADS), dtype=getattr(torch, dtype), device='cuda') * 3
                    b = torch.randn_like(a) * 3
                    g, beta = torch.empty_like(a, dtype=torch.float32), torch.empty_like(a, dtype=torch.float32)
                    run = lambda: launch(kernel, a, b, parameter, dt, g, beta,
                                         stream=torch.cuda.current_stream().cuda_stream)
                    started = time.perf_counter(); run(); torch.cuda.synchronize()
                    trial.setdefault('first_use_ms', {})[str(m)] = 1000 * (time.perf_counter() - started)
                    trial.setdefault('error', {})[str(m)] = check(g, beta, reference(a, b, al, dt))
                    trial['timings'][str(m)], graph = measure(run)
                    del graph
                report['tuning'].append(trial)
                write_json(out / 'progress.json', report)
            for shape_mode, m in (('decode', 1), ('prefill', 512)):
                trials = [t for t in report['tuning'] if t['dtype'] == dtype and t['parameter_mode'] == mode]
                best = min(trials, key=lambda t: t['timings'][str(m)]['median_ms'])
                selected[dtype, mode, shape_mode] = (best['block'], best['threads'])
    report['selected'] = {'/'.join(k): dict(block=v[0], threads=v[1]) for k, v in selected.items()}
    for (dtype, mode, shape_mode), (block, threads) in selected.items():
        kernel = kernels[dtype, mode, block, threads]
        dest = out / f'{dtype}-{mode}-{shape_mode}'
        exported = export_kernel(kernel, dest)
        cuda, host = (dest / 'kernel.cu').read_text(), (dest / 'host.txt').read_text()
        declaration = re.search(r'extern "C" __global__ void (\w+)\(([^;]+)\);', cuda)
        assert declaration
        abi = {'operator': 'op09_gdn_gates', 'entry_symbol': declaration.group(1), 'sm': 87,
               'ordered_arguments': [{'index': i, 'declaration': arg.strip(),
                                      'driver_type': 'device_ptr:u64' if '*' in arg else 'int32'}
                                     for i, arg in enumerate(declaration.group(2).split(','))],
               'logical_buffers': {'a': ['M',48,dtype], 'b': ['M',48,dtype],
                                   'parameter': [48,'float32',mode], 'dt_bias': [48,'float32'],
                                   'g': ['M',48,'float32'], 'beta': ['M',48,'float32']},
               'layout': 'contiguous row-major, parameter vectors indexed by head',
               'launch': {'grid': [f'ceildiv(M*48,{block})',1,1], 'block': [threads,1,1],
                          'shared_memory_bytes': 0, 'cooperative': False},
               'workspace_bytes': 0, 'resident_parameter_bytes': 384, 'artifacts': exported,
               'toolchain': report['environment'], 'host_wrapper': host,
               'argument_order_source': 'actual exported CUDA declaration and host.txt',
               'aliasing': 'six disjoint buffers, stable addresses for graph replay; explicit stream',
               'mathematics': 'g=-exp(A_log)*softplus(a+dt_bias), beta=sigmoid(b); g is negative log-decay',
               'rounding': report['rounding']}
        write_json(dest / 'abi.json', abi)
    for dtype in ('float16', 'float32'):
        for layer in (0, 32):
            al, dt = params[layer]
            expa = al.exp()
            report.setdefault('exp_a_identities', []).append({'layer': layer, 'dtype': dtype,
                'sha256': tensor_sha(expa), 'preparation': 'Torch CUDA FP32 exp of exact converted checkpoint A_log, outside timed path',
                'exp_a_min': float(expa.min()), 'exp_a_max': float(expa.max())})
            for m in ROWS:
                a = torch.randn((m, HEADS), dtype=getattr(torch, dtype), device='cuda') * 5
                b = torch.randn_like(a) * 5
                original_a, original_b = a.clone(), b.clone()
                ref = reference(a, b, al, dt)
                g, beta = torch.empty_like(a, dtype=torch.float32), torch.empty_like(a, dtype=torch.float32)
                for mode in ('a_log', 'exp_a'):
                    shape_mode = 'decode' if m <= 8 else 'prefill'
                    kernel = kernels[(dtype, mode) + selected[dtype, mode, shape_mode]]
                    parameter = al if mode == 'a_log' else expa
                    run = lambda: launch(kernel, a, b, parameter, dt, g, beta,
                                         stream=torch.cuda.current_stream().cuda_stream)
                    run(); torch.cuda.synchronize()
                    case = {'dtype': dtype, 'layer': layer, 'parameter_mode': mode, 'M': m,
                            'error': check(g,beta,ref), 'input_tensor_sha256': [tensor_sha(a),tensor_sha(b)],
                            'input_bytes': 2*a.numel()*a.element_size(), 'output_bytes': 8*a.numel()}
                    case['timing'], graph = measure(run)
                    a.add_(.375); b.sub_(.625); g.fill_(float('nan')); beta.fill_(float('nan'))
                    graph.replay(); torch.cuda.synchronize()
                    case['graph_changed'] = check(g,beta,reference(a,b,al,dt))
                    a.copy_(original_a); b.copy_(original_b); g.fill_(float('nan')); beta.fill_(float('nan'))
                    graph.replay(); torch.cuda.synchronize()
                    case['graph_restored'] = check(g,beta,ref)
                    report['cases'].append(case)
                    del graph
                write_json(out/'progress.json', report)
            # Compare stable FP32 path with actual locked native gating and its beta rounding.
            ng = torch.empty_like(g); nb = torch.empty_like(b)
            native[(m,1,triton.cdiv(HEADS,8))](ng,nb,al,a,b,dt,1,HEADS,1.,20.,8,num_warps=1)
            torch.cuda.synchronize()
            report['native_comparisons'].append({'dtype': dtype, 'layer': layer, 'M': m,
                'g_error_to_stable_reference': error(ng,ref[0]),
                'native_beta_dtype': str(nb.dtype), 'beta_error_to_FP32_reference': error(nb,ref[1]),
                'beta_cast_FP32_reference_match': error(nb,ref[1].to(nb.dtype))})
    # Every real GDN layer, stable extremes, negative tails and FP32 huge finite input.
    extreme = [-65504.,-1000.,-100.,-80.,-40.,-20.,-10.,-1.,-0.,0.,1.,10.,20.,40.,80.,100.,1000.,65504.]
    for dtype in ('float16','float32'):
        values = extreme if dtype == 'float16' else [-1e30] + extreme + [1e30]
        for layer, (al, dt) in params.items():
            a = torch.tensor(values, device='cuda',dtype=getattr(torch,dtype)).repeat(48).reshape(-1,48)
            b = a.flip(1).contiguous()
            g,beta = torch.empty_like(a,dtype=torch.float32),torch.empty_like(a,dtype=torch.float32)
            for mode in ('a_log','exp_a'):
                kernel = kernels[(dtype,mode) + selected[dtype,mode,'decode']]
                parameter = al if mode == 'a_log' else al.exp()
                launch(kernel,a,b,parameter,dt,g,beta,stream=stream); torch.cuda.synchronize()
                report['all_layer_cases'].append({'layer':layer,'dtype':dtype,'parameter_mode':mode,
                    'input_values': values,'error':check(g,beta,reference(a,b,al,dt))})
    # Measure a single-node graph separately: includes all graph/launch/event device interval.
    al,dt = params[0]
    report['single_node_timings'] = []
    for m in (1,512,2048,8192):
        a=torch.zeros((m,48),device='cuda',dtype=torch.float16); b=torch.zeros_like(a)
        g,beta=torch.empty_like(a,dtype=torch.float32),torch.empty_like(a,dtype=torch.float32)
        kernel=kernels[('float16','a_log')+selected['float16','a_log','decode' if m==1 else 'prefill']]
        run=lambda: launch(kernel,a,b,al,dt,g,beta,stream=torch.cuda.current_stream().cuda_stream)
        timing, graph=measure(run,launches=1,replays=100)
        report['single_node_timings'].append({'M':m,'timing':timing})
        del graph
    report['peak_torch_validation_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['budget']={'decode_M1_ms':.002,'prefill512_ms':.012,'prefill2048_ms':.048,'prefill8192_ms':.192,
                      'scope':'provisional stage design includes possible fusion; independent launch results reported without claiming model TPS'}
    report['status']='passed'
    write_json(out/'results.json',report)
    print(json.dumps({'status':'passed','cases':len(report['cases']),'all_layer_cases':len(report['all_layer_cases']),
                      'selected':report['selected'],'single_node_timings':report['single_node_timings']},indent=2))


if __name__ == '__main__':
    main()
