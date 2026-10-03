"""GDN layer0/32 and full-attention layer3/35 W4 mixer output evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
import tilelang.language as T
from safetensors import safe_open

from common import configure, environment, error, export_kernel, identity, tensor_sha, write_json
from abi import parse_host

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/projections'))
import decode_common as real
from kernels.operators.op18_mixer_out import build_mixer_out, mixer_out_partial, N_HIDDEN, K_MIXER


def load_weights(layer):
    start = time.perf_counter()
    part = 'linear_attn.out_proj' if layer in (0, 32) else 'self_attn.o_proj'
    raw, sources = {}, []
    with safe_open(str(real.MODEL), framework='pt', device='cpu') as handle:
        for field in ('weight_packed', 'weight_scale', 'weight_zero_point'):
            name = f'model.language_model.layers.{layer}.{part}.{field}'
            value = handle.get_tensor(name)
            sources.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_sha(value)))
            raw[field] = value
    raw['weight_scale'] = raw['weight_scale'].half()
    # Locked scales are BF16 originally and must remain lossless on conversion.
    assert raw['weight_scale'].dtype == torch.float16
    with safe_open(str(real.MODEL), framework='pt', device='cpu') as handle:
        assert torch.equal(raw['weight_scale'].bfloat16(),
                           handle.get_tensor(f'model.language_model.layers.{layer}.{part}.weight_scale'))
    read_s = time.perf_counter() - start
    p, s, z, q, unpack_s = real.logical(raw)
    assert p.shape == (N_HIDDEN, K_MIXER // 2)
    assert s.shape == z.shape == (N_HIDDEN, K_MIXER // 128)
    assert int(z.min()) >= 0 and int(z.max()) <= 15
    start = time.perf_counter()
    recovered = torch.stack((p & 15, p >> 4), dim=-1).reshape_as(q)
    assert torch.equal(recovered, q)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    repacked = (q.reshape(N_HIDDEN, -1, 8).long() << shifts).sum(-1).to(torch.int32)
    assert torch.equal(repacked, raw['weight_packed'])
    # Source zero words pack eight adjacent N channels, unlike weight words
    # which pack eight adjacent K columns. Preserve that source topology.
    zeros = (z.reshape(N_HIDDEN // 8, 8, K_MIXER // 128).long()
             << shifts[None, :, None]).sum(1).to(torch.int32)
    assert torch.equal(zeros, raw['weight_zero_point'])
    return p, s, z, q, dict(source_tensors=sources, source_read_and_hash_s=read_s,
                           logical_unpack_s=unpack_s, lossless_roundtrip_s=time.perf_counter()-start,
                           packed_and_zero_roundtrip=True, bf16_scale_roundtrip=True,
                           weight_tensor_sha256=dict(P=tensor_sha(p), S=tensor_sha(s), Z=tensor_sha(z)))


def load_inputs(layer, phase):
    part = 'linear_attn.out_proj' if layer in (0,32) else 'self_attn.o_proj'
    found = []
    for metadata in real.ACTIVATIONS.glob('*.json'):
        info = json.loads(metadata.read_text())
        if info['mode'] != phase or not info['kind'].endswith(f'layers.{layer}.{part}'):
            continue
        path = real.ACTIVATIONS / info['file']
        assert identity(path)['sha256'] == info['file_sha256']
        value = torch.load(path, map_location='cpu', weights_only=True)
        assert value.dtype == torch.float16 and value.shape[1] == K_MIXER
        assert tensor_sha(value) == info['tensor_sha256']
        found.append((info.get('computed_tokens_before',0), value, metadata, info))
    found.sort(key=lambda row: row[0])
    if layer in (0,32):
        assert len(found) == (8 if phase == 'decode' else 1), len(found)
        base = torch.cat([row[1] for row in found])
    else:
        assert not found, 'unexpected full attention trace; adapt source handling'
        # No captured full-attention o_proj input exists in the locked directory.
        generator = torch.Generator().manual_seed(20261002 + layer)
        base = torch.randn((8 if phase == 'decode' else 512,K_MIXER),
                           generator=generator, dtype=torch.float16) * .25
    shapes = (1,2,3,4,5,7,8) if phase == 'decode' else (511,512,513,2048,8192)
    result = []
    for m in shapes:
        value = base.repeat((m+len(base)-1)//len(base),1)[:m].contiguous()
        origin = ('synthetic FP16 input; no captured full-attention o_proj trace' if not found else
                  'consecutive real M1 decode inputs; projection test, not simultaneous model batch' if phase == 'decode' else
                  'actual layer prefill input' if m == 512 else
                  'truncated actual 512-row prefill input' if m == 511 else
                  'repeated actual 512-row input; not a real 2K/8K model trace')
        result.append((m,value,dict(origin=origin,
            semantic='GDN gated-norm output' if found else 'synthetic stand-in for sigmoid-gated full attention output',
            repeated_rows=m>len(base), base_rows=len(base),
            metadata_files=[str(row[2]) for row in found],
            source_file_sha256=[row[3]['file_sha256'] for row in found],
            tensor_sha256=tensor_sha(value))))
    return result


def graph_measure(run, a, p, s, z, y, partial, repetitions):
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    expected = y.clone()
    expected_partial = partial.clone() if partial is not None else None
    graph = torch.cuda.CUDAGraph()
    start = time.perf_counter()
    with torch.cuda.graph(graph):
        run()  # Both projection and merge are captured, in order, on current stream.
    capture_s = time.perf_counter() - start
    y.fill_(float('nan'))
    if partial is not None:
        partial.fill_(float('nan'))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, expected), 'output poison replay failed'
    if partial is not None:
        assert torch.equal(partial, expected_partial), 'partial poison replay failed'
    saved = a.clone()
    a.zero_()
    y.fill_(float('nan'))
    if partial is not None:
        partial.fill_(float('nan'))
    graph.replay()
    torch.cuda.synchronize()
    assert bool((y == 0).all()), 'changed X replay failed'
    if partial is not None:
        assert bool((partial == 0).all()), 'changed X partial replay failed'
    a.copy_(saved)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, expected), 'restored X replay failed'
    if partial is not None:
        assert torch.equal(partial, expected_partial), 'restored partial replay failed'
    # Mutate each graph-bound weight input separately, re-evaluate same W4 math,
    # and restore. This verifies actual replay reads P/S/Z at stable addresses.
    for name, tensor in [('P',p),('S',s),('Z',z)]:
        saved_weight = tensor.clone()
        if name == 'P':
            tensor.bitwise_xor_(255)
        elif name == 'S':
            tensor.mul_(.5)
        else:
            tensor.copy_(15-tensor)
        y.fill_(float('nan'))
        if partial is not None:
            partial.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(y).all())
        qchanged = torch.stack((p & 15,p >> 4),dim=-1).reshape(N_HIDDEN,K_MIXER)
        wchanged = ((qchanged.reshape(N_HIDDEN,-1,128).float()-z[:,:,None].float())
                    *s[:,:,None].float()).reshape(N_HIDDEN,K_MIXER).half()
        changed_ref = a.float() @ wchanged.float().T
        numerical = error(y,changed_ref)
        assert numerical['finite'] and numerical['relative_l2'] <= .002, (name,numerical)
        assert not torch.equal(y,expected), f'changed {name} did not change output'
        if partial is not None:
            assert bool(torch.isfinite(partial).all())
        tensor.copy_(saved_weight)
        y.fill_(float('nan'))
        if partial is not None:
            partial.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y,expected), f'restored {name} failed'
        if partial is not None:
            assert torch.equal(partial,expected_partial), f'restored {name} partial failed'
        del saved_weight,qchanged,wchanged,changed_ref
    trials = []
    for _ in range(3):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(repetitions):
            graph.replay()
        end.record()
        end.synchronize()
        trials.append(begin.elapsed_time(end) / repetitions)
    return dict(median_ms=sorted(trials)[1], trials_ms=trials, repetitions=repetitions,
                calls_per_replay=1, kernels_per_call=2 if partial is not None else 1,
                capture_s=capture_s, graph_all_partial_and_Y_poison_changed_X_P_S_Z_restore=True,
                timing_scope='CUDA-event complete graph; projection+merge for split-K')


def export_plan(plan, folder):
    result = {}
    for name, kernel in [('projection', plan.projection), ('merge', plan.merge)]:
        if kernel is None:
            continue
        directory = folder / name
        export = export_kernel(kernel, directory)
        launches = parse_host((directory / 'host.txt').read_text())
        manifest = dict(schema_version=1, route=plan.route, stage=name, target='sm_87',
                        toolchain={key: str(environment()[key]) for key in ('torch', 'tilelang', 'cuda')},
                        dynamic_dimension='M; kernel infers M from actual tensors per invocation',
                        generated_launches=launches, cooperative=False,
                        parameter_bindings=({'P': 'partialF32', 'O': 'YF16', 'M': 'rows'} if name == 'merge' else
                                            {'A': 'XF16', 'P': 'packedU4', 'S': 'scaleF16', 'Z': 'zeroI8',
                                             'O' if plan.merge else 'C': 'partialF32' if plan.merge else 'YF16',
                                             'M': 'rows'}),
                        tensors=dict(A=['M', K_MIXER, 'f16', 'row-major'],
                                     P=[N_HIDDEN, K_MIXER//2, 'u8', 'adjacent low/high U4'],
                                     S=[N_HIDDEN, K_MIXER//128, 'f16', 'NG'],
                                     Z=[N_HIDDEN, K_MIXER//128, 'i8 numeric 0..15', 'NG'],
                                     partial=[plan.splits, 'M', N_HIDDEN, 'f32', 'SMN'] if plan.merge else None,
                                     Y=['M', N_HIDDEN, 'f16', 'row-major']),
                        artifact=export)
        for output, flag in [('resources.txt', '--dump-resource-usage'), ('sass.txt', '--dump-sass')]:
            proc = subprocess.run(['/usr/local/cuda/bin/cuobjdump', flag, str(directory/'kernel.cubin')],
                                  capture_output=True, text=True)
            (directory/output).write_text(proc.stdout+proc.stderr)
            manifest[output] = dict(exit_code=proc.returncode, **identity(directory/output))
        write_json(directory/'abi.json', manifest)
        result[name] = manifest
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--layers', default='0,32,3,35')
    parser.add_argument('--phase', choices=['decode', 'prefill', 'all'], default='all')
    parser.add_argument('--routes', default='splitk_shared,splitk_register,full_register')
    parser.add_argument('--rows', default='')
    parser.add_argument('--skip-adversarial', action='store_true')
    args = parser.parse_args()
    configure()
    report = dict(environment=environment(), checkpoint=str(real.MODEL),
                  checkpoint_sha256=real.FULL_SHA, checkpoint_hash_origin='previous locked identity; full 18GB not rescanned',
                  mathematical_reference='FP32 matmul with explicitly FP16 dequantized W; TF32 disabled',
                  budget_ms=dict(decode_M1=.105, prefill_M512=.600, prefill_M2048=2.4, prefill_M8192=9.6), layers=[], routes=[], failures=[])
    def save():
        write_json(args.output/'results.json', report)
    dependencies = ['kernels/projections/candidates.py', 'kernels/operators/op03_ffn_gate_up.py',
                    'kernels/operators/op05_ffn_down.py', 'kernels/operators/op18_mixer_out.py', 'tools/operators/op18_mixer_out.py',
                    'tools/operators/common.py', 'tools/operators/abi.py', 'tools/projections/decode_common.py']
    report['frozen_dependencies'] = []
    for relative in dependencies:
        destination = args.output/'measurement-source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
        report['frozen_dependencies'].append(identity(destination))
    for bad in (0, -1):
        try:
            mixer_out_partial(bad)
            raise AssertionError('accepted nonpositive M')
        except ValueError:
            pass
    try:
        mixer_out_partial(1, SPLIT=5)
        raise AssertionError('accepted uneven split')
    except ValueError:
        pass
    report['API_rejection_checks'] = ['M0', 'M-1', 'uneven split5']
    plans = []
    for route in args.routes.split(','):
        start = time.perf_counter()
        plan = build_mixer_out(T.dynamic('M'), route=route)
        row = dict(route=route, prepare_s=time.perf_counter()-start,
                   abi=export_plan(plan, args.output/'compiled'/route), shapes=[], adversarial=[])
        plans.append((plan, row))
        report['routes'].append(row)
        save()
    phases = ('decode', 'prefill') if args.phase == 'all' else (args.phase,)
    selected = {int(item) for item in args.rows.split(',')} if args.rows else None
    for layer in map(int, args.layers.split(',')):
        p0, s0, z0, q, source = load_weights(layer)
        start = time.perf_counter()
        p, s, z = p0.cuda(), s0.cuda(), z0.cuda()
        torch.cuda.synchronize()
        source.update(layer=layer, part=('linear_attn.out_proj' if layer in (0,32) else 'self_attn.o_proj'), weight_H2D_s=time.perf_counter()-start,
                      resident_weight_bytes=p.numel()+s.numel()*2+z.numel())
        start = time.perf_counter()
        b16 = real.dequant(q, s0, z0)
        b32 = b16.float()
        del b16, q, p0, s0, z0
        torch.cuda.synchronize()
        source['reference_dequant_s'] = time.perf_counter()-start
        report['layers'].append(source)
        save()
        for phase in phases:
            for m, cpu_a, origin in load_inputs(layer, phase):
                if selected and m not in selected:
                    continue
                a = cpu_a.cuda()
                start = time.perf_counter()
                ref = a.float() @ b32.T
                torch.cuda.synchronize()
                reference_s = time.perf_counter()-start
                for plan, row in plans:
                    y = torch.empty((m, N_HIDDEN), device='cuda', dtype=torch.float16)
                    shape = plan.workspace_shape(m)
                    partial = torch.empty(shape, device='cuda', dtype=torch.float32) if shape else None
                    def run():
                        plan(a, p, s, z, y, partial, stream=torch.cuda.current_stream().cuda_stream)
                    try:
                        start = time.perf_counter()
                        run()
                        torch.cuda.synchronize()
                        first_s = time.perf_counter()-start
                        numerical = error(y, ref)
                        assert numerical['finite'] and numerical['relative_l2'] <= .002, numerical
                        measured = graph_measure(run, a, p, s, z, y, partial, 20 if m <= 8 else 3)
                        result = dict(layer=layer, phase=phase, M=m, activation=origin, error=numerical,
                                      first_launch_host_s=first_s, reference_matmul_s=reference_s,
                                      workspace_bytes=partial.numel()*4 if partial is not None else 0,
                                      output_bytes=y.numel()*2, **measured)
                        row['shapes'].append(result)
                        print(json.dumps(dict(route=plan.route, **result)), flush=True)
                    except Exception as exc:
                        failure = dict(route=plan.route, layer=layer, M=m, exception=repr(exc), traceback=traceback.format_exc())
                        report['failures'].append(failure)
                        print(failure['traceback'], flush=True)
                    save()
                    del y, partial
                del a, ref
        if not args.skip_adversarial:
            # Full dimensions and symbolic M7, no surrogate toy weights.
            for kind in ('zero', 'random', 'extreme'):
                a = (torch.zeros((7, K_MIXER), device='cuda', dtype=torch.float16) if kind == 'zero' else
                     torch.randn((7, K_MIXER), device='cuda', dtype=torch.float16)*.5 if kind == 'random' else
                     (torch.arange(K_MIXER, device='cuda') % 2 * 2 - 1).half().repeat(7, 1)*32)
                ref = a.float() @ b32.T
                for plan, row in plans:
                    y = torch.empty((7, N_HIDDEN), device='cuda', dtype=torch.float16)
                    shape = plan.workspace_shape(7)
                    partial = torch.empty(shape, device='cuda', dtype=torch.float32) if shape else None
                    plan(a, p, s, z, y, partial, stream=torch.cuda.current_stream().cuda_stream)
                    torch.cuda.synchronize()
                    numerical = error(y, ref)
                    assert numerical['finite'] and numerical['relative_l2'] <= .002, numerical
                    if kind == 'zero':
                        assert bool((y == 0).all())
                    row['adversarial'].append(dict(layer=layer, M=7, kind=kind, error=numerical))
                    del y, partial
                del a, ref
                save()
        del p, s, z, b32
    report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['production_reference_buffers_note'] = 'expanded FP32 W and reference-only allocations are not resident production weights/workspace'
    save()
    if report['failures']:
        raise RuntimeError('one or more numerical/graph cases failed; see results.json')


if __name__ == '__main__':
    main()
