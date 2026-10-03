"""Real-checkpoint gate/up correctness, graph, ABI and timing evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import torch
import tilelang.language as T
from common import configure, environment, error, export_kernel, identity, tensor_sha, write_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/projections'))
import decode_common as real
from kernels.operators.op03_ffn_gate_up import ffn_gate_up


def prefill_inputs():
    metadata = next(path for path in real.ACTIVATIONS.glob('*.json')
                    if (lambda m: m['mode'] == 'prefill' and 'layers.0.' in m['kind']
                        and m['kind'].endswith('gate_up_proj'))(json.loads(path.read_text())))
    info = json.loads(metadata.read_text())
    path = real.ACTIVATIONS / info['file']
    assert identity(path)['sha256'] == info['file_sha256']
    a = torch.load(path, map_location='cpu', weights_only=True)
    assert a.shape == (512, 5120) and a.dtype == torch.float16
    rows = []
    for m in (511, 512, 513, 2048, 8192):
        cpu_a = a.repeat((m + 511) // 512, 1)[:m].contiguous()
        rows.append((m, cpu_a, dict(origin='actual layer0 prefill rows' if m <= 512 else
                                  'repetition of actual 512-row prefill; shape/tail test, not real long-context state',
                                  metadata=str(metadata), source_file_sha256=info['file_sha256'],
                                  tensor_sha256=tensor_sha(cpu_a))))
    return rows


def graph_measure(run, a, out, repetitions):
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    start = time.perf_counter()
    with torch.cuda.graph(graph):
        run()
    capture_s = time.perf_counter() - start
    out.fill_(float('nan'))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected), 'poison output replay'
    saved = a.clone()
    a.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert bool((out == 0).all()), 'changed input zero replay'
    a.copy_(saved)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected), 'restore replay'
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
                capture_s=capture_s, graph_poison_changed_input_restore=True)


def abi_and_assembly(kernel, folder, bm, m):
    export = export_kernel(kernel, folder)
    host = (folder / 'host.txt').read_text()
    args = [x.strip().replace('.data_ptr()', '') for x in
            re.search(r'arg_values\s*=\s*([^\n]+)', host).group(1).split(',')]
    assert args == ['A', 'C', 'P', 'S', 'Z', 'M'], args
    source = (folder / 'kernel.cu').read_text()
    signature = re.search(r'extern "C" __global__ void (\w+)\(([^)]*)\);', source)
    config = {key: int(re.search(r'config\.' + key + r' = (\d+)', host).group(1))
              for key in ('gridDimX', 'blockDimX', 'sharedMemBytes')}
    manifest = dict(schema_version=1, target='sm_87', toolchain={key: str(environment()[key]) for key in ('torch', 'tilelang', 'cuda')}, abi_order=args,
                    abi_types=['pointer'] * 5 + ['i32'], source_signature=signature.group(0),
                    symbol=signature.group(1), grid=[config['gridDimX'], f'ceildiv(M,{bm})', 1],
                    block=[config['blockDimX'], 1, 1], shared_memory_bytes=config['sharedMemBytes'],
                    cooperative=False, output='C', dynamic_dimension='M',
                    tensors={'A': ['M', 5120, 'f16', 'row-major'],
                             'P': [34816, 2560, 'u8', 'NK adjacent low/high U4'],
                             'S': [34816, 40, 'f16', 'NG'], 'Z': [34816, 40, 'i8 numeric 0..15', 'NG'],
                             'C': ['M', 34816, 'f16', 'gate-then-up row-major']}, artifact=export)
    write_json(folder / 'abi.json', manifest)
    for name, flag in [('resources.txt', '--dump-resource-usage'), ('sass.txt', '--dump-sass')]:
        result = subprocess.run(['/usr/local/cuda/bin/cuobjdump', flag, str(folder / 'kernel.cubin')],
                                capture_output=True, text=True)
        (folder / name).write_text(result.stdout + result.stderr)
        manifest[name] = dict(exit_code=result.returncode, **identity(folder / name))
    write_json(folder / 'abi.json', manifest)
    return manifest


def fixture(folder, artifact_folder, abi, a, p, s, z, ref, m, bm):
    folder.mkdir(parents=True, exist_ok=True)
    import shutil
    for name in ('kernel.cu', 'kernel.cubin', 'host.txt'):
        shutil.copyfile(artifact_folder / name, folder / name)
    def fileinfo(name):
        info = identity(folder / name)
        return dict(file=name, sha256=info['sha256'])
    def binary(name, tensor):
        (folder / name).write_bytes(tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes())
        return fileinfo(name)
    buffers = []
    for name, tensor, dtype, layout in [('A', a, 'f16', 'row_major'),
                                       ('P', p, 'u8', 'nk-packed-low-high-u4'),
                                       ('S', s, 'f16', 'ng'), ('Z', z, 'i8', 'ng')]:
        buffers.append(dict(name=name, dtype=dtype, shape=list(tensor.shape), layout=layout,
                            alignment=16, access='read', data=binary(name + '.bin', tensor)))
    buffers.insert(1, dict(name='C', dtype='f16', shape=[m, 34816], layout='row_major', alignment=16, access='write'))
    manifest = dict(schema_version=1, target='sm_87', toolchain={key: str(environment()[key]) for key in ('torch', 'tilelang', 'cuda')}, buffers=buffers,
                    kernels=[dict(name='op03', module=fileinfo('kernel.cubin'), source=fileinfo('kernel.cu'),
                                  host_abi=fileinfo('host.txt'), symbol=abi['symbol'],
                                  grid=[abi['grid'][0], (m + bm - 1) // bm, 1], block=abi['block'],
                                  shared_memory_bytes=abi['shared_memory_bytes'], cooperative=False,
                                  args=[dict(kind='buffer', name=name) for name in abi['abi_order'][:-1]] +
                                       [dict(kind='i32', value=m)])],
                    validation=dict(output='C', reference=binary('reference-f32.bin', ref.float()),
                                    relative_l2_tolerance=.002, zero_input='A', repetitions=20))
    write_json(folder / 'manifest.json', manifest)
    return str(folder / 'manifest.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', choices=['decode', 'prefill'], default='decode')
    parser.add_argument('--implementations', default='shared,register')
    parser.add_argument('--rows', default='')
    args = parser.parse_args()
    configure()
    report = dict(environment=environment(), checkpoint=str(real.MODEL), checkpoint_sha256=real.FULL_SHA,
                  phase=args.phase, candidates=[], failures=[], fixtures=[])
    def save():
        write_json(args.output / 'results.json', report)
    raw, sources, source_s = real.raw_weights('gate_up')
    p0, s0, z0, q, logical_s = real.logical(raw)
    report.update(source_tensors=sources, source_read_and_hash_s=source_s, logical_unpack_s=logical_s)
    started = time.perf_counter()
    p, s, z = p0.cuda(), s0.cuda(), z0.cuda()
    torch.cuda.synchronize()
    report['weight_H2D_s'] = time.perf_counter() - started
    report['resident_weight_bytes'] = p.numel() + s.numel() * 2 + z.numel()
    report['weight_tensor_sha256'] = dict(P=tensor_sha(p0), S=tensor_sha(s0), Z=tensor_sha(z0))
    started = time.perf_counter()
    b = real.dequant(q, s0, z0)
    # FP32 weight representation is reference-only; no production expanded weight.
    b32 = b.float()
    del b, raw, q
    torch.cuda.synchronize()
    report['reference_dequant_s'] = time.perf_counter() - started
    inps = real.inputs('gate_up') if args.phase == 'decode' else prefill_inputs()
    if args.rows:
        selected = {int(item) for item in args.rows.split(',')}
        inps = [row for row in inps if row[0] in selected]
    bm = 16 if args.phase == 'decode' else 64
    for implementation in args.implementations.split(','):
        row = dict(implementation=implementation, BM=bm, BN=64, BK=128, stages=2, threads=128, shapes=[])
        report['candidates'].append(row)
        save()
        try:
            started = time.perf_counter()
            kernel = ffn_gate_up(T.dynamic('M'), implementation=implementation, BM=bm)
            row['prepare_s'] = time.perf_counter() - started
            artifact_folder = args.output / 'compiled' / implementation
            abi = abi_and_assembly(kernel, artifact_folder, bm, inps[0][0])
            row['abi'] = abi
            for m, cpu_a, origin in inps:
                a = cpu_a.cuda()
                out = torch.empty((m, 34816), dtype=torch.float16, device='cuda')
                started = time.perf_counter()
                ref = a.float() @ b32.T
                torch.cuda.synchronize()
                reference_s = time.perf_counter() - started
                def run():
                    kernel(a, p, s, z, out, stream=torch.cuda.current_stream().cuda_stream)
                started = time.perf_counter()
                run()
                torch.cuda.synchronize()
                first_s = time.perf_counter() - started
                numerical = error(out, ref)
                assert numerical['finite'] and numerical['relative_l2'] <= .002, numerical
                result = dict(M=m, activation=origin, error=numerical, first_launch_host_s=first_s,
                              reference_matmul_s=reference_s, output_bytes=2*m*34816,
                              workspace_bytes=0, **graph_measure(run, a, out, 20 if m < 512 else 5))
                row['shapes'].append(result)
                if args.phase == 'decode' and m in (1, 3):
                    report['fixtures'].append(fixture(args.output / 'rust-fixtures' / implementation / f'M{m}',
                                                     artifact_folder, abi, a, p, s, z, ref, m, bm))
                print(json.dumps(dict(implementation=implementation, **result)), flush=True)
                save()
                del a, out, ref
        except Exception as exc:
            failure = dict(implementation=implementation, exception=repr(exc), traceback=traceback.format_exc())
            report['failures'].append(failure)
            print(failure['traceback'], flush=True)
            save()
    report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save()
    if not any(candidate['shapes'] for candidate in report['candidates']):
        raise RuntimeError('No usable implementation completed')


if __name__ == '__main__':
    main()
