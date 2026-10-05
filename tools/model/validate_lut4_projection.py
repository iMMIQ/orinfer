"""Validate the retained LUT4 projection against an independent W8 reference.

Uses pre-LUT4 warp-packed model weights and captured L0 activations. Changed
quantization is reported separately from implementation errors. Full shapes,
513-row tails and actual graph replay are supported; no model quality claim.
"""
import argparse
import gc
import json
import shutil
from pathlib import Path
from tools.reference import ACTIVATIONS as REFERENCE_ACTIVATIONS

import numpy as np
import torch
from common import benchmark, configure, environment, error, export_kernel, identity, write_json
from kernels.model.w4a8_lut4 import w4a8_lut4
from kernels.model.w4a8_lut4_helpers import check_lut4_quartets
from kernels.operators.op29_w4_to_temporary_w8 import w4_warp_to_temporary_w8
from kernels.operators.op30_activation_quantization import activation_quantization
from kernels.projections.candidates import int8_gemm
from tools.quantization.w4_i8_pack import LAYOUT, pack_array
from tools.quantization.w4_i8_lut4 import prepare
from tools.quantization.w4_warp_pack import LAYOUT as BASE_LAYOUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--activations-dir', type=Path,
                        default=REFERENCE_ACTIVATIONS)
    parser.add_argument('--tokens', type=int, choices=(512, 513, 2048, 8192), default=512)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--validation-columns', type=int)
    args = parser.parse_args()
    if args.validation_columns is not None:
        assert args.validation_only and args.validation_columns > 0
        assert args.validation_columns % 128 == 0
    configure()
    model = json.loads(args.model.read_text())
    buffers = {b['name']: b for b in model['buffers']}
    sources = [Path(p) for p in ('kernels/model/w4a8_lut4.py',
               'kernels/model/w4a8_lut4_helpers.py', 'tools/quantization/w4_i8_pack.py',
               'tools/quantization/w4_i8_lut4.py', __file__)]
    for path in sources:
        shutil.copyfile(path, args.output/path.name)
    report = dict(status='running', environment=environment(), model=identity(args.model),
                  sources=[identity(p) for p in sources], lut4=True, cases=[],
                  validation_only=args.validation_only, validation_columns=args.validation_columns,
                  scope='Actual L0 weights and captured activations; other token lengths are '
                        'shape proxies; independent LUT4 W8 reference, not BF16/FP8 acceptance')
    count = 65536
    xcodes = torch.arange(count, device='cuda', dtype=torch.int32)
    table_values = torch.randint(0, 256, (count, 4), device='cuda', dtype=torch.int32)
    step = torch.minimum(torch.randint(0, 85, (count,), device='cuda', dtype=torch.int32),
                         (255-table_values.max(dim=1).values)//3)
    table = torch.zeros(count, device='cuda', dtype=torch.int32)
    for j in range(4):
        table |= table_values[:, j] << (j*8)
    output = torch.empty_like(table)
    check = check_lut4_quartets()
    check.adapter.func(xcodes.to(torch.int16), table, step.to(torch.uint8), output,
                       stream=torch.cuda.current_stream().cuda_stream)
    expected = torch.zeros_like(output)
    for j in range(4):
        code = (xcodes >> (j*4)) & 15
        value = table_values.gather(1, (code % 4)[:, None]).squeeze(1)+(code//4)*step-128
        expected |= (value & 255) << (j*8)
    assert torch.equal(output, expected)
    report['lut4_domain'] = dict(patterns=count, passed=True)
    export_kernel(check, args.output/'lut4-domain')
    for base, activation in [('L0_GateUp', 'mlp_gate_up_proj'), ('L0_Down', 'mlp_down_proj')]:
        assert buffers[base+'_P']['layout'] == BASE_LAYOUT, 'Use a pre-LUT4 model manifest'
        values = []
        for suffix in ('_P', '_S', '_Z', '_WS'):
            spec = buffers[base+suffix]
            dtype = dict(i32=np.int32, f16=np.float16, i8=np.int8)[spec['dtype']]
            array = np.fromfile(args.model.parent/spec['data']['file'], dtype=dtype).reshape(spec['shape'])
            values.append(torch.from_numpy(array).cuda())
        pp, s, z, ws = values
        n, groups = s.shape
        k = groups*128
        if args.validation_columns is not None:
            n = min(n, args.validation_columns)
            pp = pp[:n//64].contiguous()
            s, z, ws = s[:n].contiguous(), z[:n].contiguous(), ws[:n].contiguous()
        native = pack_array(pp.cpu().numpy(), verify=True)
        active_pp = torch.from_numpy(native.view(np.int32)).cuda()
        table, step, _, approximation = prepare(s.cpu().numpy(), z.cpu().numpy(), ws.cpu().numpy())
        cp = torch.from_numpy(table.view(np.int32)).cuda()
        candidate_z = torch.from_numpy(step.view(np.int8)).cuda()
        path = args.activations_dir/f'capture-512-0-language_model_model_layers_0_{activation}.pt'
        x = torch.load(path, map_location='cpu', weights_only=True).half().cuda()
        assert x.shape == (512, k)
        if args.tokens != 512:
            x = x.repeat(((args.tokens+511)//512, 1))[:args.tokens].contiguous()
        aq = torch.empty((args.tokens, k), device='cuda', dtype=torch.int8)
        asc = torch.empty(args.tokens, device='cuda', dtype=torch.float16)
        mask = torch.zeros(k, device='cuda', dtype=torch.uint8)
        quant = activation_quantization(k)
        quant.adapter.func(x, mask, aq, asc, stream=torch.cuda.current_stream().cuda_stream)
        w8 = torch.empty((n, k), device='cuda', dtype=torch.int8)
        original = torch.empty((args.tokens, n), device='cuda', dtype=torch.float16)
        expand = w4_warp_to_temporary_w8(n, k, BK=512 if k == 5120 else 256)
        gemm = int8_gemm(args.tokens, n, k, 256, 128, 128, 2, 256)
        def baseline():
            expand.adapter.func(pp, s, z, ws, w8, stream=torch.cuda.current_stream().cuda_stream)
            gemm.adapter.func(aq, w8, asc, ws, original, stream=torch.cuda.current_stream().cuda_stream)
        if args.validation_only:
            baseline()
            torch.cuda.synchronize()
            previous = None
        else:
            previous, _ = benchmark(baseline, repetitions=3)
        packed = pp.cpu().numpy().view(np.uint32)
        decoded = np.stack([(packed >> (j*4)) & 15 for j in range(8)], axis=-1).astype(np.uint8)
        codes = decoded.reshape(n//64, k//128, 4, 8, 4, 8, 2, 2, 2).transpose(
            0, 2, 6, 3, 1, 5, 7, 4, 8).copy().reshape(n, k//128, 128)
        candidate_w8 = approximation[np.arange(n)[:, None, None],
                                     np.arange(k//128)[None, :, None], codes].reshape(n, k).astype(np.int8)
        delta = candidate_w8.astype(np.int16)-w8.cpu().numpy().astype(np.int16)
        diagnostic = dict(actual_weight_code_max_difference=int(np.abs(delta).max()),
                          actual_weight_code_rmse=float(np.sqrt((delta.astype(np.float32)**2).mean())),
                          actual_weight_code_equal_fraction=float((delta == 0).mean()))
        reference_w8 = torch.from_numpy(candidate_w8).cuda()
        golden = torch.empty_like(original)
        gemm.adapter.func(aq, reference_w8, asc, ws, golden,
                          stream=torch.cuda.current_stream().cuda_stream)
        kernel = w4a8_lut4(args.tokens, n, k, 256, 128, 2, coalesced_epilogue=True)
        result = torch.empty_like(golden)
        def run():
            kernel.adapter.func(aq, active_pp, s, candidate_z, cp, asc, ws, result,
                                stream=torch.cuda.current_stream().cuda_stream)
        if args.validation_only:
            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
            torch.cuda.synchronize()
            timing = None
        else:
            timing, graph = benchmark(run, repetitions=3)
        assert torch.equal(result, golden), error(result, golden)
        saved = aq.clone()
        initial = result.clone()
        aq.zero_()
        result.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert bool((result == 0).all())
        aq.copy_(saved)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(result, initial)
        export_kernel(kernel, args.output/f'{base}-m256n128s2')
        row = dict(weight=base, BM=256, BN=128, stages=2, timing=timing,
                   baseline_timing=previous, same_output=True, graph_zero_restore=True,
                   coalesced_epilogue=True, error=error(result, golden),
                   packing=dict(layout=LAYOUT, bytes=native.nbytes, roundtrip_bitwise=True),
                   quantization_diagnostic=diagnostic, original_projection_error=error(result, original),
                   activation=identity(path))
        report['cases'].append(row)
        write_json(args.output/'result.json', report)
        print(json.dumps(dict(weight=base, same_output=True, graph_zero_restore=True)), flush=True)
        del values, pp, s, z, ws, active_pp, x, aq, asc, w8, reference_w8, golden, original, result, graph
        gc.collect()
    report['status'] = 'passed'
    write_json(args.output/'result.json', report)


if __name__ == '__main__':
    main()
