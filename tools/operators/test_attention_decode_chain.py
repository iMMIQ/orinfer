"""Frozen grouped-GQA partials -> native-gated split merge integration."""
import argparse
import shutil
from pathlib import Path

import torch
from common import (ROOT, benchmark, configure, environment, error,
                    export_kernel, identity, write_json)
from abi import parse_host
from kernels.operators.op22_attention_decode import (
    paged_attention_partials_gqa, validate_host_metadata)
from kernels.operators.op28_attention_split_merge import attention_split_merge
from tools.operators.op22_attention_decode import reference


def check(actual, expected):
    observed = error(actual, expected)
    assert observed['finite'] and observed['relative_l2'] < .002, observed
    return observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--repetitions', type=int, default=15)
    args = parser.parse_args()
    configure()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dependencies = [ROOT / p for p in (
        'kernels/operators/op22_attention_decode.py',
        'kernels/operators/op28_attention_split_merge.py',
        'tools/operators/op22_attention_decode.py',
        'tools/operators/common.py', 'tools/operators/abi.py')]
    before = [identity(p) for p in dependencies]
    for src in dependencies:
        dest = out / 'measurement-source' / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    result = {'status': 'in_progress', 'environment': environment(),
              'source': before, 'exports': [], 'cases': [],
              'scope': 'Synthetic Q/K/V/state, FP16 probability operand; no model TPS or quantization verdict'}
    bs, mp, np = 128, 68, 544
    k = torch.randn((np, bs, 4, 256), device='cuda', dtype=torch.float16) * .5
    v = torch.randn_like(k)
    for splits in (4, 8):
        partial = paged_attention_partials_gqa(mp, np, splits)
        merge = attention_split_merge(splits)
        for name, kernel in ((f'partial_s{splits}', partial), (f'merge_s{splits}', merge)):
            dest = out / 'aot' / name
            exported = export_kernel(kernel, dest)
            result['exports'].append({'name': name, 'files': exported,
                'actual_abi': parse_host((dest / 'host.txt').read_text())})
        for b, context in ((1, 512), (1, 2048), (1, 8192), (1, 8448), (3, 513), (3, 8448), (8, 8192)):
            q = torch.randn((b, 24, 256), device='cuda', dtype=torch.float16)
            gate = torch.randn_like(q) * 3
            y = torch.empty_like(q)
            pages = torch.randperm(np, device='cuda').reshape(8, mp)[:b].int().contiguous()
            if b > 1:
                pages[:, 0] = pages[0, 0]
            lengths = torch.tensor([context - r * 3 for r in range(b)], device='cuda', dtype=torch.int32)
            positions = lengths - 1
            m = torch.empty((b, 24, splits), device='cuda')
            l = torch.empty_like(m)
            o = torch.empty((b, 24, splits, 256), device='cuda')
            def validate():
                validate_host_metadata(pages.cpu().tolist(), lengths.cpu().tolist(),
                                       positions.cpu().tolist(), np, bs)
            def run():
                stream = torch.cuda.current_stream().cuda_stream
                partial(q, k, v, pages, lengths, positions, m, l, o, stream=stream)
                merge(m, l, o, gate, y, stream=stream)
            def expected():
                return reference(q, k, v, pages, lengths, positions, gate)[0]
            def poison():
                for target in (m, l, o, y):
                    target.fill_(float('nan'))
            validate()
            run()
            torch.cuda.synchronize()
            baseline = y.clone()
            observed = check(y, expected())
            hot, graph = benchmark(run, repetitions=args.repetitions)
            mutations = []
            if b == 3 and context == 8448 and splits == 8:
                for name, target in (('Q', q), ('K', k), ('V', v), ('gate', gate),
                                     ('pages', pages), ('lengths', lengths), ('positions', positions)):
                    saved = target.clone()
                    saved_positions = positions.clone() if name == 'lengths' else None
                    if name == 'pages':
                        target[:, 0] = (target[:, 0] + 1) % np
                    elif name == 'lengths':
                        # A query must remain within seqLen. Mutate the coherent
                        # endpoint pair; changing seqLen alone here is illegal.
                        target.sub_(1)
                        positions.sub_(1)
                    elif name == 'positions':
                        target.sub_(1)
                    else:
                        target.mul_(-.75).add_(.25)
                    validate()
                    poison()
                    graph.replay()
                    torch.cuda.synchronize()
                    changed = check(y, expected())
                    assert not torch.equal(y, baseline), f'Ineffective mutation: {name}'
                    target.copy_(saved)
                    if saved_positions is not None:
                        positions.copy_(saved_positions)
                    poison()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(y, baseline), name
                    mutations.append({'input': 'lengths_with_positions' if name == 'lengths' else name, 'changed_error': changed,
                                      'poisoned_all_outputs': True, 'restored_exact': True})
            row = {'B': b, 'context': context, 'splits': splits, 'error': observed,
                   'partial_plus_merge_hot': hot, 'graph_mutations': mutations,
                   'partial_bytes': b * 24 * splits * 258 * 4,
                   'decode_M1_budget_ms': .350 if b == 1 else None}
            result['cases'].append(row)
            write_json(out / 'results.json', result)
            print(f'GQA+merge B{b} context{context} S{splits}: {hot["median_ms"]:.6f} ms, L2 {observed["relative_l2"]:.6g}', flush=True)
    assert before == [identity(p) for p in dependencies], 'Sources changed during measurement'
    result['status'] = 'passed'
    write_json(out / 'results.json', result)


if __name__ == '__main__':
    main()
