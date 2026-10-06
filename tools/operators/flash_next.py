"""Check Flash Next dense/GDN/QSA arithmetic and request-owned cache state."""
import argparse
from pathlib import Path

import torch

from kernels.model.flash_next import (
    dense_projection, gdn_sigmoid_norm, qsa_prepare, qsa_short_attention,
)
from tools.operators.common import configure, environment, error, export_kernel, write_json


def checked(actual, expected, limit=.003):
    result = error(actual, expected)
    assert result['finite'] and result['relative_l2'] < limit, result
    return result


def rotate(x, positions, rotary, neox=False):
    x = x.float()
    angle = positions[:, None, None] * 1e7 ** (
        -torch.arange(0, rotary, 2, device=x.device).float()[None, None] / rotary)
    if neox:first,second = x[..., :rotary].chunk(2,dim=-1)
    else:
        pairs = x[..., :rotary].reshape(*x.shape[:-1], rotary // 2, 2)
        first, second = pairs[..., 0], pairs[..., 1]
    result = x.clone()
    a = first * angle.cos() - second * angle.sin()
    b = second * angle.cos() + first * angle.sin()
    result[..., :rotary] = torch.cat((a,b),-1) if neox else torch.stack((a,b),-1).flatten(-2)
    return result.half()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rope-layout',choices=('interleaved','neox'),default='interleaved')
    args = parser.parse_args()
    configure()
    neox = args.rope_layout == 'neox'
    report = {'environment': environment(), 'complete': False, 'cases': [],
              'scope': 'Native Flash Next operators; not full-model quality or TPS.'}
    for m, n, k in ((1, 4, 128), (3, 65, 128), (17, 128, 320)):
        x = torch.randn((m, k), device='cuda').half()
        w = (torch.randn((n, k), device='cuda') / k**.5).half()
        y = torch.empty((m, n), device='cuda', dtype=torch.float16)
        kernel = dense_projection(m, n, k)
        kernel(x, w, y)
        metric = checked(y, (x.float() @ w.float().T).half())
        export_kernel(kernel, args.output / f'dense-M{m}-N{n}-K{k}')
        report['cases'].append({'kind': 'dense', 'shape': [m, n, k], 'error': metric})
    for m in (1, 3, 17):
        x = torch.randn((m, 48, 128), device='cuda').half()
        z = torch.randn_like(x) * 3
        w = torch.rand(128, device='cuda') + .1
        y = torch.empty_like(x)
        kernel = gdn_sigmoid_norm(m)
        kernel(x, z, w, y)
        expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                    * w * torch.sigmoid(z.float())).half()
        metric = checked(y, expected)
        assert not torch.allclose(y, expected * z), 'SiLU and sigmoid must differ'
        export_kernel(kernel, args.output / f'gdn-sigmoid-M{m}')
        report['cases'].append({'kind': 'gdn_sigmoid', 'rows': m, 'error': metric})

    m, heads, kh, dim, rotary, capacity = 17, 24, 2, 256, 64, 64
    qg = torch.randn((m, heads, 2, dim), device='cuda').half()
    k = torch.randn((m, kh, dim), device='cuda').half()
    v = torch.randn_like(k)
    qw = torch.rand(dim, device='cuda') + .2
    kw = torch.rand(dim, device='cuda') + .2
    pos = torch.zeros(1, device='cuda', dtype=torch.int32)
    query = torch.empty((m, heads, dim), device='cuda', dtype=torch.float16)
    gate = torch.empty_like(query)
    kc = torch.full((capacity, kh, dim), float('nan'), device='cuda', dtype=torch.float16)
    vc = torch.full_like(kc, float('nan'))
    y = torch.empty_like(query)
    prep = qsa_prepare(m, capacity,is_neox_style=neox)
    attn = qsa_short_attention(m, capacity)

    def run():
        prep(qg, k, v, qw, kw, pos, query, kc, vc, gate)
        attn(query, kc, vc, gate, pos, y)

    def validate():
        start = int(pos.item())
        positions = torch.arange(start, start + m, device='cuda')
        q = qg[:, :, 0].float()
        q = rotate(q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6) * qw,
                   positions, rotary, neox)
        keys = k.float()
        keys = rotate(keys * torch.rsqrt(keys.square().mean(-1, keepdim=True) + 1e-6) * kw,
                      positions, rotary, neox)
        metrics = {'query': checked(query, q), 'key': checked(kc[start:start+m], keys)}
        assert torch.equal(vc[start:start+m], v) and torch.equal(gate, qg[:, :, 1])
        keys = kc[:start+m].repeat_interleave(heads // kh, 1).float()
        values = vc[:start+m].repeat_interleave(heads // kh, 1).float()
        score = torch.einsum('mhd,shd->hms', q.float(), keys) / dim**.5
        mask = torch.arange(start+m, device='cuda')[None] > positions[:, None]
        score.masked_fill_(mask[None], -torch.inf)
        expected = torch.einsum('hms,shd->mhd', score.softmax(-1), values)
        expected = (expected * torch.sigmoid(qg[:, :, 1].float())).half()
        metrics['attention'] = checked(y, expected)
        return metrics

    run()
    torch.cuda.synchronize()
    metric = validate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    saved_y, saved_k, saved_v = y.clone(), kc[:m].clone(), vc[:m].clone()
    y.fill_(float('nan'))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, saved_y)
    # Existing prefix and changed device-side position are consumed on replay.
    pos.fill_(m)
    graph.replay()
    torch.cuda.synchronize()
    changed = validate()
    assert not torch.equal(y, saved_y)
    assert torch.equal(kc[:m], saved_k) and torch.equal(vc[:m], saved_v)
    pos.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, saved_y)

    # Poisoned unused cells must never contaminate causal attention. Another
    # request uses different private cache allocations before the restore.
    chunk_k = torch.full_like(kc, float('nan'))
    chunk_v = torch.full_like(vc, float('nan'))
    other_k, other_v = torch.zeros_like(kc), torch.zeros_like(vc)
    outputs = []
    start = 0
    for size in (1, 3, 5, 8):
        p = qsa_prepare(size, capacity,is_neox_style=neox)
        a = qsa_short_attention(size, capacity)
        q = torch.empty((size, heads, dim), device='cuda', dtype=torch.float16)
        g = torch.empty_like(q)
        out = torch.empty_like(q)
        inputs = [t[start:start+size].contiguous() for t in (qg, k, v)]
        pos.fill_(start)
        previous_k, previous_v = chunk_k.clone(), chunk_v.clone()
        p(*inputs, qw, kw, pos, q, chunk_k, chunk_v, g)
        a(q, chunk_k, chunk_v, g, pos, out)
        outputs.append(out.clone())
        replay = torch.empty_like(out)
        p(*inputs, qw, kw, pos, q, other_k, other_v, g)
        a(q, other_k, other_v, g, pos, replay)
        p(*inputs, qw, kw, pos, q, previous_k, previous_v, g)
        a(q, previous_k, previous_v, g, pos, replay)
        assert torch.equal(replay, out)
        assert torch.equal(previous_k[:start+size], chunk_k[:start+size])
        start += size
    metric['chunking'] = checked(torch.cat(outputs), saved_y, .0001)
    assert torch.equal(chunk_k[:m], saved_k) and torch.equal(chunk_v[:m], saved_v)
    export_kernel(prep, args.output / 'qsa-prepare')
    export_kernel(attn, args.output / 'qsa-short-attention')
    report['cases'].append({'kind': 'qsa', 'rows': m, 'rope_layout':args.rope_layout,'errors': metric,
        'changed_position_errors': changed,
        'graph_checks': ['poison', 'changed_position', 'restore'],
        'state_checks': ['causal_unused_nan', 'chunking', 'snapshot_restore', 'request_isolation']})
    report['complete'] = True
    write_json(args.output / 'results.json', report)
    print('Flash Next operators passed', flush=True)


if __name__ == '__main__':
    main()
