"""Check Flash QSA/indexer MRoPE, nonaligned compression and feature graph replay."""
import argparse
from pathlib import Path
import torch
from kernels.model import qsa, flash_next
from kernels.vision.features import overlay
from tools.operators.common import configure, error, write_json


def rotated(x, weight, coords):
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight
    pairs = torch.arange(32, device='cuda')
    axes = torch.where((pairs % 3 == 1) & (pairs < 33), 1,
                       torch.where((pairs % 3 == 2) & (pairs < 30), 2, 0))
    angles = coords[:, axes].float() * torch.pow(1e7, -2 * pairs.float() / 64)
    while angles.ndim < x.ndim:
        angles = angles.unsqueeze(1)
    a, b = x[..., :32], x[..., 32:64]
    return torch.cat((a * angles.cos() - b * angles.sin(),
                      a * angles.sin() + b * angles.cos(), x[..., 64:]), -1).half()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); configure()
    results = []
    capacity = 256
    coords = torch.stack((torch.arange(capacity), torch.arange(capacity) // 3,
                          torch.arange(capacity) % 11), -1).int().cuda()
    for m, start in [(1, 7), (7, 3), (16, 5), (128, 2)]:
        position = torch.tensor([start], device='cuda', dtype=torch.int32)
        qk = torch.randn((m, 5, 128), device='cuda', dtype=torch.float16)
        weight = torch.rand(128, device='cuda') + .5
        output = torch.empty((m, 4, 128), device='cuda', dtype=torch.float16)
        qsa.index_query(m, capacity)(qk, weight, position, output, coords)
        reference = rotated(qk[:, :4], weight, coords[start:start+m])
        e = error(output, reference)
        assert e['relative_l2'] < .001 and e['finite'], e
        pending = torch.randn((4, 128), device='cuda', dtype=torch.float16)
        cache = torch.zeros((capacity//4, 128), device='cuda', dtype=torch.float16)
        qsa.index_compress(m, capacity, mrope=True)(qk, pending, weight, position, cache, coords)
        for block in range(start//4, (start+m)//4):
            raw = torch.stack([pending[token%4] if token < start else qk[token-start, 4]
                               for token in range(block*4, block*4+4)]).float().mean(0).half()
            expected = rotated(raw[None], weight, coords[block*4:block*4+1])[0]
            ce = error(cache[block], expected)
            assert ce['relative_l2'] < .001 and ce['finite'], ce
        gate = torch.randn((m, 24, 2, 256), device='cuda', dtype=torch.float16)
        k, v = [torch.randn((m, 2, 256), device='cuda', dtype=torch.float16) for _ in range(2)]
        qw, kw = [torch.rand(256, device='cuda') + .5 for _ in range(2)]
        query, copied_gate = [torch.empty((m, 24, 256), device='cuda', dtype=torch.float16) for _ in range(2)]
        key, value = [torch.empty_like(k) for _ in range(2)]
        flash_next.qsa_prepare(m, capacity, is_neox_style=True, staged=True, mrope=True)(
            gate, k, v, qw, kw, position, query, key, value, coords, copied_gate)
        qe = error(query, rotated(gate[:, :, 0], qw, coords[start:start+m]))
        ke = error(key, rotated(k, kw, coords[start:start+m]))
        assert max(qe['relative_l2'], ke['relative_l2']) < .001, (qe, ke)
        assert torch.equal(copied_gate, gate[:, :, 1]) and torch.equal(value, v)
        # Uniform axes must preserve the existing text-only route.
        linear = torch.arange(capacity, device='cuda', dtype=torch.int32)[:, None].expand(-1, 3).contiguous()
        qsa.index_query(m, capacity)(qk, weight, position, output, linear)
        text = torch.empty_like(output)
        qsa.index_query(m)(qk, weight, position, text)
        assert torch.equal(output, text)
        results.append(dict(rows=m, start=start, query=e, attention_query=qe, attention_key=ke, text_equal=True))
    embedding = torch.randn((7, 2560), device='cuda', dtype=torch.float16)
    original = embedding.clone()
    features = torch.randn((11, 2560), device='cuda', dtype=torch.float16)
    index = torch.full((capacity,), -1, device='cuda', dtype=torch.int32)
    index[8:12] = torch.tensor([0, 3, 7, 10], device='cuda')
    position = torch.tensor([7], device='cuda', dtype=torch.int32)
    kernel = overlay(7, 2560, capacity, 11)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): kernel(embedding, features, index, position)
    graph.replay(); torch.cuda.synchronize()
    assert torch.equal(embedding[1:5], features[[0, 3, 7, 10]])
    assert torch.equal(embedding[[0, 5, 6]], original[[0, 5, 6]])
    features.neg_(); graph.replay(); torch.cuda.synchronize()
    assert torch.equal(embedding[1:5], features[[0, 3, 7, 10]])
    shifted = torch.cat((index[1:], index.new_tensor([-1])))
    embedding.copy_(original);kernel(embedding, features, shifted, position)
    assert torch.equal(embedding[:4], features[[0, 3, 7, 10]])
    write_json(args.output/'result.json', dict(status='passed', checks=results, feature_graph_replay=True, mtp_shift=True))
    print('MRoPE / index compression / visual graph replay passed', flush=True)


if __name__ == '__main__': main()
