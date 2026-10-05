"""Compile/export a production norm and verify changed-input CUDA graph replay in a fresh cache."""
import argparse
from pathlib import Path
import torch
from kernels.operators.op02_residual_norm import residual_norm
from tools.operators.common import configure, export_kernel, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure()
    rows, hidden = 3, 513
    kernel = residual_norm(rows, hidden=hidden)
    x = torch.randn(rows, hidden, device='cuda', dtype=torch.float16)
    r = torch.randn(rows, hidden, device='cuda', dtype=torch.float32)
    w = torch.randn(hidden, device='cuda', dtype=torch.float16)
    y = torch.empty_like(x)
    residual = torch.empty_like(r)
    for _ in range(3):
        kernel(x, r, w, y, residual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(x, r, w, y, residual)
    for offset in (0, 1):
        x.add_(offset)
        graph.replay()
        torch.cuda.synchronize()
        expected = x.float() + r
        output = (expected * torch.rsqrt(expected.square().mean(-1, keepdim=True)+1e-6) * (w.float()+1)).half()
        torch.testing.assert_close(residual, expected, atol=0, rtol=0)
        torch.testing.assert_close(y, output, atol=.008, rtol=.004)
    export_kernel(kernel, args.output/'norm')
    write_json(args.output/'results.json', {'compiled': True, 'tail_hidden': hidden, 'graph_changed_input': True})


if __name__ == '__main__':
    main()
