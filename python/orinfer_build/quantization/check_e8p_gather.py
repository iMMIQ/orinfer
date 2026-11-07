"""Check shared Q4 embedding gathers and the quantized visual Linear pipeline."""

import argparse
from pathlib import Path
import numpy as np
import torch
from kernels.cuda.model import e8p_q4
from kernels.cuda.model.rotation_a8 import rotate_activation_a8
from kernels.cuda.vision.encoder import bias_activation
from orinfer_build.compiler.compile import primitive
from orinfer_build.compiler.cuda.common import configure, write_json, SEED
from orinfer_build.quantization.e8p_q4 import Weights
from orinfer_build.quantization.e8p import basis
from orinfer_build.quantization.rotation import transform
from orinfer_build.quantization.vq import e8p_selector_table


def launch(kernel, *args):
    kernel.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configure()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    vocab, k = 129, 256
    codes = rng.integers(0, 2**32, (vocab, k // 8), dtype=np.uint32)
    scales = rng.uniform(0.0001, 0.01, vocab).astype(np.float16)
    signs = rng.choice(np.array([-1, 1], np.int8), k)
    fixture = Weights(codes, scales, signs, (4, 1))
    bank = torch.from_numpy(
        np.ascontiguousarray(codes.reshape(vocab, k // 128, 16).transpose(1, 0, 2))
    ).cuda()
    ws, sign = torch.from_numpy(scales).cuda(), torch.from_numpy(signs).cuda()
    book = torch.from_numpy(e8p_selector_table(basis())).cuda()
    decoded = torch.from_numpy(fixture.integers().astype(np.float32)).cuda()
    gather = primitive(e8p_q4.embedding, 4096, vocab, k)
    results = []
    for rows in (1, 3, 17):
        ids = torch.tensor(rng.integers(-1, vocab + 1, rows), device="cuda", dtype=torch.int32)
        ids[0] = vocab - 1
        storage = torch.full((rows + 1, k), 91, device="cuda", dtype=torch.float16)
        out = storage[:rows]

        def run():
            launch(gather, ids, bank, book, ws, sign, out)

        def check():
            rotated = fixture.integers().astype(np.float32) * ws.cpu().numpy()[:, None]
            reference = transform(
                rotated, sign.cpu().numpy(), mode="block128", inverse=True
            ).astype(np.float16)
            expected = np.zeros((rows, k), np.float16)
            indices = ids.cpu().numpy()
            valid = (indices >= 0) & (indices < vocab)
            expected[valid] = reference[indices[valid]]
            torch.testing.assert_close(
                out, torch.from_numpy(expected).cuda(), rtol=0.002, atol=2e-5
            )
            assert bool((storage[rows:] == 91).all()), "Gather tail overwrite"

        run()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        ids[0] = 0
        sign.neg_()
        ws.mul_(0.5)
        graph.replay()
        check()
        ids.fill_(-1)
        graph.replay()
        check()
        ids.fill_(vocab)
        graph.replay()
        check()
        results.append(
            dict(kind="embedding", rows=rows, inverse_rotation=True, changed_input_graph=True)
        )

    for activation in ("none", "gelu", "gelu_tanh"):
        post = primitive(bias_activation, vocab, activation)
        projection = primitive(
            e8p_q4.projection,
            4096,
            vocab,
            k,
            block_m=16,
            packed_k=128,
            output_dtype="float32",
            dynamic_rows=True,
        )
        rotation = primitive(rotate_activation_a8, 4096, k, dynamic_rows=True)
        for rows in (4, 17, 129):
            x = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.1
            q = torch.empty(rows, k, device="cuda", dtype=torch.int8)
            ts = torch.empty(rows, device="cuda", dtype=torch.float16)
            product = torch.empty(rows, vocab, device="cuda")
            bias = torch.randn(vocab, device="cuda", dtype=torch.float16) * 0.1
            storage = torch.full((rows + 1, vocab), 91, device="cuda", dtype=torch.float16)
            out = storage[:rows]

            def run():
                launch(rotation, x, sign, q, ts, rows)
                launch(projection, q, bank, book, ws, ts, product, rows)
                launch(post, product, bias, out)

            def check():
                expected = ((q.float() @ decoded.T) * ws.float()[None]) * ts.float()[:, None]
                torch.testing.assert_close(product, expected, rtol=1e-5, atol=1e-6)
                expected = (expected + bias.float()).half().float()
                if activation != "none":
                    expected = torch.nn.functional.gelu(
                        expected, approximate="tanh" if activation == "gelu_tanh" else "none"
                    )
                torch.testing.assert_close(out, expected.half(), rtol=0.003, atol=1e-4)
                assert bool((storage[rows:] == 91).all()), "Vision epilogue tail overwrite"

            run()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            x.neg_()
            bias.neg_()
            graph.replay()
            check()
            results.append(
                dict(
                    kind="vision_linear", activation=activation, rows=rows, changed_input_graph=True
                )
            )
    write_json(a.output / "result.json", dict(passed=True, checks=results))
    print("PASS", len(results), flush=True)


if __name__ == "__main__":
    main()
