"""GPU checks for Qwen strided Q4 outputs, block128 rotations and dynamic KV."""

import argparse
import json
from pathlib import Path
import numpy as np
from orinfer_build.common.gpu import gpu_lock
from orinfer_build.compiler.compile import primitive


def run():
    import torch
    from orinfer_build.tuning.projection_tuning import ProjectionTuner, make_prim
    from kernels.cuda.model.rotation_a8 import rotate_activation_a8
    from orinfer_build.validation.attention_reference import attention_prefill_int8
    from orinfer_build.quantization.vq import rotate
    from orinfer_build.compiler.cuda.common import SEED

    torch.manual_seed(SEED)
    checks = []
    for kind, k in [("q4", 128), ("q4", 17408), ("w8", 128), ("w8", 17408)]:
        spec = dict(
            kind=kind,
            rows=17,
            outputs=33,
            inputs=k,
            dtype="float16",
            dynamic=True,
            output_stride=64,
        )
        for algorithm, bm, bn in [("dp4a", 4, 8), ("tensor_core", 16, 64)]:
            import tilelang

            config = dict(algorithm=algorithm, block_m=bm, block_n=bn, block_k=128, stages=2)
            if kind == "w8":
                config.pop("stages")
            kernel = tilelang.compile(
                make_prim(spec, config),
                out_idx=[],
                execution_backend="nvrtc",
                target={"kind": "cuda", "arch": "sm_87"},
            )
            for live in [1, 5, 17]:
                ProjectionTuner._validate(kernel, spec, config, live)
            checks.append(
                dict(
                    kind="strided_" + kind,
                    k=k,
                    algorithm=algorithm,
                    rows=[1, 5, 17],
                    exact=True,
                    graph=True,
                )
            )
    for k in [5120, 17408]:
        x = torch.randn((5, k), device="cuda", dtype=torch.float16) * 0.02
        signs = torch.randint(0, 2, (k,), device="cuda", dtype=torch.int8) * 2 - 1
        q = torch.empty_like(x, dtype=torch.int8)
        s = torch.empty(5, device="cuda", dtype=torch.float16)
        kernel = primitive(rotate_activation_a8, 4096, k, dynamic_rows=True)
        kernel.adapter.func(x, signs, q, s, 5, stream=torch.cuda.current_stream().cuda_stream)
        transformed = (
            rotate(x.float().cpu().numpy(), signs.cpu().numpy())
            .astype(np.float16)
            .astype(np.float32)
        )
        scale = np.maximum(np.max(np.abs(transformed), axis=1) / 127, 2**-24).astype(np.float16)
        expected = np.clip(
            np.rint(transformed / scale.astype(np.float32)[:, None]), -127, 127
        ).astype(np.int8)
        actual = q.cpu().numpy()
        max_code_error = int(np.abs(actual.astype(np.int32) - expected).max())
        assert np.array_equal(s.cpu().numpy(), scale), (k, "scale")
        assert max_code_error <= 1, (k, max_code_error)
        checks.append(dict(kind="rotation_a8", k=k, rows=5, max_code_error=max_code_error))
    dynamic = primitive(attention_prefill_int8, 1, 4096, 128, dynamic_rows=True)
    for live in [1, 17]:
        q = torch.randn((1, live, 24, 256), device="cuda", dtype=torch.float16) * 0.1
        gate = torch.randn_like(q)
        kv = [
            torch.randint(-127, 128, (1, 128, 4, 256), device="cuda", dtype=torch.int8)
            for _ in range(2)
        ]
        scales = [
            torch.full((1, 128, 4, 4), 0.001, device="cuda", dtype=torch.float16) for _ in range(2)
        ]
        pos = (torch.arange(live, device="cuda", dtype=torch.int32) + 3)[None]
        lengths = torch.tensor([live + 3], device="cuda", dtype=torch.int32)
        storage = torch.full((1, live + 1, 24, 256), 91, device="cuda", dtype=torch.float16)
        out = storage[:, :live]
        ref = torch.empty_like(out)
        args = [q, *kv, gate, pos, lengths, out, *scales]
        dynamic.adapter.func(*args, live, stream=torch.cuda.current_stream().cuda_stream)
        static = primitive(attention_prefill_int8, 1, live, 128)
        static.adapter.func(*args[:6], ref, *scales, stream=torch.cuda.current_stream().cuda_stream)
        assert torch.equal(out, ref)
        assert bool(torch.all(storage[:, live:] == 91))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dynamic.adapter.func(*args, live, stream=torch.cuda.current_stream().cuda_stream)
        q.neg_()
        out.fill_(91)
        graph.replay()
        static.adapter.func(*args[:6], ref, *scales, stream=torch.cuda.current_stream().cuda_stream)
        assert torch.equal(out, ref)
        assert bool(torch.all(storage[:, live:] == 91))
        checks.append(dict(kind="dynamic_int8_attention", rows=live, exact=True, graph=True))
    from kernels.cuda.model.gdn_sequence import _gdn_sequence, gdn_commit

    from kernels.cuda.model.speculative_factors import save_gdn_factors

    verifier = primitive(
        _gdn_sequence, 8, value_tile=16, threads=32, compact=True, dynamic_rows=True
    )
    ordinary = primitive(
        _gdn_sequence,
        4096,
        in_place=True,
        value_tile=16,
        threads=32,
        prefetch_qk=True,
        dynamic_rows=True,
    )
    save = primitive(save_gdn_factors, 8, dynamic_rows=True)
    commit = primitive(gdn_commit, 8, value_tile=16, dynamic_rows=True)
    accepted = torch.ones(1, device="cuda", dtype=torch.int32)
    scratch = torch.empty(1, device="cuda")
    for live in [2, 3, 5, 7, 8]:
        q = torch.randn((16, live, 128), device="cuda", dtype=torch.float16) * 0.02
        k = torch.randn_like(q) * 0.02
        v = torch.randn((48, live, 128), device="cuda", dtype=torch.float16) * 0.02
        g = -torch.rand((live, 48), device="cuda") * 0.1
        beta = torch.rand_like(g)
        initial = torch.randn((48, 128, 128), device="cuda") * 0.001
        state = initial.clone()
        update = torch.empty((8, 48, 128), device="cuda")
        output = torch.empty((8, 6144), device="cuda", dtype=torch.float16)
        saved_k = torch.full((16 * 8 * 128 + 128,), 91, device="cuda", dtype=torch.float16)
        saved_g = torch.full((8 * 48 + 48,), 91, device="cuda", dtype=torch.float32)
        verifier.adapter.func(
            q,
            k,
            v,
            g,
            beta,
            state,
            update,
            output,
            live,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        assert torch.equal(state, initial), "Verification mutated the uncommitted GDN state"
        save.adapter.func(
            k, g, saved_k, saved_g, live, stream=torch.cuda.current_stream().cuda_stream
        )
        assert torch.equal(saved_k[: 16 * live * 128].reshape_as(k), k)
        assert torch.equal(saved_g[: live * 48].reshape_as(g), g)
        assert bool(torch.all(saved_k[16 * live * 128 :] == 91))
        assert bool(torch.all(saved_g[live * 48 :] == 91))
        max_state_error = 0.0
        for count in range(1, live + 1):
            state.copy_(initial)
            accepted.fill_(count)
            commit.adapter.func(
                saved_k,
                saved_g,
                update,
                accepted,
                state,
                live,
                stream=torch.cuda.current_stream().cuda_stream,
            )
            reference = initial.clone()
            expected = torch.empty((count, 6144), device="cuda", dtype=torch.float16)
            ordinary.adapter.func(
                q[:, :count].contiguous(),
                k[:, :count].contiguous(),
                v[:, :count].contiguous(),
                g[:count],
                beta[:count],
                reference,
                scratch,
                expected,
                count,
                stream=torch.cuda.current_stream().cuda_stream,
            )
            error = float((state - reference).abs().max())
            max_state_error = max(max_state_error, error)
            assert error < 1e-6, (live, count, error)
            torch.testing.assert_close(output[:count], expected, rtol=1e-3, atol=1e-6)
        checks.append(
            dict(
                kind="compact_gdn_commit",
                rows=live,
                accepted_prefixes=list(range(1, live + 1)),
                max_state_error=max_state_error,
            )
        )
    torch.cuda.synchronize()
    return dict(seed=SEED, checks=checks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gpu-lock", type=Path, default=Path("artifacts/gpu-experiment.lock"))
    p.add_argument("--gpu-lock-held", action="store_true")
    a = p.parse_args()
    with gpu_lock(None if a.gpu_lock_held else a.gpu_lock):
        report = run()
    if a.output.suffix != ".json":
        a.output = a.output / "result.json"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", len(report["checks"]), flush=True)


if __name__ == "__main__":
    main()
