"""Check TP normalization, residual law, physical owner maps and Graph tails."""

import argparse
import ctypes
import json
import math
from pathlib import Path

from orinfer_build.common.files import atomic_json
from orinfer_build.common.shared_json import decode
from tests.ascend.operators.validate_sampling import reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--edge-only", action="store_true")
    a = p.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(7)
    model = decode(json.loads(a.plan.read_text()))
    c = model["sampling"]
    r = c["rejection"]
    cap, vocab, pool, ranks = c["capacity"], c["vocab"], r["pool_rows"], r["ranks"]
    parts = (vocab + 1023) // 1024
    device = "npu"
    x = torch.zeros((cap, vocab), device=device)
    q = torch.zeros((pool, vocab), device=device)
    ctl = torch.zeros((cap, 4), dtype=torch.int32, device=device)
    proposal = torch.zeros((cap, 4), dtype=torch.int32, device=device)
    active = torch.zeros(1, dtype=torch.int32, device=device)
    partial = torch.empty((cap, parts, 4), device=device)
    local = torch.empty((cap, 8), device=device)
    gathered = torch.zeros((ranks, cap, 8), device=device)
    norm = torch.full((cap, 4), 12345.0, device=device)
    candidates = torch.empty((cap, parts, 2), device=device)
    output = torch.full((cap, 2), 12345.0, device=device)
    libraries = []

    def bind(path, tensors):
        spec = json.loads(path.read_text())
        lib = ctypes.CDLL(str((path.parent / spec["library"]["file"]).resolve()))
        libraries.append(lib)
        entry = lib.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
        addresses = (ctypes.c_void_p * len(tensors))(*[v.data_ptr() for v in tensors])

        def call():
            assert (
                entry(addresses, len(tensors), cap, torch.npu.current_stream()._as_parameter_) == 0
            )

        return call

    stats = bind(a.plan.parent / r["statistics"], [x, q, ctl, proposal, active, partial])
    merge_stats = bind(
        a.plan.parent / r["merge_statistics"], [partial, x, q, ctl, proposal, active, local]
    )
    normalize = bind(a.plan.parent / r["normalize"], [gathered, proposal, active, norm])
    select = bind(a.plan.parent / r["select"], [x, q, ctl, proposal, norm, active, candidates])
    merge = bind(a.plan.parent / c["merge"], [candidates, active, output])
    stats()
    merge_stats()
    normalize()
    select()
    merge()
    torch.npu.synchronize()
    first, second = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    with torch.npu.graph(first):
        stats()
        merge_stats()
    with torch.npu.graph(second):
        normalize()
        select()
        merge()

    def inputs(n, rank, target, draft, controls, proposals):
        x[:n].copy_(target)
        q.copy_(draft)
        controls = controls.clone()
        controls[:, 2] = rank * vocab
        ctl[:n].copy_(controls)
        proposal[:n].copy_(proposals)
        active.fill_(n)
        return controls

    checks = []
    if not a.edge_only:
        for phase, n in enumerate([1, 17, 96, cap]):
            # Other shards are synthesized from an independent FP64 reduction;
            # ranks 0 and 7 execute the real device statistics and selection.
            target = torch.randn(n, ranks * vocab)
            draft = torch.randn(pool, ranks * vocab)
            controls = torch.randint(-(1 << 31), (1 << 31) - 1, (n, 4), dtype=torch.int32)
            inverse = torch.full((n,), 1.0 / (0.7 if phase & 1 else 1.0))
            controls[:, 3] = inverse.view(torch.int32)
            proposals = torch.zeros((n, 4), dtype=torch.int32)
            proposals[:, 0] = torch.arange(n) * 17 % pool
            proposals[:, 1] = torch.arange(n) * 733 % (ranks * vocab)
            proposals[:, 2] = torch.rand(n).clamp_min(1e-7).log().view(torch.int32)
            proposals[::7, 0] = -1  # Bonus/fresh owners never read stale pool rows.
            qs = draft[proposals[:, 0].clamp_min(0)]
            lp = (target.double() * inverse.double()[:, None]).log_softmax(-1)
            lq = (qs.double() * inverse.double()[:, None]).log_softmax(-1)
            delta = (
                lp.gather(1, proposals[:, 1, None])[:, 0]
                - lq.gather(1, proposals[:, 1, None])[:, 0]
            )
            accepted = proposals[:, 2].view(torch.float32).double() < delta.clamp_max(0)
            accepted &= proposals[:, 0] >= 0
            all_stats = torch.zeros((ranks, cap, 8))
            for rank in range(ranks):
                left, right = rank * vocab, (rank + 1) * vocab
                for kind, values in enumerate([target[:, left:right], qs[:, left:right]]):
                    scaled = values * inverse[:, None]
                    maximum = scaled.max(-1).values
                    all_stats[rank, :n, kind * 2] = maximum
                    all_stats[rank, :n, kind * 2 + 1] = (scaled - maximum[:, None]).exp().sum(-1)
                    all_stats[rank, :n, 4 + kind] = -float("inf")
                    own = (
                        (proposals[:, 0] >= 0)
                        & (proposals[:, 1] >= left)
                        & (proposals[:, 1] < right)
                    )
                    positions = own.nonzero()[:, 0]
                    all_stats[rank, positions, 4 + kind] = scaled[
                        positions, proposals[positions, 1] - left
                    ]
            for rank in [0, ranks - 1]:
                left, right = rank * vocab, (rank + 1) * vocab
                control = inputs(
                    n, rank, target[:, left:right], draft[:, left:right], controls, proposals
                )
                first.replay()
                torch.npu.synchronize()
                # Replace the corresponding CPU shard with real device output.
                gathered.copy_(all_stats)
                gathered[rank, :n].copy_(local[:n])
                norm.fill_(12345.0)
                output.fill_(12345.0)
                second.replay()
                torch.npu.synchronize()
                norms = norm[:n].cpu()
                assert not bool(norms[:, 3].any())
                assert torch.equal(norms[:, 2].bool(), accepted)
                torch.testing.assert_close(
                    norms[:, 0].double(),
                    (target.double() * inverse.double()[:, None]).logsumexp(-1),
                    atol=5e-5,
                    rtol=2e-6,
                )
                residual = torch.log(
                    (lp[:, left:right].exp() - lq[:, left:right].exp()).clamp_min(0)
                )
                score = torch.where(
                    (proposals[:, 0] >= 0)[:, None],
                    residual,
                    target[:, left:right].double() * inverse.double()[:, None],
                )
                score -= torch.log(-torch.log(reference(torch, control, vocab).double()))
                expected = score.argmax(-1)
                actual = output[:n].cpu()
                for row in range(n):
                    if accepted[row]:
                        token = int(proposals[row, 1]) - left
                        if 0 <= token < vocab:
                            assert actual[row, 0] == 0 and int(actual[row, 1]) == token
                        else:
                            assert actual[row, 0] == torch.finfo(torch.float32).min, (
                                phase,
                                rank,
                                row,
                                actual[row].tolist(),
                                norms[row].tolist(),
                                proposals[row].tolist(),
                            )
                    else:
                        assert int(actual[row, 1]) == int(expected[row]), (phase, rank, row)
                assert bool((norm[n:] == 12345.0).all())
                assert bool((output[n:] == 12345.0).all())
            checks.append(
                dict(rows=n, reverse_pool=True, tp_ranks=[0, ranks - 1], graph_changed_inputs=True)
            )

        # Statistical law across real selection on every shard. Proposal draws,
        # acceptance and correction randomness are independent.
        n = min(cap, pool)
        choices = [2, vocab + 1, ranks * vocab - 1]
        pvals, qvals = torch.tensor([0.1, 0.3, 0.6]), torch.tensor([0.7, 0.2, 0.1])
        target = torch.full((n, ranks * vocab), -80.0)
        draft = torch.full((pool, ranks * vocab), -80.0)
        target[:, choices] = pvals.log()
        draft[:, choices] = qvals.log()
        counts = torch.zeros(3, dtype=torch.int64)
        accepted_count = 0
        for batch in range(16):
            controls = torch.randint(-(1 << 31), (1 << 31) - 1, (n, 4), dtype=torch.int32)
            controls[:, 3] = torch.ones(n).view(torch.int32)
            proposals = torch.zeros((n, 4), dtype=torch.int32)
            proposals[:, 0] = torch.arange(n)
            proposals[:, 1] = torch.tensor(choices)[torch.multinomial(qvals, n, replacement=True)]
            proposals[:, 2] = torch.rand(n).clamp_min(1e-7).log().view(torch.int32)
            for rank in range(ranks):
                inputs(
                    n,
                    rank,
                    target[:, rank * vocab : (rank + 1) * vocab],
                    draft[:, rank * vocab : (rank + 1) * vocab],
                    controls,
                    proposals,
                )
                first.replay()
                gathered[rank].copy_(local)
            scores, ids = [], []
            for rank in range(ranks):
                inputs(
                    n,
                    rank,
                    target[:, rank * vocab : (rank + 1) * vocab],
                    draft[:, rank * vocab : (rank + 1) * vocab],
                    controls,
                    proposals,
                )
                second.replay()
                torch.npu.synchronize()
                result = output[:n].cpu()
                scores.append(result[:, 0])
                ids.append(result[:, 1].long() + rank * vocab)
            accepted_count += int(norm[:n, 2].sum().cpu())
            selected = torch.stack(scores).argmax(0)
            tokens = torch.stack(ids)[selected, torch.arange(n)]
            assert bool(sum(tokens == token for token in choices).all())
            for i, token in enumerate(choices):
                counts[i] += (tokens == token).sum()
        total = int(counts.sum())
        for observed, expected in zip((counts / total).tolist(), pvals.tolist(), strict=True):
            assert (
                abs(observed - expected) < 6 * math.sqrt(expected * (1 - expected) / total) + 0.003
            )
        acceptance = accepted_count / total
        assert abs(acceptance - 0.4) < 0.04
        checks.append(
            dict(samples=total, frequencies=(counts / total).tolist(), acceptance=acceptance)
        )
    # Identical laws always accept; an explicitly forced empty residual emits
    # the sentinel rather than silently adding a probability floor.
    n = 1
    target = torch.randn(n, ranks * vocab)
    draft = target.expand(pool, -1).clone()
    controls = torch.zeros((n, 4), dtype=torch.int32)
    controls[:, 3] = torch.ones(n).view(torch.int32)
    proposals = torch.tensor([[pool - 1, ranks * vocab - 1, 0, 0]], dtype=torch.int32)
    proposals[:, 2] = torch.tensor([-0.5]).view(torch.int32)
    for rank in range(ranks):
        inputs(
            n,
            rank,
            target[:, rank * vocab : (rank + 1) * vocab],
            draft[:, rank * vocab : (rank + 1) * vocab],
            controls,
            proposals,
        )
        first.replay()
        gathered[rank].copy_(local)
    second.replay()
    torch.npu.synchronize()
    assert norm[0, 2].item() == 1 and norm[0, 3].item() == 0
    assert int(output[0, 1].item()) == vocab - 1
    norm[0, 2] = 0
    select()
    merge()
    torch.npu.synchronize()
    assert output[0, 0].item() == torch.finfo(torch.float32).min

    # Bad model logits propagate a visible error on every shard.
    x[0, 0] = float("nan")
    first.replay()
    gathered[ranks - 1].copy_(local)
    second.replay()
    torch.npu.synchronize()
    assert norm[0, 3].item() == 1 and math.isinf(output[0, 0].item()), (
        partial[0, 0].cpu().tolist(),
        local[0].cpu().tolist(),
        norm[0].cpu().tolist(),
        output[0].cpu().tolist(),
    )
    # Fresh greedy rows neither read stale Q nor consume a rejection draw.
    x[0].zero_()
    x[0, vocab - 1] = 3
    q.fill_(float("nan"))
    ctl[0, 3] = 0
    proposal[0, 0] = -1
    first.replay()
    for rank in range(ranks):
        gathered[rank].copy_(local)
    second.replay()
    torch.npu.synchronize()
    assert norm[0, 3].item() == 0
    assert output[0, 0].item() == 3 and int(output[0, 1].item()) == vocab - 1
    checks.append(
        dict(
            identical_laws=True,
            empty_residual_sentinel=True,
            nonfinite_error=True,
            fresh_greedy_ignores_stale_pool=True,
        )
    )
    atomic_json(a.output, dict(checks=checks, weights_unchanged=True))


if __name__ == "__main__":
    main()
