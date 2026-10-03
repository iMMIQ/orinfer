"""Explicit FP16 product policies, FP32 state, paired current factored baseline.

Declared-policy reference verifies implementation. FP32 drift is diagnostic,
not a token-identity or model-quality acceptance gate. Synthetic inputs only.
"""
import argparse
import gc
from pathlib import Path
import shutil

import torch
from common import benchmark, configure, environment, error, export_kernel, identity, write_json
from gdn_reference import chunk_scan, chunk_output, chunk_matrices, expand_heads
from tools.operators.op15_gdn_chunk_state import make_inputs, outputs, invoke
from kernels.model.gdn_factored import gdn_chunk_state_factored, gdn_chunk_output_factored


POLICIES = {
    'full': (True, True, True, True),
    'high': (False, False, False, False),
    'retain-update': (False, True, False, False),
    'retain-cross': (False, False, True, False),
}


def split(value):
    high = value.half().float()
    return high, (value - high).half().float()


def declared_scan(inputs, residual_compensation, update_compensation, actual_entering=None):
    k, g, w, u, sin = inputs
    k = expand_heads(k, u.shape[1])
    state = sin.clone()
    entering, residuals = [], []
    for c in range(g.shape[2]):
        # Check every transition against exactly the history consumed by the
        # candidate. FP32 reduction/FMA differences can flip a subsequent
        # FP16 operand at a half tie; avoid classifying propagation of that
        # declared rounding boundary as an indexing/implementation defect.
        entering.append(state.clone())
        if actual_entering is not None:
            state = actual_entering[:, :, c]
        whi, wlo = split(w[:, :, c])
        shi, slo = split(state)
        predicted = whi @ shi
        if residual_compensation:
            predicted = (wlo @ shi + whi @ slo) + predicted
        r = u[:, :, c] - predicted
        residuals.append(r)
        gc = g[:, :, c]
        last = gc[..., -1]
        scaled = r * (last[..., None] - gc).exp()[..., None]
        rhi, rlo = split(scaled)
        kt = k[:, :, c].transpose(-1, -2)
        update = kt @ rhi
        if update_compensation:
            update = kt @ rlo + update
        state = last.exp()[..., None, None] * state + update
    return torch.stack(entering, dim=2), torch.stack(residuals, dim=2), state


def declared_output(q, g, qk, entering, r, cross_compensation, local_compensation):
    q = expand_heads(q, r.shape[1])
    shi, slo = split(entering)
    khi, klo = split(qk)
    rhi, rlo = split(r)
    y = q @ shi
    if cross_compensation:
        y = q @ slo + y
    y = (y * (128 ** -.5)) * g.exp()[..., None]
    # Match product accumulation order, while Torch dot reduction is separate.
    if local_compensation:
        y = y + klo @ rhi
        y = y + khi @ rlo
    return y + khi @ rhi


def state_metrics(actual, expected, implementation=False):
    metrics = {name: error(a, e) for name, a, e in
               zip(('entering', 'residual', 'final'), actual, expected)}
    assert all(m['finite'] for m in metrics.values()), metrics
    if implementation:
        assert all(m['relative_l2'] < 5e-5 for m in metrics.values()), metrics
    return metrics


def state_semantics(kernel, inputs, out):
    chunks = inputs[1].shape[2]
    cut = max(1, chunks // 2)
    first = [x[:, :, :cut].contiguous() for x in inputs[:4]] + [inputs[-1]]
    first_out = outputs(first)
    invoke(kernel, first, first_out)
    second = [x[:, :, cut:].contiguous() for x in inputs[:4]] + [first_out[-1].clone()]
    second_out = outputs(second)
    invoke(kernel, second, second_out)
    torch.cuda.synchronize()
    assert torch.equal(first_out[0], out[0][:, :, :cut])
    assert torch.equal(first_out[1], out[1][:, :, :cut])
    assert torch.equal(second_out[0], out[0][:, :, cut:])
    assert torch.equal(second_out[1], out[1][:, :, cut:])
    assert torch.equal(second_out[2], out[2])
    checkpoint = second[-1].clone()
    branch = [x.clone() for x in second]
    branch[3].add_(.013)
    branch_out = outputs(branch)
    invoke(kernel, branch, branch_out)
    torch.cuda.synchronize()
    assert torch.equal(second[-1], checkpoint)
    assert not torch.equal(branch_out[-1], second_out[-1])
    invoke(kernel, second, second_out)
    torch.cuda.synchronize()
    assert torch.equal(second_out[-1], out[-1])
    for b in range(inputs[-1].shape[0]):
        isolated = [x[b:b+1].contiguous() for x in inputs]
        isolated_out = outputs(isolated)
        invoke(kernel, isolated, isolated_out)
        torch.cuda.synchronize()
        assert all(torch.equal(a, z[b:b+1]) for a, z in zip(isolated_out, out))
    return dict(resume_bitwise=True, request_isolation_bitwise=True,
                checkpoint_immutable=True, branch_changes_state=True, restore_bitwise=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--policy', choices=POLICIES, default='high')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--validation-only', action='store_true')
    args = ap.parse_args()
    configure()
    policy = POLICIES[args.policy]
    state = gdn_chunk_state_factored(compensate_residual=policy[0], compensate_update=policy[1])
    output = gdn_chunk_output_factored(compensate_cross=policy[2], compensate_local=policy[3])
    paired_state = gdn_chunk_state_factored()
    paired_output = gdn_chunk_output_factored()
    sources = ('kernels/model/gdn_factored.py', 'tools/operators/gdn_reference.py',
               'tools/operators/op15_gdn_chunk_state.py', 'kernels/operators/op15_gdn_chunk_state.py')
    report = dict(status='running', environment=environment(), policy=args.policy,
                  flags=list(policy), persistent_state_dtype='float32', accumulation_dtype='float32',
                  scope='Synthetic complete state/output chain; declared-policy implementation '
                        'reference and FP32 diagnostic, not model quality or formal TPS', cases=[],
                  sources=[identity(p) for p in sources])
    for path in sources:
        dest = args.output / 'dependencies' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    for name, kernel in (('state', state), ('output', output),
                         ('paired-state', paired_state), ('paired-output', paired_output)):
        export_kernel(kernel, args.output/name)

    def measure(run):
        if not args.validation_only:
            return benchmark(run, repetitions=4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        return dict(median_ms=None, status='validation_only'), graph

    specs = [(1, 512, 'random'), (1, 513, 'random')]
    if not args.quick:
        specs += [(1, 2048, 'random'), (1, 8192, 'random'),
                  (3, 129, 'random'), (1, 129, 'beta0'), (1, 129, 'strong_decay')]
    for batch, tokens, mode in specs:
        inputs, auxiliary = make_inputs(batch, tokens, 64, mode)
        initial = inputs[-1].clone()
        out = outputs(inputs)
        invoke(state, inputs, out)
        torch.cuda.synchronize()
        reference = declared_scan(inputs, *policy[:2], actual_entering=out[0])
        full_reference = chunk_scan(*inputs)
        rec = dict(B=batch, T=tokens, mode=mode,
                   state_implementation_reference='Each transition conditioned on actual '
                       'entering state; initial input and resulting next state checked, '
                       'resume/branch/isolation independently checked',
                   state_implementation=state_metrics(out, reference, implementation=True),
                   state_fp32_drift=state_metrics(out, full_reference))
        assert torch.equal(initial, inputs[-1])
        if tokens % 64:
            assert bool((out[1][:, :, -1, tokens % 64:] == 0).all())
        rec['input_state_immutable'] = True
        rec['state_timing'], graph = measure(lambda: invoke(state, inputs, out))
        if tokens <= 513 and mode == 'random':
            rec['state_semantics'] = state_semantics(state, inputs, out)
        if tokens == 512:
            rec['state_graph'] = {}
            restored = [x.clone() for x in out]
            for index, name in enumerate(('K', 'G', 'W', 'U', 'Sin')):
                original = inputs[index].clone()
                if index <= 2:
                    inputs[index].mul_(.75)
                else:
                    inputs[index].add_(.017)
                for value in out:
                    value.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                expected = declared_scan(inputs, *policy[:2], actual_entering=out[0])
                rec['state_graph'][name] = state_metrics(out, expected, implementation=True)
                assert not torch.equal(out[-1], restored[-1])
                inputs[index].copy_(original)
                graph.replay()
                torch.cuda.synchronize()
                assert all(torch.equal(a, e) for a, e in zip(out, restored))
            rec['state_graph_restore_bitwise'] = True
        del graph, reference
        k, g, w, u, sin = inputs
        q = torch.nn.functional.normalize(torch.randn_like(k.float()), dim=-1).half()
        if tokens % 64:
            q[:, :, -1, tokens % 64:] = 0
        _, qk = chunk_matrices(q, k, g, auxiliary[-1])
        y = torch.empty((batch, g.shape[2]*64, 48, 128), device='cuda', dtype=torch.float16)
        def invoke_output(kernel, entering, residual):
            kernel.adapter.func(q, g, qk, entering, residual, y,
                                stream=torch.cuda.current_stream().cuda_stream)
        invoke_output(output, *out[:2])
        torch.cuda.synchronize()
        actual = y.view(batch, g.shape[2], 64, 48, 128).permute(0,3,1,2,4).contiguous()
        expected = declared_output(q, g, qk, *out[:2], *policy[2:]).half().float()
        rec['output_implementation'] = error(actual, expected)
        assert rec['output_implementation']['finite'] and rec['output_implementation']['relative_l2'] < .001
        rec['output_fp32_drift_on_candidate_state'] = error(
            actual, chunk_output(q, g, qk, *out[:2], q_scale=128**-.5))
        rec['chain_fp32_drift'] = error(
            actual, chunk_output(q, g, qk, *full_reference[:2], q_scale=128**-.5))
        assert rec['chain_fp32_drift']['finite']
        rec['output_timing'], graph = measure(lambda: invoke_output(output, *out[:2]))
        saved_q, saved_qk = q.clone(), qk.clone()
        original_y = y.clone()
        q.zero_(); qk.zero_(); y.fill_(float('nan'))
        graph.replay(); torch.cuda.synchronize()
        assert bool((y == 0).all())
        q.copy_(saved_q); qk.copy_(saved_qk)
        graph.replay(); torch.cuda.synchronize()
        assert torch.equal(y, original_y)
        rec['output_graph_zero_restore_bitwise'] = True
        del graph, saved_q, saved_qk, original_y
        if not args.validation_only:
            paired_out = outputs(inputs)
            rec['paired_state_timing'], graph = benchmark(
                lambda: invoke(paired_state, inputs, paired_out), repetitions=4)
            del graph
            rec['paired_state_fp32_drift'] = state_metrics(paired_out, full_reference)
            rec['paired_output_timing'], graph = benchmark(
                lambda: invoke_output(paired_output, *paired_out[:2]), repetitions=4)
            del graph, paired_out
        report['cases'].append(rec)
        write_json(args.output/'result.json', report)
        print(f'T={tokens} B={batch} {mode}: state={rec["state_timing"]["median_ms"]} '
              f'output={rec["output_timing"]["median_ms"]} '
              f'FP32 state L2={rec["state_fp32_drift"]["final"]["relative_l2"]} '
              f'chain L2={rec["chain_fp32_drift"]["relative_l2"]}', flush=True)
        del inputs, auxiliary, out, full_reference, initial, k, g, w, u, sin, q, qk, y, actual, expected
        gc.collect()
    report['status'] = 'passed'
    write_json(args.output/'result.json', report)


if __name__ == '__main__':
    main()
