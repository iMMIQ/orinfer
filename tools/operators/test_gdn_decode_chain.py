"""Offline composition test of op07/08/09/10/17; no model TPS claim."""
import argparse
import shutil
from pathlib import Path
import torch
import tilelang.language as T
from safetensors import safe_open
from common import configure, error, benchmark, identity, write_json, environment
from op07_gdn_ab import MODEL, input_rows
from op08_gdn_conv_prep import reference as conv_reference
from op17_gdn_gated_norm import reference as norm_reference
from gdn_reference import recurrent
from kernels.operators import op07_gdn_ab as proj
from kernels.operators import op08_gdn_conv_prep as conv
from kernels.operators import op09_gdn_gates as gates
from kernels.operators import op10_gdn_recurrent as core
from kernels.operators import op17_gdn_gated_norm as norm

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    out = Path(parser.parse_args().output); out.mkdir(parents=True, exist_ok=True)
    configure()
    report = {'environment': environment(), 'cases': [], 'scope':
              'GDN decode component chain; QKV/Z and entering state synthetic; no output projection or complete model'}
    frozen = out / 'dependencies'; frozen.mkdir()
    report['sources'] = []
    for path in [ROOT / f'kernels/operators/op{unit}.py' for unit in
                 ('07_gdn_ab', '08_gdn_conv_prep', '09_gdn_gates', '10_gdn_recurrent', '17_gdn_gated_norm')]:
        dest = frozen / path.name; shutil.copyfile(path, dest)
        report['sources'].append(identity(dest))
    with safe_open(str(MODEL), framework='pt', device='cpu') as model:
        prefix = 'model.language_model.layers.0.linear_attn.'
        wab = torch.cat([model.get_tensor(prefix + f'in_proj_{part}.weight') for part in ('a', 'b')]).half().cuda()
        wc = model.get_tensor(prefix + 'conv1d.weight').reshape(10240, 4).half().cuda()
        al, dt = [model.get_tensor(prefix + name).float().cuda() for name in ('A_log', 'dt_bias')]
        wn = model.get_tensor(prefix + 'norm.weight').half().cuda()
    projection = proj.gdn_ab_simt(T.dynamic('M'), output_dtype='float32')
    gate = gates.gdn_gates(dtype='float32', packed_ab=True)
    for fused in (False, True):
        mode = {'normalize_round_fp16': not fused, 'q_scale': 128**-.5 if fused else 1.,
                'qk_output_dtype': 'float32' if fused else 'float16', 'conv_product_round_fp16': True}
        ck = conv.gdn_conv_decode(**mode)
        rk = core.gdn_recurrent(q_scale=1. if fused else 128**-.5,
                               qk_dtype=mode['qk_output_dtype'], value_tile=32)
        nk = norm.gdn_gated_norm(x_dtype='float16')
        for batch in (1, 3, 8):
            cpu, origin = input_rows(0, batch)
            hidden = cpu.cuda()
            x = torch.randn((batch, 1, 10240), device='cuda', dtype=torch.float16)
            z = torch.randn((batch, 48, 128), device='cuda', dtype=torch.float16)
            hi = torch.randn((batch, 3, 10240), device='cuda', dtype=torch.float16)
            pos = (torch.arange(batch, device='cuda', dtype=torch.int32) * 3)
            lengths = torch.ones_like(pos)
            q = torch.empty((batch, 16, 1, 128), device='cuda', dtype=getattr(torch, mode['qk_output_dtype']))
            k = torch.empty_like(q); v = torch.empty((batch, 48, 1, 128), device='cuda', dtype=torch.float16)
            ho = torch.empty_like(hi); po = torch.empty_like(pos)
            ab = torch.empty((batch, 96), device='cuda')
            g = torch.empty((batch, 48), device='cuda'); beta = torch.empty_like(g)
            si = torch.randn((batch, 48, 128, 128), device='cuda') * .05
            so = torch.empty_like(si)
            raw = torch.empty((batch, 48, 128), device='cuda', dtype=torch.float16)
            y = torch.empty_like(raw)
            def run():
                stream = torch.cuda.current_stream().cuda_stream
                projection(hidden, wab, ab, stream=stream)
                gates.launch(gate, ab, ab, al, dt, g, beta, stream=stream)
                conv.launch(ck, x, wc, hi, lengths, pos, q, k, v, ho, po, stream=stream)
                core.launch(rk, q.view(batch, 16, 128), k.view(batch, 16, 128),
                            v.view(batch, 48, 128), g, beta, si, so, raw, stream=stream)
                norm.launch(nk, raw, z, wn, y, stream=stream)
            def reference():
                rq, rkv, rv, rh, rp, _ = conv_reference(x, wc, hi, lengths, pos, mode)
                rab = hidden.float() @ wab.float().T
                rg = -al.exp() * torch.nn.functional.softplus(rab[:, :48] + dt)
                rb = torch.sigmoid(rab[:, 48:])
                ro, rs = recurrent(rq, rkv, rv, rg.unsqueeze(-1), rb.unsqueeze(-1), si,
                                   q_scale=1. if fused else 128**-.5)
                ry = norm_reference(ro.squeeze(2).half(), z, wn).half()
                return ry, rs, rh, rp
            def check():
                ry, rs, rh, rp = reference()
                ey, es = error(y, ry), error(so, rs)
                assert ey['finite'] and ey['relative_l2'] < .002, ey
                assert es['finite'] and es['relative_l2'] < .001, es
                assert torch.equal(ho, rh) and torch.equal(po, rp)
                return {'output': ey, 'state': es, 'history_position_bitexact': True}
            run(); original = check()
            timing, graph = benchmark(run, repetitions=5, calls_per_replay=4)
            saved = [tensor.clone() for tensor in (hidden, x, z, hi, pos, si)]
            hidden.mul_(.8); x.mul_(.5); z.add_(.2); hi.mul_(.75); pos.add_(2); si.add_(.05)
            for tensor in (y, raw, so, ho, q, k, v, g, beta): tensor.fill_(float('nan'))
            po.fill_(-1); graph.replay(); torch.cuda.synchronize(); changed = check()
            for tensor, value in zip((hidden, x, z, hi, pos, si), saved): tensor.copy_(value)
            graph.replay(); torch.cuda.synchronize(); restored = check()
            report['cases'].append({'fused_normalization': fused, 'batch': batch, 'input': origin,
                'original': original, 'changed_graph': changed, 'restored_graph': restored, 'timing': timing})
    report['status'] = 'passed'; write_json(out / 'results.json', report)


if __name__ == '__main__':
    main()
