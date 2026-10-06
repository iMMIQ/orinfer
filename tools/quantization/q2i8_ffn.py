"""Calibrate Q2I8 expert FFNs and measure held-out local reconstruction error.

Samples are actual routed activations with prompt IDs and a calibration mask.
Reference: original BF16-derived weights evaluated with FP32 activations/math.
This is a layer diagnostic, not token/model-quality acceptance. The optional
community comparison accepts already dequantized weights including any scales.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.quantization.q2i8 import activation_importance, quantize, save
from tools.quantization.reconstruction import quantize_reconstruction


def a8(x, group=None):
    """Match the online FP16 boundary, FP16 scale and round-to-nearest-even."""
    x = np.asarray(x).astype(np.float16).astype(np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('A8 inputs must be finite FP16-representable matrices')
    group = x.shape[1] if group is None else group
    if type(group) is not int or group <= 0 or x.shape[1] % group:
        raise ValueError('A8 group must divide the channel dimension')
    grouped = x.reshape(len(x),-1,group)
    maximum = np.abs(grouped).max(-1)
    scale = np.where(maximum > 0,np.maximum(maximum/127,2**-24),1).astype(np.float16)
    code = np.clip(np.rint(grouped/scale.astype(np.float32)[...,None]),-127,127).astype(np.int8)
    return code.reshape(x.shape),scale


def a8_value(x, group=None):
    q,s = a8(x,group)
    return (q.reshape(len(q),s.shape[1],-1).astype(np.float32)*s.astype(np.float32)[...,None]).reshape(q.shape)


def swiglu(gu):
    g,u = np.split(np.asarray(gu,dtype=np.float32),2,axis=-1)
    e = np.exp(-np.abs(g))
    sigmoid = np.where(g >= 0,1/(1+e),e/(1+e))
    return (g*sigmoid)*u


def project(x, weights):
    q,s = a8(x)
    # Exact integer dot oracle: bounded signed-byte products/sums fit FP64.
    dot = (q.astype(np.float64)@weights.integer_weights().astype(np.float64).T).astype(np.float32)
    return ((dot*weights.scales.astype(np.float32)[None,:])*s.astype(np.float32)).astype(np.float16)


def ffn(x, gate_up, down):
    return project(swiglu(project(x,gate_up)).astype(np.float16),down).astype(np.float32)


def floating_ffn(x, gate_up, down, *, activation_group=None):
    if activation_group is None:
        return swiglu(x.astype(np.float32)@gate_up.T)@down.T
    # Community Q2_0 uses group64 A8. This floating diagnostic does not measure
    # native-kernel performance or bitwise accumulation order.
    group = None if activation_group == 'row' else activation_group
    gu = (a8_value(x,group)@gate_up.T).astype(np.float16)
    return (a8_value(swiglu(gu),group)@down.T).astype(np.float16).astype(np.float32)


def error(actual, expected):
    delta = actual.astype(np.float64)-expected.astype(np.float64)
    a,b = actual.astype(np.float64).ravel(),expected.astype(np.float64).ravel()
    denom = np.linalg.norm(b)
    return {'relative_l2':float(np.linalg.norm(delta)/max(denom,1e-30)),
            'rms_abs':float(np.sqrt(np.mean(delta*delta))),
            'max_abs':float(np.abs(delta).max()),
            'cosine':float(np.dot(a,b)/max(np.linalg.norm(a)*denom,1e-30))}


def samples(path):
    with np.load(path,allow_pickle=False) as data:
        required = {'activations','routed_ids','prompt_ids','calibration_mask','expert_ids'}
        if not required.issubset(data.files):
            raise ValueError('Samples require '+', '.join(sorted(required)))
        x,ids,pid,mask,experts = [data[k] for k in ('activations','routed_ids','prompt_ids','calibration_mask','expert_ids')]
    t = len(x)
    if x.ndim != 2 or min(x.shape) <= 0 or x.dtype.kind != 'f' or not np.isfinite(x).all():
        raise ValueError('Invalid actual activation samples')
    if ids.ndim != 2 or ids.shape[0] != t or ids.dtype.kind not in 'iu' or (ids < 0).any():
        raise ValueError('Invalid routed expert IDs')
    if ids.shape[1] < 1 or (np.diff(np.sort(ids,axis=1),axis=1) == 0).any():
        raise ValueError('Each token must route to distinct experts')
    if pid.shape != (t,) or pid.dtype.kind not in 'iu' or mask.shape != (t,) or mask.dtype != np.bool_:
        raise ValueError('Invalid prompt partition')
    if not mask.any() or mask.all() or set(pid[mask]) & set(pid[~mask]):
        raise ValueError('Calibration/evaluation must have disjoint nonempty prompt sets')
    if experts.ndim != 1 or not len(experts) or experts.dtype.kind not in 'iu' or (experts < 0).any() or len(np.unique(experts)) != len(experts):
        raise ValueError('Invalid fixture expert IDs')
    return x,ids,pid,mask,experts


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda:reader.read(4*1024*1024),b''):
            digest.update(block)
    return {'path':str(path),'sha256':digest.hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate-up',type=Path,required=True,help='Original BF16-derived float [E,2F,H] .npy')
    p.add_argument('--down',type=Path,required=True,help='Original BF16-derived float [E,H,F] .npy')
    p.add_argument('--samples',type=Path,required=True,help='Actual routed activation .npz; see samples() contract')
    p.add_argument('--activation-source',required=True,help='Producer checkpoint/revision; explicitly identify quantized producers')
    p.add_argument('--calibration-method',choices=('diagonal','reconstruction'),default='reconstruction')
    p.add_argument('--community-gate-up',type=Path)
    p.add_argument('--community-down',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    x,ids,pid,mask,experts = samples(a.samples)
    gu,down = [np.load(path,mmap_mode='r',allow_pickle=False) for path in (a.gate_up,a.down)]
    if gu.ndim != 3 or down.shape != (len(experts),x.shape[1],gu.shape[1]//2) or gu.shape != (len(experts),down.shape[2]*2,x.shape[1]):
        raise ValueError('Fixture geometry does not match activations/expert IDs')
    if gu.dtype.kind != 'f' or down.dtype.kind != 'f':
        raise ValueError('Original weights must be floating')
    if bool(a.community_gate_up) != bool(a.community_down):
        raise ValueError('Both community weight families must be supplied')
    community = None
    if a.community_gate_up:
        community = [np.load(path,mmap_mode='r',allow_pickle=False) for path in (a.community_gate_up,a.community_down)]
        if community[0].shape != gu.shape or community[1].shape != down.shape:
            raise ValueError('Community fixture geometry mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    paths = [a.gate_up,a.down,a.samples]+([a.community_gate_up,a.community_down] if community else [])
    provenance = {'inputs':[identity(path) for path in paths],'activation_source':a.activation_source,'seed':20261002}
    report = {'scope':__doc__,'provenance':provenance,'tokens':len(x),'prompts':len(np.unique(pid)),
              'calibration_prompts':sorted(int(i) for i in np.unique(pid[mask])),
              'evaluation_prompts':sorted(int(i) for i in np.unique(pid[~mask])),
              'cases':[],'complete':False}
    for index,expert in enumerate(experts):
        selected = (ids == expert).any(1)
        train,test = x[selected & mask],x[selected & ~mask]
        case = {'expert':int(expert),'calibration_rows':len(train),'evaluation_rows':len(test)}
        # Experts without routed training samples remain explicitly weight-only.
        wg = quantize(gu[index])
        wd = quantize(down[index])
        cg,cd = wg,wd
        if len(train):
            def fit(weight,actual):
                if a.calibration_method == 'reconstruction':
                    return quantize_reconstruction(weight,actual)
                return quantize(weight,importance=activation_importance(actual))
            cg = fit(gu[index],a8_value(train))
            intermediate = swiglu(project(train,cg)).astype(np.float16)
            cd = fit(down[index],a8_value(intermediate))
        case['calibration'] = a.calibration_method+' using actual routed activations; down uses quantized gate output' if len(train) else 'weight-only; no routed training samples'
        for family,w in [('gate_up',cg),('down',cd)]:
            save(a.output/f'{family}-expert{int(expert)}.safetensors',w,provenance={**provenance,'expert':int(expert),'calibration':case['calibration']})
        case['weight_bytes'] = cg.nbytes+cd.nbytes
        if len(test):
            expected = floating_ffn(test,gu[index],down[index])
            candidates = {'q2i8_weight_only':ffn(test,wg,wd),'q2i8_calibrated':ffn(test,cg,cd),
                          'bf16_weights_a8_row':floating_ffn(test,gu[index],down[index],activation_group='row')}
            if community:
                candidates['community_q2_0_a8_group64'] = floating_ffn(test,community[0][index],community[1][index],activation_group=64)
            if not np.isfinite(expected).all() or any(not np.isfinite(y).all() for y in candidates.values()):
                raise ValueError('Nonfinite FFN reconstruction')
            case['heldout_output_error'] = {name:error(y,expected) for name,y in candidates.items()}
        report['cases'].append(case)
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(case),flush=True)
    report['complete'] = True
    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
