"""Check new GDN INT8 epilogues and offline WS against actual packed values.

Reference compute is test-only Torch FP32. No quantization quality claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import tilelang.language as T
from common import configure, error, export_kernel, environment, write_json
from kernels.model.w4a8 import gdn_qkvz_int8
from kernels.operators.op29_w4_to_temporary_w8 import w4_to_temporary_w8, launch as expand
from kernels.operators.op30_activation_quantization import activation_quantization, launch as quantize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);a=ap.parse_args()
    configure();model=json.loads(a.model.read_text());root=a.model.parent
    report={'scope':'Implementation/ABI/row metadata checks against explicit quantized math, not BF16/FP8 quality','environment':environment(),'manifest_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'row_scale_checks':[],'qkvz_checks':[]}
    buffers={b['name']:b for b in model['buffers']}
    def cpu(name):
        b=buffers[name];dtype={'u8':np.uint8,'i8':np.int8,'f16':np.float16}[b['dtype']]
        return np.memmap(root/b['data']['file'],dtype=dtype,mode='r',shape=tuple(b['shape']))
    # Spread sampled rows across representative early/late matrices and both K.
    for name in ('L0_In','L0_GateUp','L0_Down','L3_In','L63_Down'):
        p,s,z,ws=[cpu(name+suffix) for suffix in ('_P','_S','_Z','_WS')];n,khalf=p.shape;k=khalf*2
        ids=np.unique(np.r_[0,1,127,128,n-1,np.linspace(0,n-1,19,dtype=int)])
        pt=torch.from_numpy(np.array(p[ids]));st=torch.from_numpy(np.array(s[ids]));zt=torch.from_numpy(np.array(z[ids]))
        codes=torch.stack((pt&15,pt>>4),-1).reshape(len(ids),k//128,128)
        w=((codes.float()-zt.float()[:,:,None])*st.float()[:,:,None]).half()
        maximum=w.float().abs().flatten(1).amax(1)
        expected=torch.where(maximum>0,(maximum/127).clamp_min(2**-24),1.).half().numpy()
        mismatches=int((expected.view(np.uint16)!=np.array(ws[ids]).view(np.uint16)).sum())
        assert mismatches==0,(name,mismatches)
        report['row_scale_checks'].append({'weight':name,'rows':ids.tolist(),'scale_bit_mismatches':mismatches})
    p,s,z,ws=[torch.from_numpy(np.array(cpu('L0_In'+suffix))).cuda() for suffix in ('_P','_S','_Z','_WS')]
    b=torch.empty((16384,5120),device='cuda',dtype=torch.int8)
    ek=w4_to_temporary_w8();stream=torch.cuda.current_stream().cuda_stream
    expand(ek,p,s,z,ws,b,stream=stream)
    # Check selected full rows, independently calculating all actual codes.
    ids=torch.tensor([0,127,128,10239,10240,16383],device='cuda')
    codes=torch.stack((p[ids]&15,p[ids]>>4),-1).reshape(len(ids),40,128)
    w=((codes.float()-z[ids].float()[:,:,None])*s[ids].float()[:,:,None]).half().reshape(len(ids),5120)
    expected=(w.float()/ws[ids,None].float()).round().clamp(-127,127).to(torch.int8)
    report['expansion_code_mismatches']=int((b[ids]!=expected).sum());assert report['expansion_code_mismatches']==0
    input_file=Path('artifacts/experimental-vllm/activations/capture-512-0-language_model_model_layers_0_linear_attn_in_proj_qkvz.pt')
    x=torch.load(input_file,map_location='cpu',weights_only=True).half().cuda()
    aq=activation_quantization(5120);mask=torch.zeros(5120,device='cuda',dtype=torch.uint8)
    dynamic=gdn_qkvz_int8(T.dynamic('M'));fixed=gdn_qkvz_int8(512)
    export_kernel(fixed,a.output/'qkvz-fixed');export_kernel(dynamic,a.output/'qkvz-dynamic')
    for m in (511,512,513):
        xm=torch.cat((x,x[:1]),0)[:m].contiguous()
        qa=torch.empty((m,5120),device='cuda',dtype=torch.int8);asc=torch.empty((m,1),device='cuda',dtype=torch.float16)
        quantize(aq,xm,mask,qa,asc,stream=stream)
        q=torch.empty((m,10240),device='cuda',dtype=torch.float16);zout=torch.empty((m,6144),device='cuda',dtype=torch.float16)
        kernel=fixed if m==512 else dynamic
        def run():kernel.adapter.func(qa,b,asc,ws,q,zout,stream=stream)
        run();reference=(qa.float()@b.float().T)*asc.float()*ws.float()[None,:]
        actual=torch.cat((q,zout),-1);e=error(actual,reference);assert e['finite'] and e['relative_l2']<.002,e
        expected=q.clone(),zout.clone()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):kernel.adapter.func(qa,b,asc,ws,q,zout,stream=torch.cuda.current_stream().cuda_stream)
        qa_saved=qa.clone();qa.zero_();q.fill_(float('nan'));zout.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
        assert bool((q==0).all() and (zout==0).all())
        qa.copy_(qa_saved);graph.replay();torch.cuda.synchronize();assert torch.equal(q,expected[0]) and torch.equal(zout,expected[1])
        report['qkvz_checks'].append({'M':m,'error':e,'graph_changed_input_and_restore':True})
    report['status']='passed';write_json(a.output/'result.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
