"""Compare shared sparse tensor-core INT8 KV attention with the FP32 oracle."""
import argparse
from pathlib import Path
import torch
from kernels.model.qsa import sparse_merge
from kernels.model.qsa_attention import sparse_attention
from tools.model.flash_qsa_reference import attention,quantize_kv,dequantize_kv
from tools.operators.common import configure,error,benchmark,write_json,export_kernel


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();configure()
    report={'complete':False,'cases':[]}
    for m,capacity,start in [(1,2053,0),(4,2053,2047),(1,262144,262143),(128,262144,262016)]:
        query=torch.randn((m,24,256),device='cuda').half();gate=torch.randn_like(query)
        key=torch.randn((capacity,2,256),device='cuda').half();value=torch.randn_like(key)
        k,ks=quantize_kv(key);v,vs=quantize_kv(value);pos=torch.tensor([start],device='cuda',dtype=torch.int32)
        selected=torch.full((m,2051),-1,device='cuda',dtype=torch.int32)
        for row in range(m):
            n=min(512,(start+row+1)//4);tail=(start+row+1)%4
            selected[row,:n*4]=torch.arange(n*4,device='cuda')
            selected[row,n*4:n*4+tail]=torch.arange((start+row+1)//4*4,start+row+1,device='cuda')
        maximum=torch.empty((m,24,8),device='cuda');den=torch.empty_like(maximum)
        partial=torch.empty((m,24,8,256),device='cuda');output=torch.empty_like(query)
        kernel=sparse_attention(m,capacity);merge=sparse_merge(m)
        def run():kernel(query,k,v,ks,vs,selected,pos,maximum,den,partial);merge(maximum,den,partial,gate,output)
        run();torch.cuda.synchronize();reference=attention(query,dequantize_kv(k,ks),dequantize_kv(v,vs),gate,selected)
        metric=error(output,reference)
        if metric['relative_l2']>=.004:
            write_json(a.output/'debug.json',{'output':output[0,:4,:8].tolist(),'ref':reference[0,:4,:8].tolist(),
                'max':maximum[0,:4].tolist(),'den':den[0,:4].tolist(),'partial':partial[0,:4,0,:8].tolist()})
        assert metric['finite'] and metric['relative_l2']<.004,metric
        timing,graph=benchmark(run,repetitions=3)
        query.mul_(.9);graph.replay();torch.cuda.synchronize()
        metric2=error(output,attention(query,dequantize_kv(k,ks),dequantize_kv(v,vs),gate,selected))
        assert metric2['finite'] and metric2['relative_l2']<.004,metric2
        export_kernel(kernel,a.output/f'attention-{m}-{capacity}')
        report['cases'].append({'rows':m,'capacity':capacity,'position':start,'error':metric,
                                'changed_graph_error':metric2,'timing':timing})
        write_json(a.output/'results.json',report);print('shared QSA',m,capacity,timing['median_ms'],flush=True)
        del query,gate,key,value,k,ks,v,vs,maximum,den,partial,output,reference,graph
    report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
