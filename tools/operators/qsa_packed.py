"""Compare signed packed KV dequantization and QSA split candidates."""
import argparse
from pathlib import Path
import torch
from kernels.model.qsa import sparse_merge
from kernels.model.qsa_attention import sparse_attention
from tools.model.flash_qsa_reference import attention,dequantize_kv
from tools.operators.common import configure,error,benchmark,write_json,export_kernel


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();configure()
    report={'complete':False,'cases':[]}
    for m,capacity,start in [(1,2053,0),(4,2053,2047),(1,262144,262143),(128,262144,262016)]:
        query=torch.randn((m,24,256),device='cuda').half();gate=torch.randn_like(query)
        k=torch.randint(-127,128,(capacity,2,256),device='cuda',dtype=torch.int8);v=torch.randint_like(k,-127,128)
        ks=(torch.rand((capacity,2,4),device='cuda')*.025+.001).half();vs=torch.rand_like(ks)*.025+.001
        position=torch.tensor([start],device='cuda',dtype=torch.int32)
        selected=torch.full((m,2051),-1,device='cuda',dtype=torch.int32)
        for row in range(m):
            full=(start+row+1)//4;n=min(512,full);tail=(start+row+1)%4
            # Random distant blocks exercise gathers and all signed byte lanes.
            blocks=torch.randperm(full,device='cuda')[:n] if full else torch.empty(0,device='cuda',dtype=torch.int64)
            selected[row,:n*4]=(blocks[:,None]*4+torch.arange(4,device='cuda')).flatten()
            selected[row,n*4:n*4+tail]=torch.arange(full*4,full*4+tail,device='cuda')
        oracle=attention(query,dequantize_kv(k,ks),dequantize_kv(v,vs),gate,selected)
        case={'rows':m,'capacity':capacity,'position':start,'candidates':[]};baseline=None
        for splits,packed in [(8,False),(8,True),(4,True),(2,True),(1,True)]:
            maximum=torch.empty((m,24,splits),device='cuda');den=torch.empty_like(maximum)
            partial=torch.empty((m,24,splits,256),device='cuda');out=torch.empty_like(query)
            kernel=sparse_attention(m,capacity,splits,packed);merge=sparse_merge(m,splits)
            kk,vv=(k.view(torch.uint32),v.view(torch.uint32)) if packed else (k,v)
            def run():kernel(query,kk,vv,ks,vs,selected,position,maximum,den,partial);merge(maximum,den,partial,gate,out)
            timing,graph=benchmark(run,repetitions=5)
            metric=error(out,oracle);assert metric['finite'] and metric['relative_l2']<.004,metric
            if baseline is None:baseline=out.clone()
            if splits==8:assert torch.equal(out,baseline),'Packed conversion changed FP16 bits'
            old=query.clone();query.mul_(.9);graph.replay();torch.cuda.synchronize()
            changed=error(out,attention(query,dequantize_kv(k,ks),dequantize_kv(v,vs),gate,selected))
            assert changed['finite'] and changed['relative_l2']<.004,changed
            query.copy_(old)
            export_kernel(kernel,a.output/f'attention-{m}-{capacity}-{splits}-{packed}')
            case['candidates'].append({'splits':splits,'packed':packed,'error':metric,'changed_graph_error':changed,'timing':timing})
            print('QSA',m,capacity,splits,packed,timing['median_ms'],flush=True)
        report['cases'].append(case);write_json(a.output/'results.json',report)
    report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
