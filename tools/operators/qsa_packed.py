"""Compare signed packed KV dequantization and QSA split candidates."""
import argparse
from pathlib import Path
import torch
from kernels.model.qsa import sparse_merge
from kernels.model.qsa_attention import sparse_attention
from tools.model.flash_qsa_reference import attention,dequantize_kv
from tools.operators.common import configure,error,benchmark,write_json,export_kernel


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--metadata-check',action='store_true',help='Check metadata graph mutation, tails and the full 256k address range')
    a=p.parse_args();configure()
    report={'complete':False,'cases':[]}
    cases=[(1,9,0),(2,35,31),(4,2053,2047),(8,8192,8184),(512,4096,3584),(8,262144,262136)] if a.metadata_check else [(1,2053,0),(4,2053,2047),(1,262144,262143),(128,262144,262016)]
    for m,capacity,start in cases:
        query=torch.randn((m,24,256),device='cuda').half();gate=torch.randn_like(query)
        k=torch.randint(-128,128,(capacity,2,256),device='cuda',dtype=torch.int8);v=torch.randint_like(k,-128,128)
        ks=(torch.rand((capacity,2,4),device='cuda')*.025+.001).half();vs=torch.rand_like(ks)*.025+.001
        position=torch.tensor([start],device='cuda',dtype=torch.int32)
        selected=torch.full((m,2051),-1,device='cuda',dtype=torch.int32)
        def select_rows(first):
            selected.fill_(-1)
            for row in range(m):
                full=(first+row+1)//4;n=min(512,full);tail=(first+row+1)%4
                # Random distant blocks exercise gathers and all signed byte lanes.
                blocks=torch.randperm(full,device='cuda')[:n] if full else torch.empty(0,device='cuda',dtype=torch.int64)
                selected[row,:n*4]=(blocks[:,None]*4+torch.arange(4,device='cuda')).flatten()
                selected[row,n*4:n*4+tail]=torch.arange(full*4,full*4+tail,device='cuda')
        select_rows(start)
        oracle=attention(query,dequantize_kv(k,ks),dequantize_kv(v,vs),gate,selected)
        case={'rows':m,'capacity':capacity,'position':start,'candidates':[]};baseline=None
        variants=[(8,False),(8,True)] if a.metadata_check else [(8,False),(8,True),(4,True),(2,True),(1,True)]
        for splits,packed in variants:
            maximum=torch.empty((m,24,splits),device='cuda');den=torch.empty_like(maximum)
            partial=torch.empty((m,24,splits,256),device='cuda')
            guarded=torch.full((m+1,24,256),91.,device='cuda',dtype=torch.float16);out=guarded[:m]
            kernel=sparse_attention(m,capacity,splits,packed);merge=sparse_merge(m,splits)
            kk,vv=(k.view(torch.uint32),v.view(torch.uint32)) if packed else (k,v)
            def run():kernel(query,kk,vv,ks,vs,selected,position,maximum,den,partial);merge(maximum,den,partial,gate,out)
            timing,graph=benchmark(run,repetitions=5)
            metric=error(out,oracle);assert metric['finite'] and metric['relative_l2']<.004,metric
            if baseline is None:baseline=out.clone()
            if splits==8:assert torch.equal(out,baseline),'Packed conversion changed FP16 bits'
            old=query.clone();old_selected=selected.clone();query.mul_(.9)
            if a.metadata_check:position.fill_(max(0,start-1));select_rows(max(0,start-1))
            graph.replay();torch.cuda.synchronize()
            changed=error(out,attention(query,dequantize_kv(k,ks),dequantize_kv(v,vs),gate,selected))
            assert changed['finite'] and changed['relative_l2']<.004,changed
            assert bool((guarded[-1]==91).all())
            query.copy_(old);selected.copy_(old_selected);position.fill_(start)
            export_kernel(kernel,a.output/f'attention-{m}-{capacity}-{splits}-{packed}')
            case['candidates'].append({'splits':splits,'packed':packed,'error':metric,'changed_graph_error':changed,
                                       'metadata_graph_mutation':a.metadata_check,'tail_guard':True,'timing':timing})
            print('QSA',m,capacity,splits,packed,timing['median_ms'],flush=True)
        report['cases'].append(case);write_json(a.output/'results.json',report)
    report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
