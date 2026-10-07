"""Validate greedy reduction, tie rules, nonfinite rejection and changed graph inputs."""
import argparse
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compile-cache',type=Path)
    a=p.parse_args()
    if a.compile_cache:shutil.copytree(a.compile_cache/'0.1.15',a.output/'cache'/'0.1.15',dirs_exist_ok=True)
    import torch
    from kernels.model.greedy import greedy_partials,greedy_merge
    from tools.operators.common import configure,benchmark,write_json,export_kernel
    configure();report={'complete':False,'greedy':[]}
    for vocab in (1,65,1024,1025,248320):
        x=torch.randn((1,vocab),device='cuda');blocks=(vocab+1023)//1024
        vals=torch.empty(blocks,device='cuda');ids=torch.empty(blocks,device='cuda',dtype=torch.int32)
        bad=torch.empty_like(ids);out=torch.empty(2,device='cuda',dtype=torch.int32)
        part,merge=greedy_partials(vocab),greedy_merge(vocab)
        def run():part(x,vals,ids,bad);merge(vals,ids,bad,out)
        timing,graph=benchmark(run,repetitions=5)
        assert out.tolist()==[int(x.argmax()),0]
        x.fill_(-3);x[0,vocab-1]=7;x[0,0]=7;graph.replay();torch.cuda.synchronize()
        assert out.tolist()==[0,0],'Argmax tie rule changed'
        for value in (float('nan'),float('inf'),float('-inf')):
            x[0,-1]=value;graph.replay();torch.cuda.synchronize();assert out[1].item()==1
        report['greedy'].append({'vocab':vocab,'changed_graph_and_nonfinite':True,'timing':timing})
        export_kernel(part,a.output/f'greedy-partials-{vocab}');export_kernel(merge,a.output/f'greedy-merge-{vocab}')
        write_json(a.output/'results.json',report)
    report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
