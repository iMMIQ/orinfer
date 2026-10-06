"""Independent MTP fusion, all-prefix histories and multi-row greedy checks."""
import argparse
from pathlib import Path
import torch
from kernels.model import flash_mtp as fm
from kernels.model.greedy import greedy_partials,greedy_merge
from tools.operators.common import configure,write_json,error


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    configure();report={'complete':False,'norm':[],'history':[],'pending':[],'greedy':[]}
    for rows in (1,2,3,8):
        for width in (65,2560,10240):
            x=torch.randn((rows,width),device='cuda').half();gamma=torch.randn(width,device='cuda')*.1+1
            out=torch.empty_like(x);kernel=fm.norm(rows,width)
            expected=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*gamma).half()
            kernel(x,gamma,out);torch.cuda.synchronize()
            torch.testing.assert_close(out,expected,atol=.004,rtol=.002)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):kernel(x,gamma,out)
            x.neg_();graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(out,-expected,atol=.004,rtol=.002)
            report['norm'].append({'rows':rows,'width':width,'error':error(out,-expected),'changed_graph':True})
        embed=torch.randn(rows,2560,device='cuda').half();hidden=torch.randn(rows,4,2560,device='cuda').half()
        out=torch.empty_like(hidden);fm.fuse(rows,2560)(embed,hidden,out)
        assert torch.equal(out,(embed[:,None].float()+hidden.float()).half())
        for history,width in ((3,10240),(9,10240)):
            x=torch.randn(rows,width,device='cuda').half();state=torch.randn(history,width,device='cuda').half()
            saved=torch.empty(rows,history,width,device='cuda',dtype=torch.float16)
            kernel=fm.history_prefix(rows,width,history);kernel(x,state,saved)
            for i in range(rows):assert torch.equal(saved[i],torch.cat((state,x[:i+1]))[-history:])
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):kernel(x,state,saved)
            x.neg_();state.neg_();graph.replay();torch.cuda.synchronize()
            for i in range(rows):assert torch.equal(saved[i],torch.cat((state,x[:i+1]))[-history:])
            report['history'].append({'rows':rows,'history':history,'exact':True,'changed_graph':True})
        for start in (0,1,3,262136):
            qk=torch.randn(rows,5,128,device='cuda').half();state=torch.randn(4,128,device='cuda').half()
            position=torch.tensor([start],device='cuda',dtype=torch.int32)
            saved=torch.empty(rows,4,128,device='cuda',dtype=torch.float16)
            kernel=fm.pending_prefix(rows);kernel(qk,state,position,saved)
            expected=state.clone()
            for i in range(rows):
                expected[(start+i)%4].copy_(qk[i,4]);assert torch.equal(saved[i],expected)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):kernel(qk,state,position,saved)
            position.add_(1);qk.neg_();graph.replay();torch.cuda.synchronize();expected=state.clone()
            for i in range(rows):
                expected[(start+1+i)%4].copy_(qk[i,4]);assert torch.equal(saved[i],expected)
            report['pending'].append({'rows':rows,'start':start,'exact':True,'changed_graph':True})
        vocab=248320;blocks=(vocab+1023)//1024
        x=torch.randn(rows,vocab,device='cuda');values=torch.empty(rows*blocks,device='cuda')
        indices,bad=(torch.empty(rows*blocks,device='cuda',dtype=torch.int32) for _ in range(2))
        out=torch.empty(rows*2,device='cuda',dtype=torch.int32)
        part,merge=greedy_partials(vocab,rows),greedy_merge(vocab,rows)
        part(x,values,indices,bad);merge(values,indices,bad,out)
        assert torch.equal(out.view(rows,2)[:,0],x.argmax(1).int()) and not out.view(rows,2)[:,1].any()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            part(x,values,indices,bad);merge(values,indices,bad,out)
        x.fill_(-3);x[:,0]=7;x[:,-1]=7;graph.replay();torch.cuda.synchronize()
        assert not out.any()
        x[-1,-1]=float('nan');graph.replay();torch.cuda.synchronize()
        assert out.view(rows,2)[-1,1]==1 and not out.view(rows,2)[:-1,1].any()
        report['greedy'].append({'rows':rows,'exact':True,'changed_graph':True,'row_isolation':True})
        write_json(a.output/'results.json',report)
    report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
