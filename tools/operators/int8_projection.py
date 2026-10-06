"""Exact integer oracle, output tails and graph replay for dense INT8."""
import argparse
from pathlib import Path
import subprocess

import torch

from kernels.model.int8_projection import int8_projection
from tools.operators.common import configure, benchmark, export_kernel, error, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args();configure()
    cases = []
    for m,n,k in ((1,48,2560),(3,65,128),(17,129,640)):
        for dtype in ('float16','float32'):
            x = torch.randint(-128,128,(m,k),device='cuda',dtype=torch.int8)
            w = torch.randint(-128,128,(n,k),device='cuda',dtype=torch.int8)
            ws = torch.full((n,),.015625,device='cuda',dtype=torch.float16)
            ts = torch.full((m,),.0078125,device='cuda',dtype=torch.float16)
            out = torch.full((m+1,n),99.,device='cuda',dtype=getattr(torch,dtype))
            kernel = int8_projection(m,n,k,dtype)
            def run():kernel(x,w,ws,ts,out[:m])
            timing,graph = benchmark(run,repetitions=5)
            expected = (((x.double()@w.double().T).float()*ws.float()[None,:])*ts.float()[:,None]).to(out.dtype)
            assert torch.equal(out[:m],expected),error(out[:m],expected)
            assert bool((out[-1] == 99).all())
            saved = x.clone();x.zero_();out[:m].fill_(99);graph.replay();torch.cuda.synchronize()
            assert bool((out[:m] == 0).all())
            x.copy_(saved);ws.mul_(2);graph.replay();torch.cuda.synchronize()
            assert torch.equal(out[:m],expected*2)
            folder = a.output/f'M{m}-N{n}-K{k}-{dtype}'
            exported = export_kernel(kernel,folder)
            sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(folder/'kernel.cubin')],text=True)
            assert 'IMMA.16832.S8.S8' in sass
            cases.append({'shape':[m,n,k],'output_dtype':dtype,'exact_integer_oracle':True,'tail_guard':True,
                          'changed_input_and_scale_graph':True,'timing':timing,'export':exported})
    write_json(a.output/'results.json',{'complete':True,'cases':cases,'scope':__doc__})


if __name__ == '__main__':main()
