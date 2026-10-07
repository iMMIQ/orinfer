"""Checkpoint-weight vision graph against a PyTorch FP16/BF16 reference.

Reference equations follow Transformers 5.2 Qwen3_5VisionModel (LayerNorm,
2D RoPE, bidirectional attention, tanh GELU, pre-shuffle merger LayerNorm).
Only offline verification uses PyTorch. Production executes these same cubins
through the Rust CUDA Driver runtime.
"""
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from tools.operators.common import configure, error, write_json
from safetensors import safe_open


class VisionGraph:
    def __init__(self, manifest, bucket=512):
        self.root=manifest.parent; self.m=json.loads(manifest.read_text())
        self.weight_map = None
        if 'metadata' in self.m:
            build = self.m
            self.m = dict(build['metadata'], kernels=build['kernels'], programs=build['programs'])
            self.weight_map = json.loads((self.root/'weights/model.safetensors.index.json').read_text())['weight_map']
        self.v=self.m['vision']
        config=self.root.parent/'config.json'
        self.fp32_interpolation=config.exists() and json.loads(config.read_text()).get('model_type')=='qwen4_exp'
        program=next(p['program'] for p in self.v['plans'] if p['patches']==bucket)
        self.ops=self.m['programs'][program];ks={k['name']:k for k in self.m['kernels']}
        self.kernels=[ks[o['name']] for o in self.ops]
        names={a['name'] for k in self.kernels for a in k['args'] if a['kind']=='buffer'}
        self.b={};self.weights={}
        for spec in self.m['buffers']:
            name=spec['name']
            if name not in names:continue
            shape=spec['shape'];dtype={'f16':torch.float16,'bf16':torch.bfloat16,'i32':torch.int32}[spec['dtype']]
            # Verification needs only this graph's rows, avoiding giant unused workspaces.
            if name.startswith('V') and not name.startswith('Vision_') and name not in ('VGrid','VLength'):
                shape=[bucket//4 if name in ('VMerged','VOutput') else bucket,*shape[1:]]
            if spec.get('data'):
                if self.weight_map is not None:
                    with safe_open(self.root/'weights'/self.weight_map[name], framework='pt', device='cpu') as source:
                        value=source.get_tensor(name).reshape(shape).cuda()
                else:
                    value=torch.from_file(str(self.root/spec['data']['file']),size=math.prod(shape),dtype=dtype).reshape(shape).cuda()
                self.weights[name.removeprefix('Vision_')]=value
            else:value=torch.zeros(shape,device='cuda',dtype=dtype)
            self.b[name]=value
        self.driver=C.CDLL('libcuda.so.1');self.modules={};self.calls=[]
        def check(code):
            if code:raise RuntimeError(f'CUDA Driver error {code}')
        self.check=check
        for k in self.kernels:
            path=str(self.root/k['module']['file'])
            if path not in self.modules:
                module=C.c_void_p();check(self.driver.cuModuleLoad(C.byref(module),path.encode()));self.modules[path]=module
            fn=C.c_void_p();check(self.driver.cuModuleGetFunction(C.byref(fn),self.modules[path],k['symbol'].encode()))
            if k['shared_memory_bytes']>49152:check(self.driver.cuFuncSetAttribute(fn,8,k['shared_memory_bytes']))
            args=[C.c_uint64(self.b[a['name']].data_ptr()) if a['kind']=='buffer' else C.c_int32(a['value']) for a in k['args']]
            ptrs=(C.c_void_p*len(args))(*(C.cast(C.byref(a),C.c_void_p) for a in args))
            self.calls.append((k,fn,args,ptrs))
        self.graph=torch.cuda.CUDAGraph()
        self.run()
        with torch.cuda.graph(self.graph):self.run()
    def run(self):
        for k,fn,args,ptrs in self.calls:
            self.check(self.driver.cuLaunchKernel(fn,*k['grid'],*k['block'],k['shared_memory_bytes'],C.c_void_p(torch.cuda.current_stream().cuda_stream),ptrs,None))
    def upload(self,x,gh,gw):
        self.b['VPixels'].zero_();self.b['VPixels'][:len(x)].copy_(x)
        self.b['VGrid'].copy_(torch.tensor([gh,gw],device='cuda',dtype=torch.int32))
        self.b['VLength'].fill_(len(x))
    def replay(self,n):
        self.graph.replay();torch.cuda.synchronize();return self.b['VOutput'][:n//4].clone()

    def reference(self,x,gh,gw,dtype):
        w={k:v.to(dtype) for k,v in self.weights.items()}
        dense=lambda a,p:F.linear(a,w[p+'_weight'],w[p+'_bias'])
        norm=lambda a,p:F.layer_norm(a,(1152,),w[p+'_weight'],w[p+'_bias'],1e-6)
        coords=torch.tensor([(br*2+ir,bc*2+ic) for br in range(gh//2) for bc in range(gw//2) for ir in range(2) for ic in range(2)],device='cuda')
        hf=torch.linspace(0,47,gh,device='cuda')[coords[:,0]];wf=torch.linspace(0,47,gw,device='cuda')[coords[:,1]]
        h0=hf.long();w0=wf.long();h1=(h0+1).clamp_max(47);w1=(w0+1).clamp_max(47);dh=hf-h0;dw=wf-w0
        table=w['pos_embed_weight']
        if self.fp32_interpolation:
            pos=(table[h0*48+w0].float()*((1-dh)*(1-dw))[:,None]
                 +table[h0*48+w1].float()*((1-dh)*dw)[:,None]
                 +table[h1*48+w0].float()*(dh*(1-dw))[:,None]
                 +table[h1*48+w1].float()*(dh*dw)[:,None]).to(dtype)
        else:
            pos=table[h0*48+w0]*((1-dh)*(1-dw)).to(dtype)[:,None]
            pos=pos+table[h0*48+w1]*((1-dh)*dw).to(dtype)[:,None]
            pos=pos+table[h1*48+w0]*(dh*(1-dw)).to(dtype)[:,None]
            pos=pos+table[h1*48+w1]*(dh*dw).to(dtype)[:,None]
        x=F.conv3d(x.to(dtype).reshape(-1,3,2,16,16),
                   w['patch_embed_proj_weight'].reshape(1152,3,2,16,16),
                   w['patch_embed_proj_bias'],stride=(2,16,16)).reshape(-1,1152)+pos
        freq=1/(10000**(torch.arange(0,36,2,device='cuda').float()/36))
        angles=(coords[:,:,None].float()*freq).flatten(1);angles=torch.cat((angles,angles),-1)[:,None,:]
        rotate=lambda z:torch.cat((-z[...,36:],z[...,:36]),-1)
        for i in range(27):
            p=f'blocks_{i}_';q,k,v=dense(norm(x,p+'norm1'),p+'attn_qkv').reshape(len(x),3,16,72).unbind(1)
            q=(q.float()*angles.cos()+rotate(q).float()*angles.sin()).to(dtype)
            k=(k.float()*angles.cos()+rotate(k).float()*angles.sin()).to(dtype)
            att=F.scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0),k.transpose(0,1).unsqueeze(0),v.transpose(0,1).unsqueeze(0),is_causal=False).squeeze(0).transpose(0,1).reshape(len(x),1152)
            x=x+dense(att,p+'attn_proj')
            x=x+dense(F.gelu(dense(norm(x,p+'norm2'),p+'mlp_linear_fc1'),approximate='tanh'),p+'mlp_linear_fc2')
        x=norm(x,'merger_norm').reshape(-1,4608)
        return dense(F.gelu(dense(x,'merger_linear_fc1')),'merger_linear_fc2')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--bucket',type=int,default=512);ap.add_argument('--pixels',type=Path);ap.add_argument('--grid',type=int,nargs=2);ap.add_argument('--official-reference',type=Path);a=ap.parse_args();configure()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    graph=VisionGraph(a.model,a.bucket);results=[]
    for gh,gw in ([tuple(a.grid)] if a.pixels else [(16,16),(14,22)]):
        n=gh*gw;x=(torch.from_file(str(a.pixels),size=n*1536,dtype=torch.float32).reshape(n,1536).cuda() if a.pixels else torch.rand((n,1536),device='cuda',dtype=torch.float32)*2-1)
        graph.upload(x,gh,gw);actual=graph.replay(n)
        if a.pixels:actual.half().cpu().contiguous().view(torch.uint8).numpy().tofile(a.output/'actual.f16')
        fp16=graph.reference(x,gh,gw,torch.float16);bf16=graph.reference(x,gh,gw,torch.bfloat16)
        e16=error(actual,fp16);ebf=error(actual,bf16)
        if a.pixels:bf16.half().cpu().contiguous().view(torch.uint8).numpy().tofile(a.output/'reference-bf16.f16')
        print(gh,gw,'FP16',e16,'BF16',ebf,flush=True)
        native=ebf if graph.v.get('dtype','f16')=='bf16' else e16
        assert native['finite'] and native['relative_l2']<.015,(gh,gw,native)
        official = None
        if a.official_reference:
            reference = torch.from_file(str(a.official_reference),size=actual.numel(),dtype=torch.float16).reshape(actual.shape).cuda()
            official = error(actual,reference)
            print('Official model reference',official,flush=True)
            assert official['finite'] and official['relative_l2']<.015,official
        graph.upload(-x,gh,gw);changed=graph.replay(n);assert not torch.equal(actual,changed)
        graph.upload(x,gh,gw);assert torch.equal(actual,graph.replay(n))
        results.append({'grid':[gh,gw],'fp16':e16,'bf16':ebf,'official_reference':official,'graph_changed_input_restore':True})
    write_json(a.output/'result.json',{'status':'passed','checks':results})

if __name__=='__main__':main()
