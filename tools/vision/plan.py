"""Shared checkpoint ViT buffers, TileLang compilation and encoder execution order."""
import hashlib
import math
from pathlib import Path
import subprocess
import torch
from safetensors import safe_open
from tools.operators.common import export_kernel
from tools.operators.abi import parse_host, evaluate
from kernels.vision import encoder


class EncoderBuilder:
    def identity(self,path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
        return {'file':str(path.relative_to(self.output)),'sha256':h.hexdigest()}

    def compile(self,name,factory):
        path=self.output/'aot'/name
        export_kernel(factory(),path)
        if self.dtype == 'bf16' and name not in ('embedding_features', 'full_prepare_mrope'):
            # CUDA 12.6 NVRTC miscompiles __bfloat1622float2 into a trap on this
            # toolchain. Compile the unchanged TileLang CUDA source with NVCC.
            import tilelang
            includes = Path(tilelang.__file__).parent
            subprocess.run(['/usr/local/cuda/bin/nvcc', '-arch=sm_87', '-std=c++17',
                            '--cubin', f'-I{includes}/src',
                            f'-I{includes}/3rdparty/cutlass/include', '-DENABLE_BF16=1',
                            str(path/'kernel.cu'), '-o', str(path/'kernel.cubin')], check=True)
        abi=parse_host((path/'host.txt').read_text());assert len(abi)==1
        self.exports[name]={**abi[0],'module':self.identity(path/'kernel.cubin'),'source':self.identity(path/'kernel.cu'),'host_abi':self.identity(path/'host.txt')}
        print('compiled',name,flush=True)

    def kernel(self,name,pointers,rows,kernel_name=None):
        abi=self.exports[name];launch=abi['launch_expressions'];scalars={'rows':rows};args=[]
        for arg in abi['ordered_arguments']:
            value,ctype=arg['value'],arg['ctype']
            if ctype in ('ctypes.c_void_p','c_void_p'):
                key=value.removesuffix('.data_ptr()');args.append({'kind':'buffer','name':pointers[key]})
            else:
                assert ctype in ('ctypes.c_int','ctypes.c_int32'),(name,arg)
                args.append({'kind':'i32','value':evaluate(value,scalars)})
        return dict(name=kernel_name or f'vision_{name}_{len(self.manifest["kernels"])}',
                    **{key:abi[key] for key in ('module','source','host_abi','symbol')},
                    grid=[evaluate(launch['gridDim'+a],scalars) for a in 'XYZ'],
                    block=[evaluate(launch['blockDim'+a],scalars) for a in 'XYZ'],
                    shared_memory_bytes=evaluate(launch['sharedMemBytes'],scalars),cooperative=False,args=args)

    def emit(self,program,name,rows,**pointers):
        k=self.kernel(name,pointers,rows);self.manifest['kernels'].append(k);program.append({'kind':'kernel','name':k['name']})

    def build_encoder(self):
        v=self.vision;h=v['hidden_size'];f=v['intermediate_size'];out=v['out_hidden_size'];n=self.max_patches;m=self.manifest
        for name,shape,dtype in [('VPixels',(n,1536),'f16'),('VGrid',(2,),'i32'),('VLength',(1,),'i32'),
              ('VHidden',(n,h),'f16'),('VNorm',(n,h),'f16'),('VQKV',(n,3*h),'f16'),('VQ',(n,h),'f16'),('VK',(n,h),'f16'),('VV',(n,h),'f16'),
              ('VAttention',(n,h),'f16'),('VProjection',(n,h),'f16'),('VFFN',(n,f),'f16'),('VMerged',(n//4,4*h),'f16'),('VOutput',(n//4,out),'f16'),
              ('Features',(self.max_features,out),'f16'),('FeatureIndex',(m['max_context'],),'i32'),('MRopePositions',(m['max_context'],3),'i32')]:self.buffer(name,shape,self.dtype if dtype=='f16' and name not in ('VOutput','Features') else dtype)
        parameters=0
        with safe_open(self.source,framework='pt',device='cpu') as source:
            for key in sorted(k for k in source.keys() if k.startswith('model.visual.')):
                original=source.get_tensor(key)
                tensor=original.to(torch.bfloat16 if self.dtype=='bf16' else torch.float16)
                if not torch.equal(tensor.to(original.dtype), original):
                    # BF16 values in FP16's subnormal range can round by at most
                    # half a subnormal ULP. Reject overflow or any larger change.
                    delta=(tensor.float()-original.float()).abs()
                    if not bool(torch.isfinite(tensor).all()) or float(delta.max()) > 2**-25:
                        raise ValueError(f'Vision weight conversion exceeds FP16 subnormal rounding: {key}')
                parameters+=tensor.numel()
                name='Vision_'+key.removeprefix('model.visual.').replace('.','_')
                if key=='model.visual.patch_embed.proj.weight':tensor=tensor.flatten(1)
                self.buffer(name,tensor.shape,dtype=self.dtype,data=tensor,weight=True)
        self.compile('patch',lambda:encoder.linear(h,1536,dtype=self.kernel_dtype,separate_bias=True))
        self.compile('qkv',lambda:encoder.linear(3*h,h,dtype=self.kernel_dtype))
        self.compile('projection',lambda:encoder.linear(h,h,dtype=self.kernel_dtype))
        self.compile('fc1',lambda:encoder.linear(f,h,'gelu_tanh',dtype=self.kernel_dtype))
        self.compile('fc2',lambda:encoder.linear(h,f,dtype=self.kernel_dtype))
        self.compile('merge1',lambda:encoder.linear(4*h,4*h,'gelu',dtype=self.kernel_dtype))
        self.compile('merge2',lambda:encoder.linear(out,4*h,dtype=self.kernel_dtype,output_dtype="float16"))
        self.compile('norm',lambda:encoder.layer_norm(h,dtype=self.kernel_dtype))
        self.compile('add',lambda:encoder.add(h,dtype=self.kernel_dtype))
        self.compile('position',lambda:encoder.position(h,int(math.isqrt(v['num_position_embeddings'])),dtype=self.kernel_dtype,fp32_interpolation=getattr(self,'fp32_interpolation',False)))
        self.compile('rope',lambda:encoder.qkv_rope(h,v['num_heads'],dtype=self.kernel_dtype))
        self.compile('attention',lambda:encoder.attention(h,v['num_heads'],dtype=self.kernel_dtype))
        plans=[]
        for rows in (256,512,1024,2048,4096,8192,16384,32768):
            if rows>n:break
            p=[]
            def dense(which,x,y,prefix,count=rows):
                self.emit(p,which,count,X=x,W='Vision_'+prefix+'_weight',Bias='Vision_'+prefix+'_bias',Y=y)
            def norm(x,y,prefix):self.emit(p,'norm',rows,X=x,W='Vision_'+prefix+'_weight',B='Vision_'+prefix+'_bias',Y=y)
            dense('patch','VPixels','VNorm','patch_embed_proj')
            self.emit(p,'position',rows,X='VNorm',W='Vision_pos_embed_weight',Grid='VGrid',Length='VLength',Y='VHidden')
            for i in range(v['depth']):
                prefix=f'blocks_{i}_'
                norm('VHidden','VNorm',prefix+'norm1')
                dense('qkv','VNorm','VQKV',prefix+'attn_qkv')
                self.emit(p,'rope',rows,X='VQKV',Grid='VGrid',Length='VLength',Q='VQ',K='VK',V='VV')
                self.emit(p,'attention',rows,Q='VQ',K='VK',V='VV',Length='VLength',Y='VAttention')
                dense('projection','VAttention','VProjection',prefix+'attn_proj')
                self.emit(p,'add',rows,X='VProjection',Residual='VHidden',Y='VHidden')
                norm('VHidden','VNorm',prefix+'norm2')
                dense('fc1','VNorm','VFFN',prefix+'mlp_linear_fc1')
                dense('fc2','VFFN','VProjection',prefix+'mlp_linear_fc2')
                self.emit(p,'add',rows,X='VProjection',Residual='VHidden',Y='VHidden')
            norm('VHidden','VNorm','merger_norm')
            dense('merge1','VNorm','VMerged','merger_linear_fc1',rows//4)
            dense('merge2','VMerged','VOutput','merger_linear_fc2',rows//4)
            program=f'vision_m{rows}';m['programs'][program]=p;plans.append({'patches':rows,'program':program})
        m['vision']={'dtype':self.dtype,'hidden':out,'patch_size':v['patch_size'],'temporal_patch_size':v['temporal_patch_size'],'merge_size':v['spatial_merge_size'],
                     'max_patches':n,'max_features':self.max_features,'pixels':'VPixels','grid':'VGrid','length':'VLength','output':'VOutput',
                     'features':'Features','feature_index':'FeatureIndex','mrope_positions':'MRopePositions','plans':plans,
                     'image_token_id':self.config['image_token_id'],'vision_start_token_id':self.config['vision_start_token_id'],'vision_end_token_id':self.config['vision_end_token_id']}
        return parameters
