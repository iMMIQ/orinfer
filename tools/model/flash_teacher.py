"""Layerwise original-BF16 teacher for the fixed Flash Next token histories.

Reference arithmetic is independent Torch, with FP32 state/attention reductions
and a FP32 output-head accumulator. Only original BF16 weight ranges are used.
No full original model is resident or stored. This is not a production engine.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from tools.model.flash_original import Original
from tools.model.flash_ple import PleLookup
from tools.model.flash_roles import role
from tools.model.flash_reference_math import attention, causal_conv, delta_rule, ple, read_streams, write_streams
from tools.model.flash_validation import probe
from tools.model.flash_qsa_reference import index as qsa_index, select as qsa_select, attention as qsa_attention
from tools.model.flash_teacher_inputs import Inputs
from tools.model.safetensors_source import Source
from tools.quantization.flash_next import atomic_json, digest
from tools.operators.common import configure, environment, SEED


class Teacher:
    def __init__(self, original, cases, *, chunk_experts=16, device='cuda', inputs=None):
        self.source,self.cases,self.chunk_experts = original,cases,chunk_experts
        self.inputs=inputs
        self.device = torch.device(device)
        self.text = original.config['text_config']
        self.h = self.text['hidden_size'];self.c = self.text['hc_count']
        self.e = self.text['num_experts'];self.topk = self.text['num_experts_per_tok']
        self.f = self.text['moe_intermediate_size'];self.vocab = self.text['vocab_size']
        if (self.text.get('norm_topk_prob',True) is not True or self.text['output_gate_type'] != 'sigmoid'
                or self.text['hidden_act'] != 'silu'):
            raise ValueError('Unsupported original router/gate semantics')
        self.tokens=[];self.segments=[]
        for case in cases:
            prompt,target = case['prompt_ids'],case['target_ids']
            if not prompt or not target:raise ValueError('Nonempty prompt and fixed target required')
            tokens = prompt+target[:-1]
            if len(tokens) > self.text.get('max_position_embeddings',2048):
                raise ValueError('Reference exceeds original configured context')
            if any(type(t) is not int or not 0 <= t < self.vocab for t in tokens+target):
                raise ValueError('Token outside original vocabulary')
            start = len(self.tokens);self.tokens.extend(tokens);self.segments.append((start,len(self.tokens)))
        if chunk_experts < 1 or chunk_experts*self.h*max(2*self.f,self.f)*2 > original.max_read_bytes:
            raise ValueError('Original expert chunk exceeds bounded read budget')
        self.features = None

    def validate(self):
        """Check every required arithmetic tensor using source headers only."""
        text=self.text;h,c,e,f=self.h,self.c,self.e,self.f;rank=text['hc_lowrank']
        required={'model.language_model.embed_tokens.weight':(self.vocab,h),'lm_head.weight':(self.vocab,h)}
        hc={'hc_norm.weight':(c*h,),'input_mix_weight_down.weight':(rank,c*h),
            'input_mix_weight_up.weight':(c*h,rank),'block_inject_weight.weight':(c,c*h)}
        terminal='model.language_model.hyper_connection_mixer.'
        required.update({terminal+name:shape for name,shape in hc.items() if name != 'block_inject_weight.weight'})
        for layer,kind in enumerate(text['layer_types']):
            prefix=f'model.language_model.layers.{layer}.'
            for family in ('attn','mlp'):
                required.update({prefix+family+'_hyper_connection.'+name:shape for name,shape in hc.items()})
            shared=text['shared_expert_intermediate_size']
            mlp={'gate.weight':(e,h),'shared_expert_gate.weight':(1,h),'experts.gate_up_proj':(e,2*f,h),
                 'experts.down_proj':(e,h,f),'shared_expert.gate_proj.weight':(shared,h),
                 'shared_expert.up_proj.weight':(shared,h),'shared_expert.down_proj.weight':(h,shared)}
            required.update({prefix+'mlp.'+name:shape for name,shape in mlp.items()})
            if kind == 'linear_attention':
                hk,hv=text['linear_num_key_heads'],text['linear_num_value_heads']
                dk,dv=text['linear_key_head_dim'],text['linear_value_head_dim']
                dimensions={'in_proj_qkv.weight':(2*hk*dk+hv*dv,h),'in_proj_z.weight':(hv*dv,h),
                    'in_proj_a.weight':(hv,h),'in_proj_b.weight':(hv,h),'out_proj.weight':(h,hv*dv),
                    'A_log':(hv,),'dt_bias':(hv,),'norm.weight':(dv,),
                    'conv1d.weight':(2*hk*dk+hv*dv,1,text['linear_conv_kernel_dim'])}
                family='linear_attn.'
            elif kind == 'full_attention':
                heads,kh,dim=text['num_attention_heads'],text['num_key_value_heads'],text['head_dim']
                dimensions={'q_proj.weight':(2*heads*dim,h),'k_proj.weight':(kh*dim,h),'v_proj.weight':(kh*dim,h),
                            'o_proj.weight':(h,heads*dim),'q_norm.weight':(dim,),'k_norm.weight':(dim,)}
                if any(end-start>text['indexer_budget'] for start,end in self.segments):
                    dimensions.update({'indexer.index_qk_proj.weight':(640,h),
                        'indexer.q_layernorm.weight':(128,),'indexer.k_layernorm.weight':(128,)})
                family='self_attn.'
            else:raise ValueError('Unknown original attention layer')
            required.update({prefix+family+name:shape for name,shape in dimensions.items()})
            if layer+1 in text['ple_layer_ids']:
                width=text['ple_embed_dim']
                dimensions={'key_proj.weight':(c*h,width),'value_proj.weight':(h,width),
                    'norm_key.weight':(c*h,),'norm_query.weight':(c*h,),'norm_conv.weight':(c*h,),
                    'conv1d.weight':(c*h,1,text['ple_conv_kernel_size'])}
                required.update({prefix+'ple.'+name:shape for name,shape in dimensions.items()})
        for name,shape in required.items():
            if name not in self.source.parts or self.source.shape(name) != shape:
                raise ValueError(f'Missing or mismatched original tensor: {name}')
            dtype=self.source.info(name)['dtype']
            if dtype not in ('BF16','F32') or len(shape) >= 2 and dtype != 'BF16':
                raise ValueError(f'Original precision required: {name}')
        return {'checked_tensors':len(required),'layers':len(text['layer_types'])}

    def initial(self):
        if self.inputs is not None:
            values=[self.inputs.case(case,self.h)[0] for case in self.cases]
            embedding=torch.from_numpy(np.concatenate(values)).to(dtype=torch.bfloat16)
            return embedding[:,None,:].expand(-1,self.c,-1).contiguous()
        name = 'model.language_model.embed_tokens.weight'
        if self.source.info(name)['dtype'] != 'BF16' or self.source.shape(name) != (self.vocab,self.h):
            raise ValueError('Original BF16 token embeddings required')
        rows = {token:self.source.rows('model.language_model.embed_tokens.weight',token,1)[0]
                for token in sorted(set(self.tokens))}
        embedding = torch.from_numpy(np.stack([rows[t] for t in self.tokens])).to(dtype=torch.bfloat16)
        return embedding[:,None,:].expand(-1,self.c,-1).contiguous()

    def load_weights(self, layer):
        prefix = f'model.language_model.layers.{layer}.' if isinstance(layer,int) else 'model.language_model.hyper_connection_mixer.'
        weights = {}
        for name in sorted(self.source.parts):
            if not name.startswith(prefix) or '.experts.' in name:continue
            label = role(name)
            if label is None:continue
            if isinstance(layer,int):label = label[len(f'blk.{layer}.'):]
            else:label = label[len('output_hc_'):]
            dtype = self.source.info(name)['dtype']
            if dtype not in ('BF16','F32') or len(self.source.shape(name)) >= 2 and dtype != 'BF16':
                raise ValueError('Original BF16 matrices/FP32 coefficients required')
            value = torch.from_numpy(self.source.tensor(name)).to(device=self.device,
                dtype=torch.float32 if dtype == 'F32' else torch.bfloat16)
            if value.ndim == 3 and value.shape[1] == 1:value = value.squeeze(1)
            weights[label] = value
        return weights

    @staticmethod
    def mixer_weights(weights, prefix):
        return {key:weights[prefix+key+'.weight'] for key in ('norm','down','up','inject') if prefix+key+'.weight' in weights}

    def get_features(self):
        if self.features is None:
            if self.inputs is not None:
                values=[self.inputs.case(case,self.h)[1] for case in self.cases]
                self.features=torch.from_numpy(np.concatenate(values)).to(dtype=torch.bfloat16)
                return self.features
            if any(self.source.info(name)['dtype'] != 'BF16' for name in self.source.parts if '.ngram_embedding.' in name):
                raise ValueError('Original BF16 PLE rows required')
            lookup = PleLookup(self.source);features=[]
            for start,end in self.segments:
                value,_ = lookup.prepare(self.tokens[start:end],[])
                features.append(value)
            self.features = torch.from_numpy(np.concatenate(features)).to(dtype=torch.bfloat16)
        return self.features

    def attention_block(self, x, weights, layer):
        if self.text['layer_types'][layer] == 'linear_attention':
            hk,hv = self.text['linear_num_key_heads'],self.text['linear_num_value_heads']
            dk,dv = self.text['linear_key_head_dim'],self.text['linear_value_head_dim']
            qkv = F.linear(x,weights['attn_qkv.weight'])
            z = F.linear(x,weights['attn_gate.weight']).reshape(-1,hv,dv)
            a = F.linear(x,weights['ssm_alpha.weight']).float()
            b = F.linear(x,weights['ssm_beta.weight']).float()
            decay = -weights['ssm_a_log'].float().exp()*F.softplus(a+weights['ssm_dt.bias'].float())
            beta = torch.sigmoid(b);outputs=[]
            for start,end in self.segments:
                activated = causal_conv(qkv[start:end],weights['ssm_conv1d.weight'])
                q,k,v = activated.split((hk*dk,hk*dk,hv*dv),dim=-1)
                out,_ = delta_rule(q.reshape(-1,hk,dk),k.reshape(-1,hk,dk),v.reshape(-1,hv,dv),
                                   decay[start:end],beta[start:end])
                value = out.float()*torch.rsqrt(out.float().square().mean(-1,keepdim=True)+self.text['rms_norm_eps'])
                value = (value*weights['ssm_norm.weight'].float()*torch.sigmoid(z[start:end].float())).to(x.dtype)
                outputs.append(value.flatten(1))
            return F.linear(torch.cat(outputs),weights['ssm_out.weight'])
        if self.text['layer_types'][layer] != 'full_attention':raise ValueError('Unsupported original layer type')
        heads,kh,dim = self.text['num_attention_heads'],self.text['num_key_value_heads'],self.text['head_dim']
        qg = F.linear(x,weights['attn_q.weight']).reshape(-1,heads,2,dim)
        key = F.linear(x,weights['attn_k.weight']).reshape(-1,kh,dim)
        value = F.linear(x,weights['attn_v.weight']).reshape(-1,kh,dim)
        rope = self.text['rope_parameters'];rotary = int(dim*rope['partial_rotary_factor'])
        outputs=[]
        for start,end in self.segments:
            if end-start<=self.text['indexer_budget']:
                out=attention(qg[start:end],key[start:end],value[start:end],weights['attn_q_norm.weight'],
                              weights['attn_k_norm.weight'],rotary=rotary,theta=rope['rope_theta'])
            else:
                if (self.text.get('indexer_n_heads'),self.text.get('indexer_kv_heads'),
                    self.text.get('indexer_head_dim'),self.text['indexer_budget'],
                    self.text.get('indexer_compress_ratio'),rotary,rope['rope_theta'])!=(4,1,128,2048,4,64,1e7):
                    raise ValueError('Unsupported original long QSA geometry')
                from tools.model.flash_reference_math import norm,neox
                iqk=F.linear(x[start:end],weights['index_qk.weight']).reshape(-1,5,128)
                iq,ck=qsa_index(iqk,weights['index_q_norm.weight'],weights['index_k_norm.weight'])
                query=neox(norm(qg[start:end,:,0],weights['attn_q_norm.weight']),rotary,rope['rope_theta'])
                keys=neox(norm(key[start:end],weights['attn_k_norm.weight']),rotary,rope['rope_theta'])
                # Per-query selection and attention bound the original oracle workspace.
                rows=[]
                for first in range(0,end-start,8):
                    stop=min(end-start,first+8)
                    selected=qsa_select(iq[first:stop],ck,range(first,stop))
                    rows.append(qsa_attention(query[first:stop],keys,value[start:end],qg[start+first:start+stop,:,1],selected,rounded_gate=True).to(x.dtype))
                out=torch.cat(rows)
            outputs.append(out.flatten(1))
        return F.linear(torch.cat(outputs),weights['attn_output.weight'])

    def experts(self, x, weights, layer):
        logits = F.linear(x,weights['ffn_gate_inp.weight']).float()
        ids = logits.argsort(dim=-1,descending=True,stable=True)[:,:self.topk]
        probabilities = logits.gather(1,ids).softmax(-1)
        counts = torch.bincount(ids.flatten(),minlength=self.e).cpu().tolist()
        gu_name = f'model.language_model.layers.{layer}.mlp.experts.gate_up_proj'
        down_name = f'model.language_model.layers.{layer}.mlp.experts.down_proj'
        if self.source.shape(gu_name) != (self.e,2*self.f,self.h) or self.source.shape(down_name) != (self.e,self.h,self.f):
            raise ValueError('Original expert geometry disagrees with config')
        if any(self.source.info(name)['dtype'] != 'BF16' for name in (gu_name,down_name)):
            raise ValueError('Original BF16 experts required')
        # Merge only contiguous selected experts. Fixed aligned chunks can
        # otherwise download most of a sparse layer through unused neighbours.
        groups=[]
        for expert,count in enumerate(counts):
            if not count:continue
            if groups and expert == groups[-1][0]+groups[-1][1] and groups[-1][1] < self.chunk_experts:
                first,size=groups[-1];groups[-1]=(first,size+1)
            else:groups.append((expert,1))
        result = torch.zeros_like(x,dtype=torch.float32)
        def fetch(group):
            first,count = group
            return self.source.rows(gu_name,first,count),self.source.rows(down_name,first,count)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {}
            for index in range(min(2,len(groups))):pending[index] = pool.submit(fetch,groups[index])
            for index,(first,count) in enumerate(groups):
                gate_up,down = pending.pop(index).result()
                if index+2 < len(groups):pending[index+2] = pool.submit(fetch,groups[index+2])
                gu = torch.from_numpy(gate_up).to(device=x.device,dtype=x.dtype)
                wd = torch.from_numpy(down).to(device=x.device,dtype=x.dtype)
                del gate_up,down
                for offset in range(count):
                    expert = first+offset
                    if not counts[expert]:continue
                    token,slot = torch.where(ids == expert)
                    gate,up = F.linear(x[token],gu[offset]).chunk(2,-1)
                    output = F.linear(F.silu(gate)*up,wd[offset])
                    result.index_add_(0,token,output.float()*probabilities[token,slot,None])
                del gu,wd
        gate = F.linear(x,weights['ffn_gate_shexp.weight'])
        up = F.linear(x,weights['ffn_up_shexp.weight'])
        shared = F.linear(F.silu(gate)*up,weights['ffn_down_shexp.weight'])
        shared = shared*torch.sigmoid(F.linear(x,weights['ffn_gate_inp_shexp.weight']))
        return (result.to(x.dtype)+shared),counts

    def layer(self, residual, layer):
        weights = self.load_weights(layer);x = residual.to(device=self.device)
        if layer+1 in self.text['ple_layer_ids']:
            features = self.get_features().to(device=self.device)
            ple_weights = {label:weights['ple_'+label+'.weight'] for label in ('key','value','norm_key','norm_query','norm_conv')}
            ple_weights['conv'] = weights['ple_conv1d.weight']
            x = torch.cat([ple(x[start:end],features[start:end],ple_weights,dilation=self.text['ngram_size'])
                           for start,end in self.segments])
        mixed,inject = read_streams(x,self.mixer_weights(weights,'hc_attn_'))
        x = write_streams(x,self.attention_block(mixed,weights,layer),inject)
        mixed,inject = read_streams(x,self.mixer_weights(weights,'hc_ffn_'))
        block,counts = self.experts(mixed,weights,layer)
        result = write_streams(x,block,inject).cpu()
        if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite BF16 teacher residual')
        return result,{'layer':layer,'expert_route_counts':counts,'tokens':len(self.tokens)}

    def head(self, residual):
        if self.source.info('lm_head.weight')['dtype'] != 'BF16' or self.source.shape('lm_head.weight') != (self.vocab,self.h):
            raise ValueError('Original BF16 output head required')
        weights = self.load_weights('terminal')
        positions = [start+len(case['prompt_ids'])-1+position
                     for case,(start,end) in zip(self.cases,self.segments) for position in range(len(case['target_ids']))]
        selected = residual[positions].to(device=self.device)
        mixed,_ = read_streams(selected,{key:weights[key+'.weight'] for key in ('norm','down','up')},inject=False)
        logits = np.empty((len(positions),self.vocab),dtype=np.float32)
        for first in range(0,self.vocab,8192):
            count = min(8192,self.vocab-first)
            weight = torch.from_numpy(self.source.rows('lm_head.weight',first,count)).to(device=self.device)
            logits[:,first:first+count] = F.linear(mixed.float(),weight).cpu().numpy()
        probes=[];cursor=0
        for case in self.cases:
            for position in range(len(case['target_ids'])):
                row = probe(logits[cursor],case['prompt_ids'],case['target_ids'],position,case['id'])
                row['execution_mode']='original-bf16-teacher';probes.append(row);cursor+=1
        return probes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--scenes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo',default='Qwen/Qwen3.8-Flash-Next')
    p.add_argument('--revision',required=True)
    p.add_argument('--source-dir',type=Path)
    p.add_argument('--cases',nargs='+')
    p.add_argument('--inputs',type=Path,help='Verified small original embedding/PLE fixtures')
    p.add_argument('--stop-after-layers',type=int)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda',help='Reference arithmetic device; unchanged BF16 weights')
    p.add_argument('--header-cache',type=Path,help='Reuse pinned source headers from conversion')
    a = p.parse_args()
    if a.device == 'cuda':configure()
    else:torch.set_num_threads(2);torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark=False
    scenes = json.loads(a.scenes.read_text())
    if scenes['seed'] != SEED:raise ValueError('Frozen scene seed changed')
    cases = scenes['cases']
    if a.cases:
        if set(a.cases)-{case['id'] for case in cases}:p.error('Unknown fixed scene')
        cases = [case for case in cases if case['id'] in a.cases]
    config = json.loads(a.config.read_text());layers = config['text_config']['num_hidden_layers']
    if a.stop_after_layers is not None and not 0 <= a.stop_after_layers <= layers:p.error('Invalid layer stop')
    a.output.mkdir(parents=True,exist_ok=True)
    source = Original(Source(a.index,directory=a.source_dir,repo=a.repo,revision=a.revision,cache=a.header_cache or a.output/'source-headers'),
                      config,max_read_bytes=128*1024**2)
    source_code = [Path(__file__).with_name(name) for name in ('flash_teacher.py','flash_reference_math.py',
        'flash_original.py','flash_ple.py','flash_roles.py','flash_validation.py','safetensors_source.py','flash_teacher_inputs.py',
        'flash_qsa_reference.py')]
    root=Path(__file__).resolve().parents[2]
    source_code.extend(root/path for path in ('tools/eval/scoring_common.py','tools/operators/common.py','tools/quantization/flash_next.py'))
    contract = {'format':'orinfer.flash_next.bf16_teacher.v1','source':a.repo,'revision':a.revision,
                'index_sha256':digest(a.index),'config_sha256':digest(a.config),'scenes_sha256':digest(a.scenes),
                'cases':[case['id'] for case in cases],'seed':SEED,'torch':torch.__version__,'device':a.device,
                'source_directory':str(a.source_dir.resolve()) if a.source_dir else None,
                'inputs_sha256':digest(a.inputs/'inputs.json') if a.inputs else None,
                'reference_source_sha256':{str(path.relative_to(root)):digest(path) for path in source_code}}
    contract_sha = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    progress_path = a.output/'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'contract':contract,'layers':{},'complete':False}
    if progress['contract'] != contract:raise ValueError('Original teacher resume contract changed')
    for row in progress.get('source_ranges',[]):
        source.reads[f'{row["tensor"]}:{row["first"]}:{row["count"]}'] = row
    inputs=Inputs(a.inputs,source=a.repo,revision=a.revision,config_sha256=digest(a.config),
                  scenes_sha256=digest(a.scenes),index_sha256=digest(a.index)) if a.inputs else None
    if inputs:
        for case in cases:inputs.case(case,config['text_config']['hidden_size'])
        for row in inputs.manifest['source_ranges']:
            source.reads[f'{row["tensor"]}:{row["first"]}:{row["count"]}']=row
    teacher = Teacher(source,cases,inputs=inputs,device=a.device)
    teacher.validate()
    last = -1
    for key,record in sorted(progress['layers'].items(),key=lambda item:int(item[0])):
        if int(key) != last+1 or digest(a.output/record['filename']) != record['sha256']:
            raise ValueError('Missing/corrupt teacher prefix layer')
        last = int(key)
    if progress['complete']:
        if len(progress['layers']) != layers or digest(a.output/'results.json') != progress['results_sha256']:
            raise ValueError('Corrupt completed BF16 teacher result')
        print('Verified completed original-BF16 teacher',flush=True);return
    residual = load_file(a.output/progress['layers'][str(last)]['filename'])['residual'] if last >= 0 else teacher.initial()
    for layer in range(last+1,layers):
        if a.stop_after_layers is not None and layer >= a.stop_after_layers:break
        started = time.perf_counter();residual,record = teacher.layer(residual,layer)
        filename = f'layer-{layer:02d}.safetensors';temporary = a.output/(filename+'.tmp')
        save_file({'residual':residual},temporary,metadata={'contract_sha256':contract_sha,'layer':str(layer)})
        temporary.replace(a.output/filename)
        record.update(filename=filename,sha256=digest(a.output/filename),seconds=time.perf_counter()-started)
        progress['layers'][str(layer)] = record
        progress['source_ranges'] = list(source.reads.values())
        atomic_json(progress_path,progress)
        print(json.dumps({'layer':layer,'seconds':record['seconds'],'expert_coverage':sum(c > 0 for c in record['expert_route_counts'])}),flush=True)
    if len(progress['layers']) == layers:
        probes = teacher.head(residual)
        observed = environment() if a.device == 'cuda' else {'torch':torch.__version__,'device':'cpu',
            'cpu_threads':torch.get_num_threads(),'seed':SEED,'cuda_initialized':torch.cuda.is_initialized()}
        report = {'complete':True,'baseline_precision':'original-bf16','scope':__doc__,
                  'contract':contract,'environment':observed,'probes':probes,'full_model_quality_verified':False}
        atomic_json(a.output/'results.json',report)
        progress['source_ranges'] = list(source.reads.values())
        progress['complete']=True;progress['results_sha256']=digest(a.output/'results.json')
    atomic_json(progress_path,progress)


if __name__ == '__main__':main()
