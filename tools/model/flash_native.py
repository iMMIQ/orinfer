"""Offline full-model execution of our Flash Next E8P/A8 checkpoint.

Python is diagnostic control only. All GPU arithmetic uses native TileLang
kernels. Packed experts are never expanded globally; dense weights remain
INT8 and HC/router coefficients retain BF16. PLE is looked up on the CPU.
QSA uses native group-4 indexing, INT8 group-64 KV and sparse selection
through 262144 tokens. Only the current preparation tile is temporarily FP16.
This validation recipe is not the online Rust architecture adapter or BF16
teacher. Its token results still require independent original-model comparison.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from kernels.model import flash_next as fn
from kernels.model import qsa
from kernels.model import qsa_attention as qsat
from kernels.model.hyperconnection import hc_norm, hc_silu, hc_projection, hc_mix, hc_combine
from kernels.model.ple import ple_gate, ple_conv, ple_history
from kernels.model.gdn_sequence import gdn_sequence
from kernels.model.greedy import greedy_partials, greedy_merge
from kernels.model.int8_projection import int8_projection, int8_gemv
from kernels.model.integer_vq import integer_vq_grouped, rotate_activation
from kernels.model.moe import router_topk, expert_histogram, expert_offsets, expert_tiles, expert_dispatch, moe_combine
from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
from kernels.operators.op09_gdn_gates import gdn_gates, launch as launch_gates
from kernels.operators.op30_activation_quantization import activation_quantization, launch as launch_quant
from tools.model.flash_checkpoint import Checkpoint
from tools.model.flash_chunks import chunks
from tools.model.flash_lookup import RowCache
from tools.model.flash_ple import PleLookup
from tools.model.flash_roles import role
from tools.model.flash_validation import validate_snapshot, probe, baseline_probes
from tools.operators.common import configure, environment, export_kernel, write_json


class Model:
    H, F, E, K, C, R, V = 2560, 640, 512, 10, 4, 320, 248320

    def __init__(self, source, capacity, output, *, use_graph=True):
        maximum = source.config['text_config']['max_position_embeddings']
        if type(capacity) is not int or not 1 <= capacity <= min(maximum,262144):
            raise ValueError('Context exceeds the checkpoint/native 262144 limit')
        self.source, self.capacity, self.output = source, capacity, output
        self.weights, self.states, self.plans = {}, {}, {}
        self.prefix_divisors = {}
        self.kernels = {}
        self.history = []
        self.position = 0
        self.ple = PleLookup(source)
        self.embedding = RowCache(source,capacity_bytes=8*1024**2)
        self.use_graph = use_graph
        self.position_gpu = torch.zeros(1, device='cuda', dtype=torch.int32)
        self.lengths = torch.empty_like(self.position_gpu)
        self.position_out = torch.empty_like(self.position_gpu)
        self.load()

    def load(self):
        text = self.source.config['text_config']
        expected = {'hidden_size':self.H,'moe_intermediate_size':self.F,'num_experts':self.E,
                    'num_experts_per_tok':self.K,'hc_count':self.C,'hc_lowrank':self.R,
                    'vocab_size':self.V,'num_hidden_layers':48,'head_dim':256,
                    'num_attention_heads':24,'num_key_value_heads':2,'linear_num_key_heads':16,
                    'linear_num_value_heads':48,'linear_key_head_dim':128,'linear_value_head_dim':128,
                    'linear_conv_kernel_dim':4,'shared_expert_intermediate_size':self.F,
                    'rms_norm_eps':1e-6,'hidden_act':'silu','output_gate_type':'sigmoid',
                    'ple_layer_ids':[2],'ple_conv_kernel_size':4,'ple_embed_dim':self.H,
                    'ngram_size':3,'heads_per_ngram':8,'indexer_budget':2048,
                    'indexer_compress_ratio':4,'indexer_head_dim':128,
                    'indexer_n_heads':4,'indexer_kv_heads':1}
        if any(text.get(key) != value for key,value in expected.items()):
            raise ValueError('Checkpoint geometry differs from initial native Flash Next recipe')
        if text['layer_types'] != ['linear_attention','linear_attention','linear_attention','full_attention']*12:
            raise ValueError('Unsupported Flash Next layer sequence')
        rope = text.get('rope_parameters',{})
        if (rope.get('rope_theta') != 1e7 or rope.get('partial_rotary_factor') != .25
                or rope.get('rope_type') != 'default' or text.get('norm_topk_prob',True) is not True):
            raise ValueError('Unsupported rotation or expert normalization semantics')
        for i in range(3,48,4):
            prefix=f'model.language_model.layers.{i}.self_attn.indexer.'
            for suffix,shape in [('index_qk_proj.weight',(640,self.H)),
                                 ('q_layernorm.weight',(128,)),('k_layernorm.weight',(128,))]:
                if prefix+suffix not in self.source.parts or self.source.shape(prefix+suffix)!=shape:
                    raise ValueError('Missing or mismatched native QSA indexer weight')
        for original in sorted(self.source.parts):
            name = role(original)
            if name is None or name == 'token_embd.weight':continue
            shape = self.source.shape(original)
            kind = self.source.kind(original)
            if kind == 'e8p-expert':
                e,n,k = shape
                packed = torch.empty((e,k//128,n,16),device='cuda',dtype=torch.uint16)
                scales = torch.empty((e,n),device='cuda',dtype=torch.float16)
                table,signs = None,None
                for first,part in self.source.expert_parts(original):
                    count = len(part['scales'])
                    packed[first:first+count].copy_(torch.from_numpy(part['indices']))
                    scales[first:first+count].copy_(torch.from_numpy(part['scales']))
                    if table is None:table,signs = part['table'].copy(),part['signs'].copy()
                    elif not np.array_equal(table,part['table']) or not np.array_equal(signs,part['signs']):
                        raise ValueError('Expert bank does not share its declared basis/rotation')
                book = torch.from_numpy(table.view('<u4').reshape(1,256,2).copy()).cuda()
                self.weights[name] = {'packed':packed,'table':book,'scale':scales,
                                      'signs':torch.from_numpy(signs).cuda()}
                if name.endswith('ffn_down_exps.weight'):print('loaded expert layer',name.split('.')[1],flush=True)
                continue
            if kind == 'int8-row':
                n,k = shape
                integer = torch.empty((n,k),device='cuda',dtype=torch.int8)
                scale = torch.empty(n,device='cuda',dtype=torch.float16)
                for first,part,s in self.source.int8_parts(original):
                    integer[first:first+len(part)].copy_(torch.from_numpy(part))
                    scale[first:first+len(s)].copy_(torch.from_numpy(s))
                self.weights[name] = (integer,scale)
                continue
            if kind != 'original':raise ValueError('Unsupported GPU weight kind')
            dtype = torch.bfloat16
            if 'norm.weight' in name or 'ple_norm_' in name or name.endswith(('ssm_a_log','ssm_dt.bias','ssm_conv1d.weight')):
                dtype = torch.float32
            tensor = torch.from_numpy(self.source.tensor(original)).to(device='cuda',dtype=dtype)
            if name.endswith(('attn_q_norm.weight','attn_k_norm.weight','index_q_norm.weight','index_k_norm.weight')):
                # Original Gemma RMSNorm stores zero-centered weights. The
                # QSA kernel consumes ordinary FP32 gamma; add one exactly.
                tensor = tensor+1.0
            if tensor.ndim == 3 and shape[1] == 1:tensor = tensor.squeeze(1)
            if 'hc_' in name and name.endswith('norm.weight') or 'ple_norm_' in name:
                tensor = tensor.reshape(self.C,self.H)
            self.weights[name] = tensor
        for i in range(48):
            if i%4 != 3:
                self.states[f'{i}:conv'] = torch.zeros((1,3,10240),device='cuda',dtype=torch.float16)
                self.states[f'{i}:gdn'] = torch.zeros((48,128,128),device='cuda')
            else:
                for suffix in ('key','value'):
                    self.states[f'{i}:{suffix}'] = torch.empty((self.capacity,2,256),device='cuda',dtype=torch.int8)
                    self.prefix_divisors[f'{i}:{suffix}'] = 1
                    self.states[f'{i}:{suffix}_scale'] = torch.empty((self.capacity,2,4),device='cuda',dtype=torch.float16)
                    self.prefix_divisors[f'{i}:{suffix}_scale'] = 1
                self.states[f'{i}:index'] = torch.empty(((self.capacity+3)//4,128),device='cuda',dtype=torch.float16)
                self.prefix_divisors[f'{i}:index'] = 4
                self.states[f'{i}:pending'] = torch.zeros((4,128),device='cuda',dtype=torch.float16)
        self.states['ple'] = torch.zeros((9,self.C*self.H),device='cuda',dtype=torch.float16)
        print('resident GPU GiB',torch.cuda.memory_allocated()/2**30,flush=True)

    def kernel(self, key, build):
        if key not in self.kernels:
            self.kernels[key] = build()
            export_kernel(self.kernels[key], self.output / 'aot' / key)
        return self.kernels[key]

    def plan(self, m):
        if m in self.plans:
            return self.plans[m]
        if not 1 <= m <= 128:
            raise ValueError('Recurrent plan supports chunks of 1..128')
        h, f, c, rank, e, k = self.H, self.F, self.C, self.R, self.E, self.K
        ops = []
        labels = []
        def empty(shape, dtype=torch.float16):
            return torch.empty(shape, device='cuda', dtype=dtype)
        def call(label, build, *args):
            kernel = self.kernel(label, build)
            ops.append((kernel, args))
            labels.append(label)
        dense_workspace = {}
        def quantize(a,rows=None):
            rows=m if rows is None else rows
            width=a.shape[-1]
            if (width,rows) not in dense_workspace:
                dense_workspace[width,rows] = (empty((rows,width),torch.int8),empty((rows,1)),
                                              torch.zeros(width,device='cuda',dtype=torch.uint8))
            q,as_,mask = dense_workspace[width,rows]
            quant = self.kernel(f'dense-quant-{width}',lambda:activation_quantization(width))
            ops.append((lambda x,mask,q,scale,quant=quant:launch_quant(quant,x,mask,q,scale,
                        stream=torch.cuda.current_stream().cuda_stream),(a,mask,q,as_)))
            labels.append(f'dense-quant-{width}')
            return q,as_
        def projection(label,a,name,out,hc=False,rows=None,a8=None):
            rows=m if rows is None else rows
            weight = self.weights[name]
            if isinstance(weight,tuple):
                integer,scale = weight;n,width = integer.shape
                q,as_ = quantize(a,rows) if a8 is None else a8
                od = str(out.dtype).split('.')[-1]
                if rows==1 and 2048<=n<self.V and n*width>=8*1024**2 and width%128==0:
                    call(f'dense-gemv-{n}-{width}-{od}',lambda:int8_gemv(n,width,od,8),
                         q.view(torch.int32),integer.view(torch.int32),scale,as_.view(-1),out)
                else:
                    bm=64 if rows>=64 and n>=512 else 32 if rows>=32 and n>=512 else 16
                    call(f'dense-a8-{rows}-{n}-{width}-{od}-{bm}',lambda:int8_projection(rows,n,width,od,bm),
                         q,integer,scale,as_.view(-1),out)
                labels[-1]=f'{label}:{labels[-1]}'
            else:
                n,width = weight.shape
                if hc:
                    bm=64 if m>=64 and n>=c*h else 16
                    bn=32 if m==1 and n<=rank else 64
                    call(f'hc-{m}-{n}-{width}-{bm}-{bn}',lambda:hc_projection(m,n,width,block_m=bm,dtype='float16',block_n=bn),a,weight,out)
                else:
                    wd = str(weight.dtype).split('.')[-1];od = str(out.dtype).split('.')[-1]
                    call(f'dense-{m}-{n}-{width}-{wd}-{od}',lambda:fn.dense_projection(m,n,width,wd,od),a,weight,out)
                labels[-1]=f'{label}:{labels[-1]}'
        embedding, ple_embedding = empty((m,h)), empty((m,h))
        residual, normed, up = (empty((m,c,h)) for _ in range(3))
        down, activated = empty((m,rank)), empty((m,rank))
        mixed, block = empty((m,h)), empty((m,h))
        inject_attn, inject_ffn = empty((m,c)), empty((m,c))
        def mixer(prefix, inject):
            call(f'hc-norm-{m}', lambda: hc_norm(m,h,c,dtype='float16'),residual,self.weights[prefix+'norm.weight'],normed)
            projection('hc-down',normed.flatten(1),prefix+'down.weight',down,True)
            call(f'hc-silu-{m}',lambda:hc_silu(m,rank,c,'float16'),down,activated)
            projection('hc-up',activated,prefix+'up.weight',up.flatten(1),True)
            call(f'hc-mix-{m}',lambda:hc_mix(m,h,c,'float16'),normed,up,mixed)
            if inject is not None:
                projection('hc-inject',normed.flatten(1),prefix+'inject.weight',inject,True)
        call(f'initialize-{m}',lambda:fn.hc_initialize(m),embedding,residual)
        qkv, z = empty((m,10240)),empty((m,6144))
        alpha,beta_raw = empty((m,48)),empty((m,48))
        g,beta = empty((m,48),torch.float32),empty((m,48),torch.float32)
        q,keys,values = empty((1,16,m,128)),empty((1,16,m,128)),empty((1,48,m,128))
        ho = empty((1,3,10240))
        prefix_state = empty((1,48,128,128),torch.float32)  # In-place GDN never reads/writes Prefix.
        recurrent,gated = empty((m,6144)),empty((m,6144))
        qgate,qsa_k,qsa_v = empty((m,12288)),empty((m,512)),empty((m,512))
        qsa_query,qsa_gate,qsa_out = (empty((m,24,256)) for _ in range(3))
        staged_key,staged_value=(empty((m,2,256)) for _ in range(2))
        index_qk,index_q = empty((m,640)),empty((m,4,128))
        blocks=(self.capacity+3)//4;segments=(blocks+1023)//1024
        index_scores=empty((m,blocks),torch.float32)
        index_prefix,index_remaining,index_greater=(empty((m,),torch.int32) for _ in range(3))
        index_hist=empty((m,segments,256),torch.int32)
        index_counts,index_offsets=(empty((m,2,segments),torch.int32) for _ in range(2))
        selected=empty((m,2051),torch.int32)
        sparse_max,sparse_den=(empty((m,24,8),torch.float32) for _ in range(2))
        sparse_out=empty((m,24,8,256),torch.float32)
        ple_key,ple_normed_key,ple_normed_query,ple_gated,ple_normed,ple_out = (empty((m,c,h)) for _ in range(6))
        ple_value = empty((m,h))
        logits = empty((m,e),torch.float32)
        ids,prob = empty((m,k),torch.int32),empty((m,k),torch.float32)
        counts,offsets,tile_offsets = (empty((e,),torch.int32) for _ in range(3))
        relative,slot_map = empty((m,k),torch.int32),empty((m,k),torch.int32)
        assignments=m*k
        tiles=(assignments+15)//16+min(e,assignments)
        tile_expert,tile_row,tile_count = empty((tiles,),torch.int32),empty((tiles,),torch.int32),empty((1,),torch.int32)
        rotated = empty((m,h))
        aq,sa = empty((m,h),torch.int8),empty((m,1))
        dispatch,ds = empty((assignments,h),torch.int8),empty((assignments,1))
        gu,rotated_ffn = empty((assignments,2*f)),empty((assignments,f))
        fq,fs = empty((assignments,f),torch.int8),empty((assignments,1))
        patch = torch.zeros(1,device="cuda",dtype=torch.uint16)
        expert_out = empty((assignments,h))
        shared_gate,shared_up,shared_act = (empty((m,f)) for _ in range(3))
        shared_out,shared_logit = empty((m,h)),empty((m,1))
        masks = [torch.zeros(width,device='cuda',dtype=torch.uint8) for width in (h,f)]
        quant = self.kernel('a8-hidden',lambda:activation_quantization(h))
        ffn_quant = self.kernel('a8-ffn',lambda:activation_quantization(f))
        gates = self.kernel('gdn-gates',lambda:gdn_gates(parameter_mode='a_log'))
        def quant_call(kernel,*args):
            ops.append((lambda *items:launch_quant(kernel,*items,stream=torch.cuda.current_stream().cuda_stream),args))
            labels.append('expert-quant')
        for i in range(48):
            prefix=f'blk.{i}.'
            if i == 1:
                ple_a8=quantize(ple_embedding)
                projection('ple-key',ple_embedding,prefix+'ple_key.weight',ple_key.flatten(1),a8=ple_a8)
                projection('ple-value',ple_embedding,prefix+'ple_value.weight',ple_value,a8=ple_a8)
                for x,w,out in [(ple_key,'ple_norm_key',ple_normed_key),(residual,'ple_norm_query',ple_normed_query)]:
                    call(f'hc-norm-{m}',lambda:hc_norm(m,h,c,dtype='float16'),x,self.weights[prefix+w+'.weight'],out)
                call(f'ple-gate-{m}',lambda:ple_gate(m,h,c),ple_normed_key,ple_normed_query,ple_value,ple_gated)
                call(f'hc-norm-{m}',lambda:hc_norm(m,h,c,dtype='float16'),ple_gated,self.weights[prefix+'ple_norm_conv.weight'],ple_normed)
                call(f'ple-conv-{m}',lambda:ple_conv(m,c*h),ple_normed.flatten(1),ple_gated.flatten(1),self.states['ple'],self.weights[prefix+'ple_conv1d.weight'].half(),ple_out.flatten(1))
                call(f'ple-history-{m}',lambda:ple_history(m,c*h),ple_normed.flatten(1),self.states['ple'],self.states['ple'])
                call(f'ple-add-{m}',lambda:fn.residual_add(m,c*h),residual.flatten(1),ple_out.flatten(1),residual.flatten(1))
            mixer(prefix+'hc_attn_',inject_attn)
            attn_a8=quantize(mixed)
            if i % 4 != 3:
                projection('gdn-qkv',mixed,prefix+'attn_qkv.weight',qkv,a8=attn_a8)
                projection('gdn-z',mixed,prefix+'attn_gate.weight',z,a8=attn_a8)
                projection('gdn-alpha',mixed,prefix+'ssm_alpha.weight',alpha,a8=attn_a8)
                projection('gdn-beta',mixed,prefix+'ssm_beta.weight',beta_raw,a8=attn_a8)
                ops.append((lambda a,b,p,dt,g,beta:launch_gates(gates,a,b,p,dt,g,beta,stream=torch.cuda.current_stream().cuda_stream),
                    (alpha,beta_raw,self.weights[prefix+'ssm_a_log'],self.weights[prefix+'ssm_dt.bias'],g,beta)))
                labels.append('gdn-gates')
                call(f'gdn-conv-{m}',lambda:gdn_conv_prep(B=1,tokens=m,normalize_round_fp16=False,
                    conv_product_round_fp16=False,weight_dtype='float32'),qkv.view(1,m,10240),self.weights[prefix+'ssm_conv1d.weight'],self.states[f'{i}:conv'],self.lengths,self.position_gpu,q,keys,values,ho,self.position_out)
                ops.append((lambda source,destination:destination.copy_(source),(ho,self.states[f'{i}:conv'])))
                labels.append('gdn-history-copy')
                call(f'gdn-sequence-{m}',lambda:gdn_sequence(m,in_place=True),q[0],keys[0],values[0],g,beta,self.states[f'{i}:gdn'],prefix_state,recurrent)
                call(f'gdn-sigmoid-{m}',lambda:fn.gdn_sigmoid_norm(m),recurrent.view(m,48,128),z.view(m,48,128),self.weights[prefix+'ssm_norm.weight'],gated.view(m,48,128))
                projection('gdn-out',gated,prefix+'ssm_out.weight',block)
            else:
                projection('qsa-query',mixed,prefix+'attn_q.weight',qgate,a8=attn_a8)
                projection('qsa-key',mixed,prefix+'attn_k.weight',qsa_k,a8=attn_a8)
                projection('qsa-value',mixed,prefix+'attn_v.weight',qsa_v,a8=attn_a8)
                call(f'qsa-prepare-{m}',lambda:fn.qsa_prepare(m,self.capacity,is_neox_style=True,staged=True),qgate.view(m,24,2,256),qsa_k.view(m,2,256),qsa_v.view(m,2,256),self.weights[prefix+'attn_q_norm.weight'],self.weights[prefix+'attn_k_norm.weight'],self.position_gpu,qsa_query,staged_key,staged_value,qsa_gate)
                call(f'qsa-kv-store-{m}',lambda:qsa.kv_store(m,self.capacity),staged_key,staged_value,self.position_gpu,self.states[f'{i}:key'],self.states[f'{i}:value'],self.states[f'{i}:key_scale'],self.states[f'{i}:value_scale'])
                projection('index-qk',mixed,prefix+'index_qk.weight',index_qk,a8=attn_a8)
                call(f'index-query-{m}',lambda:qsa.index_query(m),index_qk.view(m,5,128),self.weights[prefix+'index_q_norm.weight'],self.position_gpu,index_q)
                call(f'index-compress-{m}',lambda:qsa.index_compress(m,self.capacity),index_qk.view(m,5,128),self.states[f'{i}:pending'],self.weights[prefix+'index_k_norm.weight'],self.position_gpu,self.states[f'{i}:index'])
                call(f'index-pending-{m}',lambda:qsa.index_pending(m),index_qk.view(m,5,128),self.position_gpu,self.states[f'{i}:pending'])
                call(f'index-scores-{m}',lambda:qsa.index_scores(m,self.capacity),index_q,self.states[f'{i}:index'],self.position_gpu,index_scores)
                for shift in (24,16,8,0):
                    call(f'index-hist-{m}-{shift}',lambda shift=shift:qsa.radix_histogram(m,self.capacity,shift),index_scores,index_prefix,self.position_gpu,index_hist)
                    call(f'index-choose-{m}-{shift}',lambda shift=shift:qsa.radix_choose(m,self.capacity,shift),index_hist,index_prefix,index_remaining,self.position_gpu)
                call(f'index-counts-{m}',lambda:qsa.selection_counts(m,self.capacity),index_scores,index_prefix,self.position_gpu,index_counts)
                call(f'index-offsets-{m}',lambda:qsa.selection_offsets(m,self.capacity),index_counts,index_offsets,index_greater,self.position_gpu,selected)
                call(f'index-scatter-{m}',lambda:qsa.selection_scatter(m,self.capacity),index_scores,index_prefix,index_offsets,index_greater,self.position_gpu,selected)
                call(f'qsa-sparse-packed-{m}',lambda:qsat.sparse_attention(m,self.capacity,packed=True),
                     qsa_query,self.states[f'{i}:key'].view(torch.uint32),self.states[f'{i}:value'].view(torch.uint32),
                     self.states[f'{i}:key_scale'],self.states[f'{i}:value_scale'],selected,self.position_gpu,sparse_max,sparse_den,sparse_out)
                call(f'qsa-merge-{m}',lambda:qsa.sparse_merge(m),sparse_max,sparse_den,sparse_out,qsa_gate,qsa_out)
                projection('qsa-out',qsa_out.flatten(1),prefix+'attn_output.weight',block)
            call(f'hc-combine-{m}',lambda:hc_combine(m,h,c,'float16'),block,residual,inject_attn,residual)
            mixer(prefix+'hc_ffn_',inject_ffn)
            projection('router',mixed,prefix+'ffn_gate_inp.weight',logits)
            call(f'router-{m}',lambda:router_topk(m,e,k),logits,ids,prob)
            call(f'histogram-{m}',lambda:expert_histogram(m,e,k),ids,counts,relative)
            call('offsets',lambda:expert_offsets(e),counts,offsets,tile_offsets,tile_count)
            call(f'tiles-{m}',lambda:expert_tiles(m,e,k),counts,tile_offsets,tile_expert,tile_row)
            gate_weight = self.weights[prefix+'ffn_gate_up_exps.weight']
            down_weight = self.weights[prefix+'ffn_down_exps.weight']
            call(f'rotate-hidden-{m}',lambda:rotate_activation(m,h),mixed,gate_weight['signs'],rotated)
            quant_call(quant,rotated,masks[0],aq,sa)
            call(f'dispatch-{m}',lambda:expert_dispatch(m,h,e,k,scale_group=h),aq,sa,ids,relative,offsets,dispatch,ds,slot_map)
            call(f'expert-gu-{m}',lambda:integer_vq_grouped(assignments,e,tiles,2*f,h,kind='e8p',shared_table=True),
                 dispatch,gate_weight['packed'],gate_weight['table'],patch,gate_weight['scale'],ds.view(-1),
                 counts,offsets,tile_expert,tile_row,tile_count,gu)
            call(f'rotate-ffn-{m}',lambda:rotate_activation(assignments,f,swiglu=True),gu,down_weight['signs'],rotated_ffn)
            quant_call(ffn_quant,rotated_ffn,masks[1],fq,fs)
            call(f'expert-down-{m}',lambda:integer_vq_grouped(assignments,e,tiles,h,f,kind='e8p',shared_table=True),
                 fq,down_weight['packed'],down_weight['table'],patch,down_weight['scale'],fs.view(-1),
                 counts,offsets,tile_expert,tile_row,tile_count,expert_out)
            shared_a8=quantize(mixed)
            projection('shared-gate',mixed,prefix+'ffn_gate_shexp.weight',shared_gate,a8=shared_a8)
            projection('shared-up',mixed,prefix+'ffn_up_shexp.weight',shared_up,a8=shared_a8)
            call(f'shared-swiglu-{m}',lambda:fn.swiglu(m,f),shared_gate,shared_up,shared_act)
            projection('shared-down',shared_act,prefix+'ffn_down_shexp.weight',shared_out)
            projection('shared-router',mixed,prefix+'ffn_gate_inp_shexp.weight',shared_logit,a8=shared_a8)
            call(f'combine-{m}',lambda:moe_combine(m,h,assignments,k),expert_out,slot_map,prob,shared_out,shared_logit.view(m),block)
            call(f'hc-combine-{m}',lambda:hc_combine(m,h,c,'float16'),block,residual,inject_ffn,residual)
        body_count=len(ops)
        mixer('output_hc_',None)
        output=empty((1,self.V),torch.float32)
        projection('head',mixed[-1:],'output.weight',output,rows=1)
        blocks=(self.V+1023)//1024
        top_values=empty((blocks,),torch.float32)
        top_indices,top_invalid=(empty((blocks,),torch.int32) for _ in range(2))
        token=empty((2,),torch.int32)
        greedy_ops=[(self.kernel('greedy-partials',lambda:greedy_partials(self.V)),
                     (output,top_values,top_indices,top_invalid)),
                    (self.kernel('greedy-merge',lambda:greedy_merge(self.V)),
                     (top_values,top_indices,top_invalid,token))]
        plan={'ops':ops,'body_count':body_count,'greedy_ops':greedy_ops,'token':token,
              'labels':labels,'embedding':embedding,'ple_embedding':ple_embedding,'output':output,
              'residual':residual,'graphs':{}}
        self.plans[m]=plan
        print('compiled plan',m,'ops',len(ops),flush=True)
        return plan

    def reset(self):
        for name,state in self.states.items():
            if name not in self.prefix_divisors:state.zero_()
        self.history=[];self.position=0;self.position_gpu.zero_()

    def live_states(self, position=None):
        position=self.position if position is None else position
        return {name:value[:position//self.prefix_divisors[name]] if name in self.prefix_divisors else value
                for name,value in self.states.items()}

    def snapshot(self, *, cpu=False):
        return {'states':{name:value.to(device="cpu",copy=True) if cpu else value.clone()
                          for name,value in self.live_states().items()},
                'history':list(self.history),'position':self.position}

    def restore(self, snapshot):
        validate_snapshot(snapshot,self.states,self.capacity,self.V,self.ple.ngram-1,
                          prefix_divisors=self.prefix_divisors,allow_cpu=True)
        if self.ple.eos in snapshot['history']:
            raise ValueError('PLE history contains a reset token')
        for name,value in snapshot['states'].items():
            live=self.states[name]
            if name in self.prefix_divisors:live=live[:snapshot['position']//self.prefix_divisors[name]]
            live.copy_(value)
        self.history=list(snapshot['history']);self.position=int(snapshot['position'])
        self.position_gpu.fill_(self.position)

    def execute(self, tokens, *, output='logits'):
        if output not in ('logits','none','token'):
            raise ValueError('Output must be logits, none or token')
        if not tokens or self.position+len(tokens)>self.capacity:
            raise ValueError('Request exceeds native QSA context budget')
        plan=self.plan(len(tokens))
        embedding=np.stack([self.embedding.read('model.language_model.embed_tokens.weight',token) for token in tokens])
        features,next_history=self.ple.prepare(tokens,self.history)
        plan['embedding'].copy_(torch.from_numpy(embedding).half())
        plan['ple_embedding'].copy_(torch.from_numpy(features).half())
        self.position_gpu.fill_(self.position);self.lengths.fill_(len(tokens))
        ops=plan['ops'][:plan['body_count']] if output=='none' else plan['ops']
        if output=='token':ops=ops+plan['greedy_ops']
        if self.use_graph:
            if output not in plan['graphs']:
                snapshot=self.snapshot(cpu=True)
                for kernel,arguments in ops:kernel(*arguments)
                torch.cuda.synchronize();self.restore(snapshot)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for kernel,arguments in ops:kernel(*arguments)
                self.restore(snapshot);plan['graphs'][output]=graph
                del snapshot
            plan['graphs'][output].replay()
        else:
            for kernel,arguments in ops:kernel(*arguments)
        result=None
        if output=='logits':
            result=plan['output'][-1].cpu().numpy()
            if not np.isfinite(result).all():raise ValueError('Nonfinite model logits')
        elif output=='token':
            selected=plan['token'].cpu().numpy()
            if selected[1]:raise ValueError('Nonfinite model logits')
            result=int(selected[0])
        self.history=next_history;self.position+=len(tokens)
        return result


def prefill(model, tokens, chunk, *, output='logits'):
    """Run bucketed chunks; the output head is needed only at the final token."""
    if output not in ('logits','none','token'):
        raise ValueError('Output must be logits, none or token')
    cursor = 0
    for batch in chunks(tokens,chunk):
        cursor+=len(batch)
        result=model.execute(batch,output=output if cursor==len(tokens) else 'none')
    return result


def generate(model, logits, eos, budget):
    """Greedy generation with GPU selection; full logits remain available for probes."""
    if type(budget) is not int or budget<1:raise ValueError('Positive output budget required')
    token=int(np.argmax(logits));tokens=[];reason='length'
    for _ in range(budget):
        tokens.append(token)
        if token in eos:reason='stop';break
        if len(tokens)==budget:break
        token=model.execute([token],output='token')
    return tokens,reason


def state_checks(model, tokens):
    """Real chunk, branch/restore, changed-input graph and request isolation."""
    model.reset();chunk_logits = model.execute(tokens)
    chunk = model.snapshot()
    model.reset()
    for token in tokens:serial_logits = model.execute([token])
    serial = model.snapshot()
    from tools.operators.common import error
    state_errors = {name:error(serial['states'][name],value) for name,value in chunk['states'].items()}
    logits_error = error(torch.from_numpy(serial_logits),torch.from_numpy(chunk_logits))
    if serial['history'] != chunk['history'] or serial['position'] != chunk['position']:
        raise ValueError('Chunking changed private PLE history or position')
    # Chunk arithmetic may round differently; record all actual state errors.
    if not logits_error['finite'] or any(not x['finite'] for x in state_errors.values()):
        raise ValueError('Nonfinite state in chunk comparison')
    continuation = [int(np.argmax(chunk_logits))]
    model.restore(chunk);expected = model.execute(continuation)
    after = model.snapshot()
    model.reset();model.execute([int((tokens[0]+37)%model.V)])
    model.restore(chunk);actual = model.execute(continuation)
    if not np.array_equal(actual,expected):raise ValueError('Prefix replay logits changed')
    if any(not torch.equal(model.live_states()[name],value) for name,value in after['states'].items()):
        raise ValueError('Prefix replay state changed')
    if model.history != after['history'] or model.position != after['position']:
        raise ValueError('Prefix replay CPU history changed')
    model.reset();fresh = model.execute(tokens)
    if not np.array_equal(fresh,chunk_logits):raise ValueError('Fresh request depends on previous request')
    if not any(x['relative_l2'] > 0 for x in state_errors.values()):
        chunking_status = 'exact'
    else:chunking_status = 'numerical_difference_requires_quality_review'
    return {'prefix_restore_exact':True,'fresh_request_exact':True,
            'changed_input_graph_replayed':model.use_graph,'chunking_status':chunking_status,
            'chunk_logits_error':logits_error,'chunk_state_errors':state_errors}


def main():
    from tools.model.flash_scenes import scenes
    from tools.eval.scoring_common import pair_probes, validate_task, SEED
    from tools.operators.common import identity
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--context',type=int,default=262144)
    p.add_argument('--chunk',type=int,choices=(1,2,4,8,16,32,64,128),default=8)
    p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--graph',choices=('on','off'),default='on')
    p.add_argument('--baseline',type=Path,help='Original BF16/FP8 probes on exactly these forced histories')
    p.add_argument('--cases',nargs='+',help='Run selected scene IDs')
    a = p.parse_args()
    if a.max_new_tokens < 1:p.error('--max-new-tokens must be positive')
    configure();np.random.seed(SEED);a.output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter();source = Checkpoint(a.checkpoint)
    tokenizer,cases = scenes(a.checkpoint)
    if a.cases:
        if set(a.cases)-{case['id'] for case in cases}:p.error('Unknown scene ID')
        cases = [case for case in cases if case['id'] in a.cases]
    for case in cases:
        if len(case['prompt_ids'])+max(a.max_new_tokens,len(case['target_ids'])) > a.context:
            p.error('Scene exceeds context budget')
    baseline = baseline_probes(json.loads(a.baseline.read_text()),source.config['quantization_config'],cases) if a.baseline else []
    baseline_map = {(row['case_id'],row['position']):row for row in baseline}
    if len(baseline_map) != len(baseline):raise ValueError('Duplicate baseline scoring positions')
    model = Model(source,a.context,a.output,use_graph=a.graph=='on')
    torch.cuda.synchronize();load_s = time.perf_counter()-started
    write_json(a.output/'scenes.json',{'seed':SEED,'cases':cases,
        'frontend':[identity(a.checkpoint/name) for name in ('tokenizer.json','chat_template.jinja')]})
    warm_started = time.perf_counter();model.plan(1);model.plan(a.chunk)
    checks = state_checks(model,cases[0]['prompt_ids'][:a.chunk])
    model.reset();model.execute(cases[0]['prompt_ids'][:1],output='token');model.reset()
    torch.cuda.synchronize();warm_s = time.perf_counter()-warm_started
    generation_config = json.loads((a.checkpoint/'generation_config.json').read_text())
    eos = generation_config['eos_token_id'];eos = {eos} if isinstance(eos,int) else set(eos)
    rows,probes = [],[]
    for case in cases:
        model.reset();begin = time.perf_counter()
        logits = prefill(model,case['prompt_ids'],a.chunk)
        prefill_s = time.perf_counter()-begin;prefix = model.snapshot();prompt_logits = logits.copy()
        decode_started = time.perf_counter()
        tokens,reason=generate(model,logits,eos,a.max_new_tokens)
        decode_s = time.perf_counter()-decode_started
        text = tokenizer.decode(tokens,skip_special_tokens=True)
        model.restore(prefix)
        # Graph workspaces are not prefix state. Keep the actual prompt logits.
        logits = prompt_logits
        case_probes = []
        for position,target in enumerate(case['target_ids']):
            base = baseline_map.get((case['id'],position))
            queries = [x['token_id'] for x in base['top3']] if base else []
            row = probe(logits,case['prompt_ids'],case['target_ids'],position,case['id'],queries)
            case_probes.append(row)
            if position+1 < len(case['target_ids']):logits = model.execute([target])
        probes.extend(case_probes)
        write_json(a.output/'probes.json',{'seed':SEED,'probes':probes})
        row = {'id':case['id'],'thinking':case['thinking'],'prompt_tokens':len(case['prompt_ids']),
               'generated_token_ids':tokens,'text':text,'finish_reason':reason,
               'prefill_s':prefill_s,'prefill_tps':len(case['prompt_ids'])/prefill_s,
               'decode_s':decode_s,'decode_steps':max(0,len(tokens)-1),
               'decode_tps':max(0,len(tokens)-1)/max(decode_s,1e-9),
               'target_nll':-float(np.mean([x['reference_logprob'] for x in case_probes])),
               'task':validate_task(case['rule'],text)}
        rows.append(row);write_json(a.output/'requests.json',rows)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    paired = []
    if baseline:
        selected = [baseline_map[(x['case_id'],x['position'])] for x in probes]
        # Pairing labels identify the numerical path; histories remain exact.
        normalized = [dict(row,execution_mode='flash-native') for row in selected]
        paired = pair_probes(normalized,probes)
    write_json(a.output/'results.json',{'execution_complete':True,'full_model_quality_verified':False,
        'scope':__doc__,'seed':SEED,'environment':environment(),'load_s':load_s,'warm_s':warm_s,
        'graph':a.graph,'chunk':a.chunk,'state_checks':checks,'requests':rows,'probes':probes,
        'baseline':identity(a.baseline) if a.baseline else None,'paired_probes':paired,
        'checkpoint_index':identity(a.checkpoint/'model.safetensors.index.json')})


if __name__ == '__main__':main()
