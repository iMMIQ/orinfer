"""Add native Flash MTP kernels and state to a registered serving package.

Target data is hardlinked and verified; embedding/head are shared with the draft.
The online Rust recipe validates this offline binding export before publication.
"""
import argparse
import math
import json
from pathlib import Path
import shutil

import torch
from tools.model.flash_next.checkpoint import Checkpoint
from tools.model.flash_next.native import Model
from tools.model.flash_next.prepare import Publisher, PREFILL_PROFILES, CHUNK_TOKENS
from tools.model.publication import write_json, link_or_copy, seed_compile_cache
from tools.operators.common import configure
from kernels.model import flash_control as fc
from kernels.model.control import advance
from kernels.model.speculation import select_prefix, gather_target_hidden
from kernels.model.gdn_sequence import gdn_commit

class MtpPublisher(Publisher):
    def __init__(self,target,draft,base):
        self.__dict__.update(base.__dict__)
        self.target,self.draft=target,draft
        self.ring=torch.empty((CHUNK_TOKENS,10240),device='cuda',dtype=torch.float16)
        self.buffer(self.ring,'MtpHiddenRing','sequence')
        self.buffers['MtpHiddenRing']['shape']=[CHUNK_TOKENS,10240]
        self.vlogits=torch.empty((8,target.V),device='cuda',dtype=torch.float32)
        self.vpairs=torch.empty((8,2),device='cuda',dtype=torch.int32)
        self.buffer(self.vlogits,'VerificationLogits','sequence');self.buffers['VerificationLogits']['shape']=[8,target.V]
        self.buffer(self.vpairs,'VerificationPairs','workspace')
        self.accepted=torch.empty((1,),device='cuda',dtype=torch.int32)
        self.buffer(self.accepted,'AcceptedInputs','sequence')
        self.fixed('VerificationTokens','i32',[8]);self.fixed('VerificationStatus','i32',[8])
        self.fixed('MtpInput','i32',[512]);self.fixed('MtpToken','i32',[1]);self.fixed('MtpStatus','i32',[1])
        self.fixed('DraftPendingSaved','f16',[4,128])
        self.captures=[];self.warms=[];self.verifications=[]
    def emit(self,program,section,key,build,args,m):
        self.model.kernel(key,build)
        op=self.bind(program,section,key,args,m)
        self.groups.setdefault(program,{}).setdefault(section,[]).append(op)
        return op
    def split(self,program,section,plan,m,verify=False,draft=False):
        key=f'control-selections-{m if verify else 1}'
        op=self.emit(program,section,key,lambda:fc.selections(m if verify else 1),
                     (plan['token'],plan['token'],plan['token']),m)
        names={'Pair':'VerificationPairs' if verify else 'DraftSelected' if draft else 'Selected',
               'Tokens':'VerificationTokens' if verify else 'MtpToken' if draft else 'Token',
               'Status':'VerificationStatus' if verify else 'MtpStatus' if draft else 'Status'}
        self.kernels[-1]['args']=[dict(kind='buffer',name=names[a['value'].removesuffix('.data_ptr()')]) for a in self.exports[key][1]['ordered_arguments']]
        return op
    def native(self,model,m,verify=False):
        self.model=model;model.position=model.capacity-m
        plan=model.plan(m,verify=verify);model.position=0
        draft=model.is_mtp
        self.scores(plan,m,draft=draft,verify=verify)
        program=f'mtp_warm_m{m}' if draft else f'verify_m{m}'
        if verify:
            old_output,old_pairs=plan['output'],plan['token']
            plan['output']=self.vlogits[:m];plan['token']=self.vpairs[:m].reshape(-1)
            def replace(ops):
                return [(k,tuple(plan['output'] if a is old_output else plan['token'] if a is old_pairs else a for a in args)) for k,args in ops]
            plan['ops']=replace(plan['ops']);plan['greedy_ops']=replace(plan['greedy_ops'])
        for key,name,scope in [('embedding',f'{"Draft" if draft else ""}M{m}_Embedding','workspace'),
                               ('ple_embedding',f'M{m}_Ple','workspace'),
                               ('residual',f'{"Draft" if draft else "Verify"}M{m}_HC','workspace')]:
            if not draft or key!='ple_embedding':self.buffer(plan[key],name,scope)
        if draft:
            self.buffer(plan['condition'],f'DraftM{m}_Condition','sequence' if m==1 else 'workspace')
            if m==1:self.state.add('DraftM1_Condition')
            self.buffer(plan['output'],'DraftLogits','sequence');self.buffers['DraftLogits']['shape']=[1,model.V]
            self.buffer(plan['token'],'DraftSelected','workspace')
        for name,value in plan['prefix_states'].items():self.buffer(value,f'VerifyM{m}_Prefix_'+name.replace(':','_'))
        for name,values in plan['prefix_updates'].items():
            for suffix,value in zip(('K','G','Update'),values):self.buffer(value,f'VerifyM{m}_Saved_'+name.replace(':','_')+'_'+suffix)
        for (_,args),label in zip(plan['ops'],plan['labels']):
            if label=='gdn-history-copy':self.buffer(args[0],f'M{m}_HistoryOut')
            elif label=='gdn-update-save':self.buffer(args[0],f'VerifyM{m}_'+('Keys' if args[0].dtype==torch.float16 else 'Gates'))
        ops=[];section='begin'
        if draft:
            ops.append(self.emit(program,section,f'control-gather-{m}',lambda:gather_target_hidden(m,10240,CHUNK_TOKENS,ring=True),
                     (self.ring,model.position_gpu,plan['condition'].flatten(1)),m))
            ops.append(self.emit(program,section,f'control-length-{m}',lambda:fc.length(m),(model.lengths,),m))
        for (kernel,args),label in zip(plan['ops'][:plan['body_count']],plan['labels']):
            layers=[]
            for a in args:
                if isinstance(a,torch.Tensor):
                    name=self.storage.get(a.untyped_storage().data_ptr(),'')
                    prefix='DW_blk.' if draft else 'W_blk.'
                    if name.startswith(prefix) and not name.endswith('_table'):layers.append(int(name.split('.')[1]))
            if layers:section=f'layer{min(layers)}'
            op=self.bind(program,section,label,args,m);self.groups.setdefault(program,{}).setdefault(section,[]).append(op);ops.append(op)
        ops.append(self.emit(program,'end',f'advance-{m}',lambda:advance(m),(model.position_gpu,),m))
        head=f'mtp_head_m{m}' if draft else program
        hsection='body' if draft else 'head';head_ops=[]
        for (kernel,args),label in list(zip(plan['ops'][plan['body_count']:],plan['labels'][plan['body_count']:]))+[(op,f'greedy-partials-{m if verify else 1}-{model.V}' if i==0 else f'greedy-merge-{m if verify else 1}-{model.V}') for i,op in enumerate(plan['greedy_ops'])]:
            op=self.bind(head,hsection,label,args,m);self.groups.setdefault(head,{}).setdefault(hsection,[]).append(op);head_ops.append(op)
        head_ops.append(self.split(head,hsection,plan,m,verify=verify,draft=draft))
        if draft:
            head_ops.append(self.emit(head,hsection,f'control-last-{m}',lambda:fc.last_hidden(m,10240),
                        (plan['residual'].flatten(1),self.draft_one['condition'].flatten(1)),m))
            self.programs[head]=head_ops;self.programs[program]=ops
            self.warms.append(dict(tokens=m,program=program,head_program=head))
            if m==1:self.programs['mtp_draft']=ops[1:]+head_ops
        else:
            self.programs[program]=ops+head_ops
            restore=f'restore_m{m}';commit=[]
            for layer in range(48):
                sec=f'layer{layer}'
                if layer==1:
                    name='ple';value=plan['prefix_states'][name];state=model.states[name]
                    commit.append(self.emit(restore,sec,f'control-prefix-{m}-{state.numel()}',lambda:select_prefix(m,state.numel(),'float16'),(value.reshape(m,-1),self.accepted,state.reshape(-1)),m))
                if layer%4==3:
                    name=f'{layer}:pending';value=plan['prefix_states'][name];state=model.states[name]
                    commit.append(self.emit(restore,sec,f'control-prefix-{m}-{state.numel()}',lambda:select_prefix(m,state.numel(),'float16'),(value.reshape(m,-1),self.accepted,state.reshape(-1)),m))
                else:
                    name=f'{layer}:conv';value=plan['prefix_states'][name];state=model.states[name]
                    commit.append(self.emit(restore,sec,f'control-prefix-{m}-{state.numel()}',lambda:select_prefix(m,state.numel(),'float16'),(value.reshape(m,-1),self.accepted,state.reshape(-1)),m))
                    k,g,update=plan['prefix_updates'][f'{layer}:gdn']
                    commit.append(self.emit(restore,sec,f'gdn-commit-{m}',lambda:gdn_commit(m),(k,g,update,self.accepted,model.states[f'{layer}:gdn']),m))
            self.programs[restore]=commit
            capture=self.capture(model,plan,m)
            self.verifications.append(dict(tokens=m,program=program,restore_program=restore,capture_program=capture))
        return plan
    def capture(self,model,plan,m):
        self.model=model;program=f'mtp_capture_m{m}'
        self.programs[program]=[self.emit(program,'body',f'control-capture-{m}',lambda:fc.capture(m,10240,CHUNK_TOKENS),(plan['residual'].flatten(1),model.position_gpu,self.ring),m)]
        self.captures.append(dict(tokens=m,program=program));return program


def register(publish,model,prefix,context):
    for logical,value in model.weights.items():
        items=value.items() if isinstance(value,dict) else zip(('weight','scale'),value) if isinstance(value,tuple) else [('value',value)]
        for suffix,tensor in items:publish.buffer(tensor,f'{prefix}W_{logical}_{suffix}','weights')
    for logical,tensor in model.states.items():
        name='State_'+logical.replace(':','_');publish.buffer(tensor,name,'sequence');publish.state.add(name)
        if logical in model.prefix_divisors:
            publish.kv[name]=tensor.untyped_storage().nbytes()//context
            if model.prefix_divisors[logical]!=1:publish.divisors[name]=model.prefix_divisors[logical]
    publish.buffer(model.position_gpu,'MtpPosition' if model.is_mtp else 'Position','sequence')
    publish.buffer(model.lengths,'MtpLength' if model.is_mtp else 'Length','sequence')
    publish.buffer(model.position_out,'MtpPositionOut' if model.is_mtp else 'PositionOut','sequence')
    if model.is_mtp:publish.state.add('MtpPosition')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--mtp-checkpoint',type=Path,required=True)
    p.add_argument('--base-model',type=Path,required=True);p.add_argument('--model-output',type=Path,required=True)
    p.add_argument('--compile-cache',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();configure()
    if a.compile_cache:
        seed_compile_cache(a.compile_cache, a.output/'cache/0.1.15')
    def clone(source,destination):
        if Path(source).suffix in (".json", ".jinja", ".jsonc"):return shutil.copyfile(source,destination)
        return link_or_copy(source,destination)
    if not a.model_output.exists():shutil.copytree(a.base_model,a.model_output,copy_function=clone,ignore=shutil.ignore_patterns("packages"))
    source=Checkpoint(a.checkpoint,verify_hashes=False)
    target=Model(source,262144,a.output/'target',use_graph=False)
    draft=Model(Checkpoint(a.mtp_checkpoint,verify_hashes=False),262144,a.output/'draft',use_graph=False,target=target)
    # Register target operands first, preserving base immutable shard identities.
    publish=Publisher(target,a.model_output,reuse_data=True);register(publish,target,'',262144)
    for name,dtype,shape in [('Input','i32',[CHUNK_TOKENS]),('Token','i32',[1]),('Status','i32',[1]),('LastIndex','i32',[1])]:publish.fixed(name,dtype,shape)
    for m in (*reversed(PREFILL_PROFILES),1):publish.plan(m)
    extended=MtpPublisher(target,draft,publish)
    register(extended,draft,'D',262144)
    for m in (*range(1,9),16,128,512):
        if m==1:
            draft.position=draft.capacity-1;extended.draft_one=draft.plan(1);draft.position=0
        extended.native(draft,m)
    for m in range(2,9):extended.native(target,m,verify=True)
    for m in (1,*PREFILL_PROFILES):
        target.position=target.capacity-m;plan=target.plan(m);target.position=0
        extended.capture(target,plan,m)
    extended.programs['mtp_snapshot']=[dict(kind='copy',source='State_48_pending',destination='DraftPendingSaved',bytes=1024)]
    extended.programs['mtp_restore_draft']=[dict(kind='copy',source='DraftPendingSaved',destination='State_48_pending',bytes=1024)]
    extended.model=target
    extended.finish(source)
    build_path=extended.cache/'build.json'
    build=json.loads(build_path.read_text());metadata=build['metadata']
    metadata['mtp']=dict(position='MtpPosition',input='MtpInput',token='MtpToken',status='MtpStatus',verification_tokens='VerificationTokens',verification_status='VerificationStatus',
        draft_logits='DraftLogits',verification_logits='VerificationLogits',feature_index=None,accepted_inputs='AcceptedInputs',target_length='PositionOut',draft_program='mtp_draft',hidden_ring='MtpHiddenRing',
        commit_always=True,draft_snapshot_program='mtp_snapshot',draft_restore_program='mtp_restore_draft',default_verification_tokens=4,
        warm_plans=extended.warms,capture_plans=[p for p in extended.captures if p['tokens'] in (1,*PREFILL_PROFILES)],verification_plans=extended.verifications)
    metadata['weight_parameters']+=sum(math.prod(draft.source.shape(name)) for name in draft.source.parts)
    metadata['weight_scope']+='; native Flash MTP with shared embedding and full output head'
    write_json(build_path,build);print('FLASH MTP BUILD COMPLETE',flush=True)
if __name__=='__main__':main()
