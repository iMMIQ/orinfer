"""Paired full-model MTP optimization sweep with frozen baseline and state checks.

Source-based draft vocabularies exclude measured continuations. Diagnostic
phase events run separately from uninstrumented throughput trials.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import shutil
import time
import torch
from tools.model.flash_checkpoint import Checkpoint
from tools.model.flash_native import Model
from tools.model.flash_mtp import Session
from tools.model.flash_policy import AdaptiveDepth,code_vocabulary
from tools.model.flash_profile import PhaseTrace
from tools.model.flash_long import state_hash
from tools.model.tune_flash_mtp import requests,session_state
from tools.operators.common import configure,environment,identity,write_json,SEED


VARIANTS=('baseline','control','vocab16k','vocab32k','vocab64k','hc','gdn','rotation','direct','combined','adaptive','selected','selected-adaptive')


def frozen(root,name):
    spec=importlib.util.spec_from_file_location('frozen_'+name,root/'tools/model'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--mtp-checkpoint',type=Path,required=True);p.add_argument('--baseline-source',type=Path,required=True)
    p.add_argument('--compile-cache',nargs='+',type=Path,default=[])
    p.add_argument('--context',type=int,default=262144);p.add_argument('--lengths',nargs='+',type=int,default=[2048])
    p.add_argument('--decode',type=int,default=384);p.add_argument('--trials',type=int,default=2)
    p.add_argument('--cases',nargs='+',choices=('python','rust','typescript'),default=['python','rust','typescript'])
    p.add_argument('--thinking',choices=('off','on','both'),default='both')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--combined-vocab',type=int,default=32768)
    p.add_argument('--profile-tokens',type=int,default=64)
    p.add_argument('--prefill-chunk',type=int,choices=(0,128,256,512),default=0,
                   help='Compare warmed prefill with the original 128-token chunks; 0 reuses baseline prefixes')
    a=p.parse_args()
    if min(a.lengths)<256 or max(a.lengths)+a.decode+16>a.context or a.decode<32 or a.trials<1:
        p.error('Invalid request workload')
    for cache in a.compile_cache:shutil.copytree(cache/'0.1.15',a.output/'cache/0.1.15',dirs_exist_ok=True)
    configure();root=Path(__file__).resolve().parents[2]
    paths=[Path(__file__),root/'tools/model/flash_native.py',root/'tools/model/flash_mtp.py',
           root/'tools/model/flash_policy.py',root/'tools/model/flash_profile.py',root/'tools/model/tune_flash_mtp.py',
           root/'kernels/model/hyperconnection.py',root/'kernels/model/gdn_sequence.py',root/'kernels/model/integer_vq.py',
           root/'kernels/model/rotation_a8.py']
    report={'complete':False,'seed':SEED,'environment':environment(),'context':a.context,'measurements':[],
            'profiles':[],'prefill':[],'prefill_chunk':a.prefill_chunk,'sources':[identity(path) for path in paths],
            'baseline_sources':[identity(a.baseline_source/'tools/model'/f'{name}.py') for name in ('flash_native','flash_mtp')],
            'quality_scope':'Exact optimization regression against the same quantized target, not a new BF16 quality assessment',
            'timing':'Warmed complete decode; diagnostic events, loading, compilation and prefill excluded',
            'checkpoint':identity(a.checkpoint/'model.safetensors.index.json'),
            'mtp_checkpoint':identity(a.mtp_checkpoint/'model.safetensors.index.json')}
    for path in paths:
        dest=a.output/'measurement-source'/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
    tokenizer,cases=requests(a.checkpoint,a.lengths,a.cases,
                            [False,True] if a.thinking=='both' else [a.thinking=='on'])
    report['cases']=cases
    eos=json.loads((a.checkpoint/'generation_config.json').read_text())['eos_token_id'];eos={eos} if isinstance(eos,int) else set(eos)
    started=time.perf_counter();model=Model(Checkpoint(a.checkpoint),a.context,a.output/'target')
    draft=Model(Checkpoint(a.mtp_checkpoint),a.context,a.output/'draft',target=model)
    torch.cuda.synchronize();report['load_s']=time.perf_counter()-started
    legacy=frozen(a.baseline_source,'flash_native');controller=frozen(a.baseline_source,'flash_mtp')
    old_target=legacy.Model.__new__(legacy.Model);old_target.__dict__=model.__dict__.copy()
    old_draft=legacy.Model.__new__(legacy.Model);old_draft.__dict__=draft.__dict__.copy()
    for name,obj in (('target',old_target),('draft',old_draft)):
        obj.plans={};obj.kernels={};obj.output=a.output/('baseline-'+name)
    baseline=controller.Session(old_target,old_draft)
    prefixes={};references={};target_states={};prefill_states={}
    for case in cases:
        baseline.prefill(case['prompt_ids'],128)
        if a.prefill_chunk:
            torch.cuda.synchronize();started=time.perf_counter()
            baseline.prefill(case['prompt_ids'],128);torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            report['prefill'].append({'variant':'baseline','case_id':case['id'],'chunk':128,
                                     'prefill_s':elapsed,'prefill_tps':len(case['prompt_ids'])/elapsed})
            signature=session_state(baseline);prefill_states[case['id']]=signature[:7]+signature[8:]
        prefixes[case['id']]=baseline.snapshot(cpu=True)
        depth=3 if case['thinking'] else 7
        baseline.generate(a.decode,eos,drafts=depth)
        baseline.restore(prefixes[case['id']]);started=time.perf_counter()
        output,reason=baseline.generate(a.decode,eos,drafts=depth);torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        references[case['id']]=(output,reason);target_states[case['id']]=state_hash(old_target)
        report['measurements'].append({'variant':'baseline','case_id':case['id'],'trial':0,
                                      'decode_s':elapsed,'decode_tps':(len(output)-1)/elapsed,
                                      'output_tokens':len(output),'greedy_equal':True,'drafts':depth,
                                      'statistics':dict(baseline.statistics),'allocated_gpu_bytes':torch.cuda.memory_allocated()})
        write_json(a.output/'results.json',report);print('baseline',report['measurements'][-1],flush=True)
    report['references']={key:{'tokens':value[0],'finish_reason':value[1],
                              'text':tokenizer.decode(value[0],skip_special_tokens=True)} for key,value in references.items()}
    old_target.plans.clear();old_draft.plans.clear();old_target.last_plan=None;old_target.transaction=None
    old_draft.last_plan=None;baseline.hidden=None;gc.collect()
    session=Session(model,draft)
    for variant in a.variants:
        if variant=='baseline':continue
        for obj in (model,draft):
            obj.transaction=None;obj.last_plan=None;obj.plans.clear()
            obj.hc_fused=variant in ('hc','combined','adaptive','selected','selected-adaptive')
            obj.compact_gdn=variant in ('gdn','combined','adaptive','selected','selected-adaptive')
            obj.direct_experts=variant in ('direct','combined','adaptive')
            obj.fused_rotation=variant in ('rotation','combined','adaptive','selected','selected-adaptive')
        session.hidden=None;gc.collect()
        draft.draft_vocab=None;draft.weights.pop('draft_output.weight',None)
        size={'vocab16k':16384,'vocab32k':32768,'vocab64k':65536}.get(variant,
                    65536 if variant in ('selected','selected-adaptive') else
                    a.combined_vocab if variant in ('combined','adaptive') else 0)
        if size:draft.set_draft_vocab(code_vocabulary(tokenizer,root,size))
        if size:write_json(a.output/f'{variant}-vocabulary.json',{'ids':draft.draft_vocab,'source':'project code and authored prose, no measured outputs'})
        for case in cases:
            depth=3 if case['thinking'] else 7
            # Fixed candidates and tail plans; no capture in throughput timing.
            adaptive=variant in ('adaptive','selected-adaptive')
            depths=(1,3,7) if adaptive else (depth,)
            session.warm(case['prompt_ids'][0],depths)
            # Reuse the identical real prefill, refreshing only the current
            # draft head. Prefix restoration remains outside throughput timing.
            if a.prefill_chunk:
                session.prefill(case['prompt_ids'],a.prefill_chunk)
                torch.cuda.synchronize();started=time.perf_counter()
                session.prefill(case['prompt_ids'],a.prefill_chunk);torch.cuda.synchronize()
                elapsed=time.perf_counter()-started
                signature=session_state(session)
                exact=signature[:7]+signature[8:]==prefill_states[case['id']]
                report['prefill'].append({'variant':variant,'case_id':case['id'],'chunk':a.prefill_chunk,
                                         'prefill_s':elapsed,'prefill_tps':len(case['prompt_ids'])/elapsed,
                                         'target_and_draft_state_equal':exact})
                write_json(a.output/'results.json',report);print('prefill',report['prefill'][-1],flush=True)
                assert exact,'Chunking changed target or draft state'
            else:session.restore(prefixes[case['id']])
            saved=session.snapshot(cpu=True)
            def run(trace=None):
                return session.generate(a.decode if trace is None else min(a.profile_tokens,a.decode),eos,drafts=depth,
                    policy=AdaptiveDepth(initial=depth) if adaptive else None,trace=trace)
            run();torch.cuda.synchronize();expected=session_state(session)
            for trial in range(a.trials):
                session.restore(saved);torch.cuda.synchronize();started=time.perf_counter()
                output,reason=run();torch.cuda.synchronize();elapsed=time.perf_counter()-started
                equal=(output,reason)==references[case['id']]
                replay=session_state(session)==expected;target_equal=state_hash(model)==target_states[case['id']]
                row={'variant':variant,'case_id':case['id'],'trial':trial,'decode_s':elapsed,
                     'decode_tps':(len(output)-1)/elapsed,'output_tokens':len(output),'drafts':depth,
                     'greedy_equal':equal,'private_state_replay_exact':replay,'target_state_equal':target_equal,
                     'statistics':dict(session.statistics),'allocated_gpu_bytes':torch.cuda.memory_allocated()}
                report['measurements'].append(row);write_json(a.output/'results.json',report);print('measured',row,flush=True)
                assert equal,'Greedy output changed'
                assert replay,'Session replay changed'
                assert target_equal,'Target state changed'
            if a.profile_tokens:
                session.restore(saved);trace=PhaseTrace();run(trace)
                report['profiles'].append({'variant':variant,'case_id':case['id'],'phases':trace.summary()})
                write_json(a.output/'results.json',report)
            del saved
    report['sources_unchanged']=all(identity(Path(x['path']))['sha256']==x['sha256'] for x in report['sources']+report['baseline_sources'])
    assert report['sources_unchanged'],'Implementation changed during measurement'
    report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
