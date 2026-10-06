"""Real Flash Next MTP equivalence, rollback, prefix restore and throughput."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from transformers import AutoTokenizer
from tools.model.flash_checkpoint import Checkpoint
from tools.model.flash_native import Model,prefill,generate
from tools.model.flash_mtp import Session
from tools.model.flash_policy import code_vocabulary
from tools.model.flash_long import state_hash,short_quality
from tools.model.flash_scenes import scenes
from tools.operators.common import configure,environment,write_json,identity,SEED


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--mtp-checkpoint',type=Path,required=True);p.add_argument('--compile-cache',nargs='+',type=Path)
    p.add_argument('--context',type=int,default=262144);p.add_argument('--decode',type=int,default=64)
    p.add_argument('--chunk',type=int,choices=(1,2,4,8,16,32,64,128,256,512),default=512)
    p.add_argument('--lengths',nargs='+',type=int,default=[512,2048,8192])
    p.add_argument('--drafts',nargs='+',type=int,default=[1,2]);p.add_argument('--baseline',type=Path)
    p.add_argument('--mtp-vocab-size',type=int,default=65536,help='Draft-only vocabulary; 0 uses the full head')
    p.add_argument('--benchmark-only',action='store_true',help='Repeat timing after a separate complete quality/state validation')
    a=p.parse_args()
    if a.decode<2 or not a.lengths or min(a.lengths)<1 or max(a.lengths)+a.decode>a.context or any(not 1<=d<=7 for d in a.drafts):
        p.error('Invalid workload or draft depth')
    if a.mtp_vocab_size and (not 256<=a.mtp_vocab_size<=Model.V or a.mtp_vocab_size%64):p.error('Invalid draft vocabulary size')
    if a.compile_cache:
        for cache in a.compile_cache:shutil.copytree(cache/'0.1.15',a.output/'cache'/'0.1.15',dirs_exist_ok=True)
    configure();np.random.seed(SEED)
    report={'complete':False,'seed':SEED,'environment':environment(),'context':a.context,
            'benchmark_only':a.benchmark_only,'chunk':a.chunk,'timing':'Decode graphs and each measured decode history are warmed before timing',
            'requests':[],'measurements':[],'state_checks':[],
            'checkpoint':identity(a.checkpoint/'model.safetensors.index.json'),
            'mtp_checkpoint':identity(a.mtp_checkpoint/'model.safetensors.index.json')}
    root=Path(__file__).resolve().parents[2]
    paths=[Path(__file__),root/'tools/model/flash_native.py',root/'tools/model/flash_mtp.py',
           root/'tools/model/flash_policy.py',root/'kernels/model/hyperconnection.py',
           root/'kernels/model/gdn_sequence.py',root/'kernels/model/rotation_a8.py']
    report['sources']=[identity(path) for path in paths]
    for path in paths:
        destination=a.output/'measurement-source'/path.relative_to(root)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    started=time.perf_counter();source=Checkpoint(a.checkpoint)
    model=Model(source,a.context,a.output/'target',use_graph=True)
    draft=Model(Checkpoint(a.mtp_checkpoint),a.context,a.output/'draft',use_graph=True,target=model)
    session=Session(model,draft);torch.cuda.synchronize()
    report['load_s']=time.perf_counter()-started;report['resident_gpu_bytes']=torch.cuda.memory_allocated()
    assert draft.weights['output.weight'] is model.weights['output.weight'] and draft.embedding is model.embedding
    tokenizer,cases=scenes(a.checkpoint)
    if a.mtp_vocab_size:draft.set_draft_vocab(code_vocabulary(tokenizer,Path(__file__).resolve().parents[2],a.mtp_vocab_size))
    eos=json.loads((a.checkpoint/'generation_config.json').read_text())['eos_token_id']
    eos={eos} if isinstance(eos,int) else set(eos)
    # Warm profiles and every output graph outside performance measurements.
    started=time.perf_counter()
    for size in sorted({1,8,a.chunk,*[d+1 for d in a.drafts]}):
        model.reset();model.execute(cases[0]['prompt_ids'][:1]*size,output='token')
        model.reset();model.execute(cases[0]['prompt_ids'][:1]*size,output='logits')
        draft.reset();draft.execute(cases[0]['prompt_ids'][:1]*size,hidden=model.last_plan['residual'],output='token')
        model.reset();model.execute(cases[0]['prompt_ids'][:1]*size,output='none')
        draft.reset();draft.execute(cases[0]['prompt_ids'][:1]*size,hidden=model.last_plan['residual'],output='none')
        if size<=8:
            model.reset();model.execute(cases[0]['prompt_ids'][:1]*size,output='verify');model.commit(size)
    session.prefill(cases[0]['prompt_ids'],a.chunk);session.generate(8,set(),drafts=max(a.drafts))
    report['warm_s']=time.perf_counter()-started
    print('loaded and warm',report['load_s'],report['warm_s'],flush=True)
    # Every rejected branch must restore all recurrent, ring and PLE state.
    for size in ([] if a.benchmark_only else sorted({d+1 for d in a.drafts})):
        for start in (5,6,7):
            model.reset();prefill(model,cases[0]['prompt_ids'][:start],a.chunk)
            prefix=model.snapshot(cpu=True);tokens=[int((11+i*71)%model.V) for i in range(size)]
            expected=model.execute(tokens,output='verify')
            try:model.snapshot(cpu=True)
            except ValueError:pass
            else:raise AssertionError('Uncommitted verification snapshot accepted')
            plan=model.last_plan
            hidden=plan['residual'].clone()
            saved={k:v.clone() for k,v in plan['prefix_states'].items()}
            for accepted in range(1,size+1):
                model.restore(prefix);actual=model.execute(tokens,output='verify')
                assert actual==expected
                model.commit(accepted)
                assert model.position==start+accepted
                assert all(torch.equal(model.states[k],v[accepted-1].reshape(model.states[k].shape)) for k,v in saved.items())
                assert model.history==model.ple.row_ids(tokens[:accepted],prefix['history'])[1]
                after=model.snapshot(cpu=True);prediction=model.execute([expected[accepted-1]],output='token');hashes=state_hash(model)
                model.reset();model.execute([37]);model.restore(after)
                assert model.execute([expected[accepted-1]],output='token')==prediction and state_hash(model)==hashes
                # Same accepted prefix with different rejected suffix: it must
                # produce exactly the same live state and final prefix output.
                model.restore(prefix)
                alternative=tokens[:accepted]+[int((t+19)%model.V) for t in tokens[accepted:]]
                changed=model.execute(alternative,output='verify');model.commit(accepted)
                assert changed[:accepted]==expected[:accepted]
                for name,value in after['states'].items():assert torch.equal(model.live_states()[name].cpu(),value),name
                del after
            report['state_checks'].append({'size':size,'start':start,'all_acceptance_lengths':True,
                                           'private_state_count':len(model.states),'suffix_isolation':True,'prefix_replay_exact':True})
            write_json(a.output/'results.json',report)
            del prefix,saved,hidden
    for case in ([] if a.benchmark_only else cases):
        model.reset();logits=prefill(model,case['prompt_ids'],a.chunk)
        reference,finish=generate(model,logits,eos,96)
        for depth in a.drafts:
            session.prefill(case['prompt_ids'],a.chunk);prefix=session.snapshot(cpu=True)
            generated,reason=session.generate(96,eos,drafts=depth)
            assert generated==reference and reason==finish,f'Greedy output changed: {case["id"]}, depth {depth}'
            final_target,final_draft=state_hash(model),state_hash(draft)
            session.restore(prefix);replay,replay_reason=session.generate(96,eos,drafts=depth)
            assert replay==generated and replay_reason==reason and state_hash(model)==final_target and state_hash(draft)==final_draft
            row={'id':case['id'],'drafts':depth,'tokens':generated,'text':tokenizer.decode(generated,skip_special_tokens=True),
                 'finish_reason':reason,'greedy_equal':True,'prefix_restore_exact':True,'statistics':dict(session.statistics)}
            report['requests'].append(row);write_json(a.output/'results.json',report)
            print('scene',case['id'],depth,row['statistics'],flush=True)
            if case['id']=='code' and depth==min(a.drafts):
                session.restore(prefix)
                rejected,why=session.generate(96,eos,drafts=depth,override=lambda round,index,value:model.V-1)
                assert rejected==reference and why==finish and session.statistics['accepted']==0
                report['forced_rejection']={'greedy_equal':True,'statistics':dict(session.statistics)}
                # Malformed restore must reject before either runtime changes.
                session.restore(prefix);before=(state_hash(model),state_hash(draft))
                broken=copy.copy(prefix);broken['draft']=dict(prefix['draft'],position=prefix['draft']['position']+1)
                try:session.restore(broken)
                except ValueError:pass
                else:raise AssertionError('Invalid cursor snapshot accepted')
                assert (state_hash(model),state_hash(draft))==before
                for value in (float('nan'),float('inf'),float('-inf')):
                    broken=dict(prefix,hidden=prefix['hidden'].clone().fill_(value))
                    try:session.restore(broken)
                    except ValueError:pass
                    else:raise AssertionError('Nonfinite draft condition accepted')
                    assert (state_hash(model),state_hash(draft))==before
                report['invalid_restore_atomic']=True
            del prefix
    # Paired throughput includes draft + verification + accepted state commit
    # + refresh, but excludes compilation and initial prefill/warm.
    rng=np.random.default_rng(SEED)
    text=''.join(f'Record {i}: city={rng.choice(["Paris","Tokyo","北京"])}; temperature={int(rng.integers(-20,40))}; id={int(rng.integers(100000,999999))}.\n' for i in range(max(a.lengths)//6+1))
    unit=tokenizer.encode(text,add_special_tokens=False)
    tokens=(unit*((max(a.lengths)+len(unit)-1)//len(unit)))[:max(a.lengths)]
    for length in a.lengths:
        model.reset();started=time.perf_counter();logits=prefill(model,tokens[:length],a.chunk)
        torch.cuda.synchronize();prefill_s=time.perf_counter()-started
        prefix=model.snapshot(cpu=True);prompt_logits=logits.copy()
        generate(model,prompt_logits,set(),a.decode)
        model.restore(prefix)
        started=time.perf_counter();reference,_=generate(model,logits,set(),a.decode)
        torch.cuda.synchronize();plain_s=time.perf_counter()-started
        report['measurements'].append({'length':length,'mtp':False,'decode_tokens':len(reference)-1,
                                      'decode_s':plain_s,'decode_tps':(len(reference)-1)/plain_s,
                                      'prefill_s':prefill_s,'prefill_tps':length/prefill_s})
        del prefix
        for depth in a.drafts:
            started=time.perf_counter();session.prefill(tokens[:length],a.chunk)
            torch.cuda.synchronize();prefill_s=time.perf_counter()-started
            prefix=session.snapshot(cpu=True)
            session.generate(a.decode,set(),drafts=depth)
            session.restore(prefix)
            started=time.perf_counter();generated,_=session.generate(a.decode,set(),drafts=depth)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            assert generated==reference,f'Throughput workload tokens changed: {length}, {depth}'
            row={'length':length,'mtp':True,'drafts':depth,'decode_tokens':len(generated)-1,
                 'decode_s':elapsed,'decode_tps':(len(generated)-1)/elapsed,
                 'prefill_s':prefill_s,'prefill_tps':length/prefill_s,'statistics':dict(session.statistics),'greedy_equal':True}
            report['measurements'].append(row);write_json(a.output/'results.json',report)
            print('measured',row,flush=True)
            del prefix
    if a.baseline:report['original_quality']=short_quality(model,source,tokenizer,a.checkpoint,a.baseline,a.output)
    report['sources_unchanged']=all(identity(Path(x['path']))['sha256']==x['sha256'] for x in report['sources'])
    assert report['sources_unchanged'],'Implementation changed during validation'
    report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
