"""Offline full-model long QSA validation; does not certify quantization quality.

Processes every input token through all 48 layers. No fabricated KV, compressed
keys or recurrent history. Milestone decodes branch from actual prefix snapshots.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from tools.model.flash_native import Model,prefill,state_checks,generate
from tools.model.flash_checkpoint import Checkpoint
from tools.model.flash_chunks import chunks
from tools.operators.common import configure,environment,write_json,SEED


def state_hash(model):
    # Bounded CPU staging even for six GiB of private cache state.
    result={}
    for name,state in model.live_states().items():
        h=hashlib.sha256()
        if state.ndim:
            stride=max(1,4*1024**2//max(1,state[0].numel()*state.element_size())) if len(state) else 1
            for first in range(0,len(state),stride):
                h.update(state[first:first+stride].contiguous().cpu().view(torch.uint8).numpy().tobytes())
        result[name]=h.hexdigest()
    return result


def short_quality(model, source, tokenizer, checkpoint, baseline_path, output):
    from tools.model.flash_scenes import scenes
    from tools.model.flash_validation import baseline_probes,probe
    from tools.eval.scoring_common import pair_probes,validate_task
    import json
    _,cases=scenes(checkpoint)
    original=baseline_probes(json.loads(baseline_path.read_text()),source.config['quantization_config'],cases)
    baseline={(x['case_id'],x['position']):x for x in original}
    rows=[];probes=[]
    eos=json.loads((checkpoint/'generation_config.json').read_text())['eos_token_id']
    eos={eos} if isinstance(eos,int) else set(eos)
    for case in cases:
        model.reset();logits=prefill(model,case['prompt_ids'],8)
        saved=model.snapshot(cpu=True);prompt=logits.copy()
        generated,reason=generate(model,logits,eos,96)
        text=tokenizer.decode(generated,skip_special_tokens=True)
        rows.append({'id':case['id'],'text':text,'tokens':generated,'finish_reason':reason,
                     'task':validate_task(case['rule'],text)})
        model.restore(saved);logits=prompt
        for position,target in enumerate(case['target_ids']):
            base=baseline[case['id'],position]
            probes.append(probe(logits,case['prompt_ids'],case['target_ids'],position,case['id'],
                                [x['token_id'] for x in base['top3']]))
            if position+1<len(case['target_ids']):logits=model.execute([target])
        print('short quality',case['id'],repr(text),flush=True)
    paired=pair_probes([dict(baseline[x['case_id'],x['position']],execution_mode='flash-native') for x in probes],probes)
    report={'complete':True,'scope':'short scenes against original BF16; no long quality certification',
            'kv_dtype':'int8','requests':rows,'probes':probes,'paired_probes':paired}
    write_json(output/'short-quality.json',report)
    return {'path':'short-quality.json','probes':len(probes),
            'candidate_choice_in_original_top3':sum(x['candidate_selected_token_id'] in [t['token_id'] for t in baseline[p['case_id'],p['position']]['top3']] for x,p in zip(paired,probes))}


def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--context',type=int,default=262144);p.add_argument('--chunk',type=int,default=512)
    p.add_argument('--lengths',nargs='+',type=int,default=[2049,8192,32768,65536,131072,262136])
    p.add_argument('--decode',type=int,default=8)
    p.add_argument('--baseline',type=Path,help='Original short-scene BF16 probes, optional regression check')
    a=p.parse_args()
    if not 1<=a.chunk<=512 or a.decode<1 or not a.lengths or a.lengths!=sorted(set(a.lengths)):
        p.error('Sorted unique lengths, chunk 1..512 and positive decode required')
    if a.lengths[0]<1 or a.lengths[-1]+a.decode>a.context:p.error('Leave context room for decode')
    configure();source=Checkpoint(a.checkpoint);tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True)
    seed_text='春天的山谷里，河水缓缓流过草地。The river flows through a green valley. 记录编号42。\n'
    unit=tokenizer.encode(seed_text,add_special_tokens=False)
    tokens=(unit*((a.lengths[-1]+len(unit)-1)//len(unit)))[:a.lengths[-1]]
    write_json(a.output/'input.json',{'seed':SEED,'kind':'repeated bilingual synthetic capacity workload',
        'unit_ids':unit,'input_tokens':len(tokens),'quality_acceptance':False})
    started=time.perf_counter();model=Model(source,a.context,a.output,use_graph=True)
    torch.cuda.synchronize();load=time.perf_counter()-started
    report={'complete':False,'seed':SEED,'environment':environment(),'load_s':load,'capacity':a.context,
        'chunk':a.chunk,'quality_acceptance':False,'milestones':[]}
    warm=time.perf_counter();model.plan(1);model.plan(a.chunk)
    # Short fresh-request/graph and every-state restoration checks.
    report['short_state_checks']=state_checks(model,tokens[:min(a.chunk,8)])
    model.reset();torch.cuda.synchronize();report['warm_s']=time.perf_counter()-warm
    if a.baseline:report['short_quality']=short_quality(model,source,tokenizer,a.checkpoint,a.baseline,a.output)
    model.reset()
    consumed=0;total_s=0
    for target in a.lengths:
        begin=time.perf_counter()
        for batch in chunks(tokens[consumed:target],a.chunk):
            consumed+=len(batch)
            logits=model.execute(batch,output='logits' if consumed==target else 'none')
            if consumed%8192==0:
                print('prefill',consumed,'seconds',time.perf_counter()-begin,'allocated_GiB',torch.cuda.memory_allocated()/2**30,flush=True)
                write_json(a.output/'progress.json',{'position':consumed,'target':target,'elapsed_s':time.perf_counter()-begin})
        seconds=time.perf_counter()-begin;total_s+=seconds
        # Save only live prefix on CPU. It includes GDN/conv/PLE/index/pending.
        snapshot=model.snapshot(cpu=True);before=logits.copy();initial_hash=state_hash(model)
        decode_start=time.perf_counter();generated=[]
        for _ in range(a.decode):
            token=int(np.argmax(logits));generated.append(token);logits=model.execute([token])
        decode_s=time.perf_counter()-decode_start;after=state_hash(model);final=logits.copy()
        model.restore(snapshot);replay=before
        for token in generated:replay=model.execute([token])
        assert np.array_equal(replay,final),'Long prefix replay changed logits'
        assert state_hash(model)==after,'Long prefix replay changed private state'
        model.restore(snapshot);assert state_hash(model)==initial_hash
        del snapshot;gc.collect()
        row={'tokens':target,'prefill_segment_s':seconds,'prefill_total_s':total_s,'prefill_total_tps':target/total_s,
             'decode_s':decode_s,'decode_tps':a.decode/decode_s,'generated_ids':generated,
             'text':tokenizer.decode(generated),'position':model.position,'decode_end_position':target+a.decode,'prefix_restore_exact':True,
             'all_state_count':len(after),'state_sha256':after,'gpu_allocated_GiB':torch.cuda.memory_allocated()/2**30}
        report['milestones'].append(row);write_json(a.output/'results.json',report)
        print('milestone',target,'prefill_tps',row['prefill_total_tps'],'decode_tps',row['decode_tps'],flush=True)
    # Boundary rejection must happen before any state mutation.
    snapshot_position=model.position
    with_error=False
    try:model.execute([0]*(a.context-model.position+1))
    except ValueError:with_error=True
    assert with_error and model.position==snapshot_position
    report['capacity_overflow_rejected']=True;report['complete']=True;write_json(a.output/'results.json',report)


if __name__=='__main__':main()
