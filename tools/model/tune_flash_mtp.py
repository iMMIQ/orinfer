"""Tune complete greedy Flash Next MTP decode on authored code requests.

Reports and generated code stay in the requested artifact directory. Every
candidate uses the same full private prefix and is compared with MTP off.
This is a workload-specific offline benchmark, not a code-quality benchmark.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

import torch
from transformers import AutoTokenizer

from tools.model.flash_checkpoint import Checkpoint
from tools.model.flash_mtp import Session
from tools.model.flash_speculation import verification_size
from tools.model.flash_policy import code_vocabulary
from tools.model.flash_native import Model, generate
from tools.model.flash_long import state_hash
from tools.operators.common import configure, environment, identity, write_json, SEED


TASKS = {
    'python': ('Python', 'Implement a thread-safe TTL LRU cache in Python using only the standard library. '
               'Include a generic class, constructor validation, get, put, delete, clear, and len. '
               'Use time.monotonic, lazy expiry, OrderedDict and an RLock. An expired get must raise '
               'KeyError. Add unittest tests for eviction, expiry using an injected clock, update, '
               'and deletion. Output the complete source code, without prose.'),
    'rust': ('Rust', 'Write a complete Rust module implementing a streaming binary frame decoder. '
             'The format is a big-endian u32 payload length followed by payload bytes. '
             'Use only std, accept arbitrary chunks, retain incomplete frames, reject frames '
             'above a configurable maximum, and avoid copying the unconsumed buffer on every '
             'push. Include explicit error types, constructor, push, reset and unit tests '
             'covering fragmented headers, fragmented payloads, multiple frames and oversized '
             'frames. Output the complete Rust code without prose.'),
    'typescript': ('TypeScript', 'Implement a fully typed TypeScript async task pool without dependencies. '
                   'Provide mapConcurrent<T,R>(items, concurrency, fn, signal?) returning results '
                   'in input order. Validate concurrency, cap in-flight tasks, propagate the first '
                   'error, honor AbortSignal before starting a task, and settle already started '
                   'promises without unhandled rejections. Include usage examples and tests for '
                   'ordering, maximum concurrency, errors and cancellation. Output source code '
                   'only.'),
}


def requests(checkpoint, lengths, names, thinking, reasoning_effort='xhigh'):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    template = (checkpoint/'chat_template.jinja').read_text()
    # Deterministic repository context, rather than repeated numbered prose.
    context = ''.join(f'def normalize_record_{i}(record: dict) -> dict:\n'
                      f'    key = "field_{i}"\n    value = record.get(key, {i})\n'
                      '    return {key: str(value).strip(), "valid": value is not None}\n\n'
                      for i in range(max(lengths)//12+32))
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    result = []
    for name in names:
        language, task = TASKS[name]
        for enabled in thinking:
            for requested in lengths:
                def render(count):
                    content = ('Repository reference helpers (not part of the requested implementation):\n'
                               '```python\n'+tokenizer.decode(context_ids[:count])+'\n```\n\n'+task)
                    messages = [{'role':'system','content':f'You are a careful {language} programmer.'},
                                {'role':'user','content':content}]
                    ids = tokenizer.apply_chat_template(messages,chat_template=template,tokenize=True,return_dict=False,
                                                        add_generation_prompt=True,enable_thinking=enabled,
                                                        reasoning_effort=reasoning_effort)
                    return messages, ids
                messages, ids = render(0)
                count = max(0,requested-len(ids))
                # Retokenization at context boundaries can change a few tokens.
                for _ in range(4):
                    messages,ids = render(count)
                    if len(ids)==requested:break
                    count=max(0,min(len(context_ids),count+requested-len(ids)))
                result.append({'id':f'{name}-{"thinking" if enabled else "direct"}-{requested}',
                               'language':language,'thinking':enabled,'requested_length':requested,
                               'reasoning_effort':reasoning_effort if enabled else None,
                               'prompt_ids':ids,'messages':messages})
    return tokenizer,result


def summarize(rows):
    """Aggregate actual tokens/time within each paired workload trial."""
    configs=sorted({row['drafts'] for row in rows})
    summary=[]
    for depth in configs:
        selected=[r for r in rows if r['drafts']==depth]
        trials=sorted({r['trial'] for r in selected})
        rates=[sum(r['decode_tokens'] for r in selected if r['trial']==t)/
               sum(r['decode_s'] for r in selected if r['trial']==t) for t in trials]
        stats=[r['statistics'] for r in selected if r['statistics']]
        proposed=sum(s['proposed'] for s in stats);accepted=sum(s['accepted'] for s in stats)
        summary.append({'drafts':depth,'median_decode_tps':statistics.median(rates),
                        'trial_decode_tps':rates,'requests':len(selected),
                        'all_greedy_equal':all(r['greedy_equal'] for r in selected),
                        'acceptance':accepted/proposed if proposed else None})
    return sorted(summary,key=lambda r:r['median_decode_tps'],reverse=True)


def session_state(session):
    """Hash the live prefix, including CPU cursors and the next draft condition."""
    model,draft=session.target,session.draft
    return (state_hash(model),state_hash(draft),model.position,draft.position,
            list(model.history),list(draft.history),session.pending,session.draft_token,
            hashlib.sha256(session.hidden.cpu().view(torch.uint8).numpy().tobytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--mtp-checkpoint',type=Path,required=True)
    parser.add_argument('--compile-cache',nargs='+',type=Path,default=[])
    parser.add_argument('--context',type=int,default=262144)
    parser.add_argument('--chunk',type=int,choices=(1,2,4,8,16,32,64,128,256,512),default=512)
    parser.add_argument('--lengths',nargs='+',type=int,default=[512,2048,8192])
    parser.add_argument('--cases',nargs='+',choices=tuple(TASKS),default=list(TASKS))
    parser.add_argument('--thinking',choices=['off','on','both'],default='off')
    parser.add_argument('--reasoning-effort',choices=['low','medium','xhigh'],default='xhigh')
    parser.add_argument('--decode',type=int,default=256)
    parser.add_argument('--budgets',nargs='+',type=int,help='Sweep output budgets; overrides --decode')
    parser.add_argument('--drafts',nargs='+',type=int,default=[0,1,2,3,4,5,6,7],help='0 is MTP off')
    parser.add_argument('--mtp-vocab-size',type=int,default=65536,help='Draft-only vocabulary; 0 uses the full head')
    parser.add_argument('--trials',type=int,default=2)
    args=parser.parse_args()
    budgets=args.budgets or [args.decode]
    if (min(budgets)<2 or len(set(budgets))!=len(budgets) or args.trials<1 or min(args.lengths)<256 or
            max(args.lengths)+max(budgets)+16>args.context or
            any(not 0<=d<=7 for d in args.drafts) or len(set(args.drafts))!=len(args.drafts) or
            len(set(args.lengths))!=len(args.lengths) or len(set(args.cases))!=len(args.cases)):
        parser.error('Invalid tuning workload')
    if args.mtp_vocab_size and (not 256<=args.mtp_vocab_size<=Model.V or args.mtp_vocab_size%64):
        parser.error('Invalid draft vocabulary size')
    for cache in args.compile_cache:
        shutil.copytree(cache/'0.1.15',args.output/'cache'/'0.1.15',dirs_exist_ok=True)
    configure()
    tokenizer,cases=requests(args.checkpoint,args.lengths,args.cases,
                             [False,True] if args.thinking=='both' else [args.thinking=='on'],args.reasoning_effort)
    cases=[dict(case,id=case['id']+(f'-out{budget}' if len(budgets)>1 else ''),decode_budget=budget)
           for case in cases for budget in budgets]
    report={'complete':False,'seed':SEED,'environment':environment(),'context':args.context,
            'decode_budgets':budgets,'trials':args.trials,'chunk':args.chunk,'measurements':[],'mtp_vocab_size':args.mtp_vocab_size,
            'checkpoint':identity(args.checkpoint/'model.safetensors.index.json'),
            'mtp_checkpoint':identity(args.mtp_checkpoint/'model.safetensors.index.json'),
            'cases':cases,'timing':'Warmed full decode including drafting, verification, commit and refresh; '
            'prefill, compilation and prefix restore excluded; first prefill-selected token excluded',
            'quality_scope':'Greedy equivalence to our own target, not independent BF16 code quality'}
    source_files=('tools/model/tune_flash_mtp.py','tools/model/flash_native.py',
                  'tools/model/flash_mtp.py','tools/model/flash_speculation.py','tools/model/flash_long.py',
                  'tools/model/flash_policy.py','kernels/model/flash_mtp.py','kernels/model/greedy.py',
                  'kernels/model/hyperconnection.py','kernels/model/gdn_sequence.py','kernels/model/integer_vq.py',
                  'kernels/model/rotation_a8.py')
    report['sources']=[identity(Path(name)) for name in source_files]
    for name in source_files:
        destination=args.output/'measurement-source'/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,destination)
    started=time.perf_counter()
    model=Model(Checkpoint(args.checkpoint),args.context,args.output/'target',use_graph=True)
    draft=Model(Checkpoint(args.mtp_checkpoint),args.context,args.output/'draft',use_graph=True,target=model)
    if args.mtp_vocab_size:
        draft.set_draft_vocab(code_vocabulary(tokenizer,Path(__file__).resolve().parents[2],args.mtp_vocab_size))
        write_json(args.output/'draft-vocabulary.json',{'ids':draft.draft_vocab,'source':'project code and authored prose'})
    session=Session(model,draft)
    eos=json.loads((args.checkpoint/'generation_config.json').read_text())['eos_token_id']
    eos={eos} if isinstance(eos,int) else set(eos)
    torch.cuda.synchronize();report['load_s']=time.perf_counter()-started
    print('loaded',report['load_s'],flush=True)
    references={};prefixes={}
    # Freeze all shared prefixes once. CPU snapshots leave GPU memory to plans.
    for case in cases:
        session.prefill(case['prompt_ids'],args.chunk)
        prefixes[case['id']]=session.snapshot(cpu=True)
        logits=model.last_plan['output'][-1].cpu().numpy().copy()
        session.restore(prefixes[case['id']])
        reference,reason=generate(model,logits,eos,case['decode_budget'])
        if len(reference)<32:raise AssertionError('Code output too short for useful timing')
        references[case['id']]=(reference,reason,logits)
        print('prefix',case['id'],len(case['prompt_ids']),'outputs',len(reference),flush=True)
    report['references']={name:{'tokens':values[0],'finish_reason':values[1],
                              'text':tokenizer.decode(values[0],skip_special_tokens=True)}
                          for name,values in references.items()}
    write_json(args.output/'results.json',report)
    # Profiles are grouped by depth to keep the verification snapshots bounded.
    for depth in args.drafts:
        model.transaction=None;model.last_plan=None;draft.last_plan=None
        for key in list(model.plans):
            if isinstance(key,tuple):del model.plans[key]
        gc.collect()
        # Warm every possible tail verification and accepted-prefix refresh
        # before timing. Compilation is never hidden in a candidate's TPS.
        if depth:
            profiles=sorted({verification_size(depth,n,args.context) for n in range(1,depth+2)})
            for size in profiles:
                model.reset();model.execute([cases[0]['prompt_ids'][0]]*size,
                                             output='verify' if size>1 else 'token')
                if size>1:model.commit(size)
            hidden=model.last_plan['residual']
            for size in range(1,depth+2):
                draft.reset();draft.execute([cases[0]['prompt_ids'][0]]*size,
                                             hidden=hidden[:size],output='token')
        for case in cases:
            reference,reason,logits=references[case['id']]
            saved=prefixes[case['id']]
            def run():
                if depth:return session.generate(case['decode_budget'],eos,drafts=depth)
                return generate(model,logits,eos,case['decode_budget'])
            session.restore(saved);run();torch.cuda.synchronize()
            expected_states=session_state(session)
            for trial in range(args.trials):
                session.restore(saved);torch.cuda.synchronize()
                started=time.perf_counter();output,finish=run();torch.cuda.synchronize()
                elapsed=time.perf_counter()-started
                equal=output==reference and finish==reason
                replay_exact=session_state(session)==expected_states
                row={'case_id':case['id'],'prompt_tokens':len(case['prompt_ids']),
                     'decode_budget':case['decode_budget'],
                     'thinking':case['thinking'],'drafts':depth,'trial':trial,
                     'decode_tokens':len(output)-1,'decode_s':elapsed,
                     'decode_tps':(len(output)-1)/elapsed,'finish_reason':finish,
                     'greedy_equal':equal,'private_state_replay_exact':replay_exact,
                     'output_sha256':hashlib.sha256(
                         json.dumps(output).encode()).hexdigest(),
                     'statistics':dict(session.statistics) if depth else None,
                     'allocated_gpu_bytes':torch.cuda.memory_allocated()}
                report['measurements'].append(row)
                report['summary']=summarize(report['measurements'])
                write_json(args.output/'results.json',report)
                print('measured',row,flush=True)
                if not equal:raise AssertionError(f'Greedy output changed: {case["id"]}, depth {depth}')
                if not replay_exact:raise AssertionError(f'Private state replay changed: {case["id"]}, depth {depth}')
    report['complete']=True
    report['summary']=summarize(report['measurements'])
    write_json(args.output/'results.json',report)
    print('summary',report['summary'],flush=True)


if __name__=='__main__':main()
