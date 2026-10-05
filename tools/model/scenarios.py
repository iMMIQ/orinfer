"""Token-ID request preparation and task checks for the native model executable.

512-token system padding is explicit; this is a task smoke test, not a paired
BF16/FP8 quality benchmark. Never execute generated code.
"""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
TOKENIZER=None


def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True)


def prepare(directory):
    directory.mkdir(parents=True,exist_ok=False)
    fixtures=json.loads((ROOT/'fixtures/quick-quality-scenarios.json').read_text())
    tok=tokenizer();cases=[];requests=[]
    left=tok.encode('<|im_start|>system\n下面的干扰记录与用户问题无关。只回答用户的问题。\n',add_special_tokens=False)
    right=tok.encode('\n<|im_end|>\n',add_special_tokens=False)
    filler=tok.encode('干扰记录：设备编号000，温度25度。\n',add_special_tokens=False)
    for case in fixtures['cases']:
        prompt=tok.apply_chat_template(case['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
        if not isinstance(prompt,list) or not all(type(i) is int for i in prompt):
            raise TypeError('Tokenizer must return a flat list of integer token IDs')
        count=512-len(left)-len(right)-len(prompt)
        if count<0:raise ValueError('Scene exceeds the current 512-token smoke shape')
        ids=left+(filler*((count+len(filler)-1)//len(filler)))[:count]+right+prompt
        assert len(ids)==512
        requests.append({'id':case['id'],'input_tokens':ids,'max_new_tokens':48})
        cases.append({'id':case['id'],'messages':case['messages'],'validation':case['validation'],'original_prompt_tokens':len(prompt),'system_padding_tokens':512-len(prompt),'rendered_input':tok.decode(ids,skip_special_tokens=False)})
    identity={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [TOKENIZER/'tokenizer.json',TOKENIZER/'tokenizer_config.json',TOKENIZER/'chat_template.jinja']}
    (directory/'requests.json').write_text(json.dumps({'requests':requests},indent=2)+'\n')
    (directory/'scenes.json').write_text(json.dumps({'seed':20261002,'generation':fixtures['generation'],'tokenizer_files_sha256':identity,'scope':'Actual 512-token padded chat scenes, greedy fixed 48 outputs; score stops at first EOS; no BF16/FP8 reference','cases':cases},indent=2,ensure_ascii=False)+'\n')


def score(directory,report_path,output):
    from tools.eval.scoring_common import validate_task
    scenes=json.loads((directory/'scenes.json').read_text());requests=json.loads((directory/'requests.json').read_text());report=json.loads(report_path.read_text())
    tok=tokenizer();rows=[]
    eos=tok.eos_token_id
    stops={eos} if isinstance(eos,int) else set(eos)
    for case in scenes['cases']:
        result=next(r for r in report['requests'] if r['id']==case['id'])
        request=next(r for r in requests['requests'] if r['id']==case['id'])
        ids=result['output_tokens'];stop=next((i for i,t in enumerate(ids) if t in stops),len(ids))
        text=tok.decode(ids[:stop],skip_special_tokens=True)
        rows.append({'id':case['id'],'input_tokens':len(request['input_tokens']),'text':text,'visible_output_token_ids':ids[:stop],'finish_reason':'eos' if stop<len(ids) else 'length','task_check':validate_task(case['validation'],text)})
    summary={'seed':scenes['seed'],'scope':scenes['scope'],'manifest_sha256':report['manifest_sha256'],'requests_sha256':hashlib.sha256((directory/'requests.json').read_bytes()).hexdigest(),'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'counts':{status:sum(r['task_check']['status']==status for r in rows) for status in ['pass','fail','manual_review']},'cases':rows,'limitation':'Absolute scenario checks only, not a BF16/FP8 quality comparison; manual cases and truncation are not counted as passing'}
    with output.open('x') as f:json.dump(summary,f,indent=2,ensure_ascii=False)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);sub=ap.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('directory',type=Path)
    scoring=sub.add_parser('score');scoring.add_argument('directory',type=Path);scoring.add_argument('report',type=Path);scoring.add_argument('output',type=Path)
    a=ap.parse_args()
    global TOKENIZER
    TOKENIZER=a.checkpoint
    if a.command=='prepare':prepare(a.directory)
    else:score(a.directory,a.report,a.output)


if __name__=='__main__':main()
