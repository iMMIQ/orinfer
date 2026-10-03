"""Compare OCR with thinking off/on through the real Chat API."""
import argparse
import base64
import json
import os
from pathlib import Path
import time
import urllib.request

PROMPT = 'Read the large text in this image. Output only the text.'
SEED = 20261002


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8088/v1')
    parser.add_argument('--model', default='qwen3.8-27b')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--reasoning-effort', choices=('low', 'medium', 'xhigh'), default='xhigh')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    mime = {'.png':'image/png', '.jpg':'image/jpeg', '.jpeg':'image/jpeg', '.webp':'image/webp'}[args.image.suffix.lower()]
    image = {'type':'image_url', 'image_url':{'url':f'data:{mime};base64,'+base64.b64encode(args.image.read_bytes()).decode()}}
    headers = {'Content-Type':'application/json'}
    if os.getenv('ORIN_API_KEY'):
        headers['Authorization'] = 'Bearer '+os.environ['ORIN_API_KEY']
    report = {'image':str(args.image), 'prompt':PROMPT, 'seed':SEED, 'temperature':0, 'max_tokens':args.max_tokens, 'checks':[]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for thinking in (False, True):
        body = {'model':args.model, 'messages':[{'role':'user','content':[{'type':'text','text':PROMPT},image]}],
                'temperature':0, 'seed':SEED, 'max_tokens':args.max_tokens, 'enable_thinking':thinking,
                'reasoning_effort':args.reasoning_effort, 'stream':True, 'stream_options':{'include_usage':True}}
        start = time.monotonic()
        record = {'enable_thinking':thinking, 'reasoning_effort':args.reasoning_effort, 'content':'', 'reasoning_content':'', 'finish_reason':None, 'usage':None, 'first_delta_s':None, 'first_content_s':None}
        report['checks'].append(record)
        request = urllib.request.Request(args.base_url+'/chat/completions',data=json.dumps(body).encode(),headers=headers)
        print(f'Thinking {thinking}, effort {args.reasoning_effort}', flush=True)
        done = False
        with urllib.request.urlopen(request,timeout=300) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].decode().strip()
                if payload == '[DONE]':
                    done = True
                    break
                chunk = json.loads(payload)
                if chunk.get('error'):
                    raise RuntimeError(chunk['error'])
                if chunk.get('usage'):
                    record['usage'] = chunk['usage']
                for choice in chunk.get('choices', []):
                    delta = choice['delta']
                    for field in ('content', 'reasoning_content'):
                        text = delta.get(field, '')
                        if text:
                            if record['first_delta_s'] is None:
                                record['first_delta_s'] = time.monotonic()-start
                            if field == 'content' and record['first_content_s'] is None:
                                record['first_content_s'] = time.monotonic()-start
                            record[field] += text
                    if choice.get('finish_reason'):
                        record['finish_reason'] = choice['finish_reason']
                args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        assert done and record['usage'] is not None
        record['elapsed_s'] = time.monotonic()-start
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
