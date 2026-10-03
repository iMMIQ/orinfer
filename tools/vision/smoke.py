"""Real image/multi-image Chat API checks; fixture and result files stay offline."""
import argparse
import base64
import io
import json
import os
import time
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://127.0.0.1:8088/v1');p.add_argument('--model',default='qwen3.8-27b');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def image(color):
        im=Image.new('RGB',(256,256),'white');ImageDraw.Draw(im).rectangle((48,48,208,208),fill=color)
        buf=io.BytesIO();im.save(buf,format='PNG')
        return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}}
    red=image('red');blue=image('blue')
    text=lambda s:{'type':'text','text':s}
    cases=[
      ('red',[text('What color is the square? Answer with one English word.'),red],'red',False),
      ('blue',[blue,text('What color is the square? Answer with one English word.')],'blue',True),
      ('multi',[red,blue,text('Give the colors of the squares in the FIRST and SECOND image, in order. Only two English color words.')],['red','blue'],False),
      ('multi-reverse',[blue,text('First image above, second image below.'),red,text('Give the colors of the squares in image order. Only two English color words.')],['blue','red'],True),
      ('text-after-images','Reply with exactly the word OK.','ok',False),
      ('red-restored',[red,text('What color is the square? Answer with one English word.')],'red',False),
    ]
    results=[]
    headers={'Content-Type':'application/json'}
    if os.getenv('ORIN_API_KEY'):headers['Authorization']='Bearer '+os.environ['ORIN_API_KEY']
    for name,content,expected,stream in cases:
        body={'model':a.model,'messages':[{'role':'user','content':content}],'temperature':0,'seed':20261002,'max_tokens':32,'stream':stream}
        if stream:body['stream_options']={'include_usage':True}
        start=time.monotonic();req=urllib.request.Request(a.base_url+'/chat/completions',data=json.dumps(body).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=300) as response:
            if stream:
                chunks=[];done=False;answer='';usage=None
                for line in response:
                    if not line.startswith(b'data: '):continue
                    data=line[6:].decode().strip()
                    if data=='[DONE]':done=True;break
                    item=json.loads(data);chunks.append(item)
                    if item.get('error'):raise AssertionError(item)
                    if item.get('usage'):usage=item['usage']
                    if item.get('choices'):answer+=item['choices'][0]['delta'].get('content','')
                assert done and usage,(name,chunks)
                result={'chunks':chunks,'usage':usage}
            else:
                result=json.load(response);answer=result['choices'][0]['message']['content']
        lower=answer.lower()
        passed=expected in lower if isinstance(expected,str) else all(e in lower for e in expected) and lower.index(expected[0])<lower.index(expected[1])
        record={'case':name,'answer':answer,'elapsed_s':time.monotonic()-start,'passed':passed,'response':result}
        results.append(record);print(name,repr(answer),round(record['elapsed_s'],3),passed,flush=True)
        a.output.write_text(json.dumps({'status':'running','checks':results},ensure_ascii=False,indent=2)+'\n')
        assert passed,record
    a.output.write_text(json.dumps({'status':'passed','checks':results},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
