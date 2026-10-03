"""Generate offline reference patches with the checkpoint's image processor."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    path=a.model / 'cache/model.json' if a.model.is_dir() else a.model
    data=json.loads(path.read_text());spec=data.get('metadata',data)['vision']
    processor=Qwen2VLImageProcessor(min_pixels=65536,max_pixels=min(16777216,spec['max_patches']*256),patch_size=16,temporal_patch_size=2,merge_size=2,image_mean=[.5]*3,image_std=[.5]*3)
    cases=[]
    for i,(h,w,fmt) in enumerate([(300,450,'PNG'),(99,133,'JPEG'),(256,256,'WEBP')]):
        y,x=np.mgrid[:h,:w]
        rgb=np.stack((x*255//max(1,w-1),y*255//max(1,h-1),((x//16+y//16)%2)*255),-1).astype(np.uint8)
        path=a.output/f'case{i}.{fmt.lower()}';Image.fromarray(rgb).save(path)
        with Image.open(path) as image:ref=processor.preprocess(image.convert('RGB'),return_tensors='np')
        pixels=ref['pixel_values'].astype('<f4');binary=a.output/f'case{i}.f32';pixels.tofile(binary)
        cases.append({'image':path.name,'pixels':binary.name,'grid':ref['image_grid_thw'][0].tolist()})
    (a.output/'reference.json').write_text(json.dumps({'vision':spec,'cases':cases},indent=2)+'\n')

if __name__=='__main__':main()
