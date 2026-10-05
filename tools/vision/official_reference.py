"""Run the actual Transformers vision model, independently of kernel equations."""
import argparse
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pixels', type=Path, required=True)
    parser.add_argument('--grid', type=int, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=('bf16', 'f16'), default='bf16')
    parser.add_argument('--trace', action='store_true', help='Save patch and block outputs for numerical diagnosis')
    args = parser.parse_args()
    if (args.output/'result.json').exists() or (args.output/'features.f16').exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261002)
    torch.set_num_threads(6 if args.device == 'cpu' else 2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    config = Qwen3_5VisionConfig(**json.loads((args.checkpoint/'config.json').read_text())['vision_config'])
    config._attn_implementation = 'sdpa'
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    with torch.device('meta'):
        model = Qwen3_5VisionModel(config).to(dtype)
    with safe_open(args.checkpoint/'model.safetensors', framework='pt', device='cpu') as source:
        weights = {k.removeprefix('model.visual.'): source.get_tensor(k).to(dtype)
                   for k in source.keys() if k.startswith('model.visual.')}
    model.load_state_dict(weights, strict=True, assign=True)
    # Recreate nonpersistent meta buffers through the official initializer.
    model.rotary_pos_emb = type(model.rotary_pos_emb)(config)
    model = model.to(args.device).eval()
    def snapshot(name, value):
        value.half().detach().cpu().contiguous().view(torch.uint8).numpy().tofile(args.output/(name+'.f16'))
    if args.trace:
        model.patch_embed.register_forward_hook(lambda module, inputs, value: snapshot('patch',value))
        for index, block in enumerate(model.blocks):
            block.register_forward_hook(lambda module, inputs, value, index=index: snapshot(f'block{index}',value))
        model.merger.norm.register_forward_hook(lambda module, inputs, value: snapshot('merger_norm',value))
    h, w = args.grid
    pixels = torch.from_file(str(args.pixels), size=h*w*1536, dtype=torch.float32).reshape(h*w,1536).to(device=args.device, dtype=dtype)
    grid = torch.tensor([[1,h,w]], device=args.device)
    start = time.monotonic()
    with torch.inference_mode():
        features = model(pixels, grid_thw=grid, return_dict=True).pooler_output
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    assert tuple(features.shape) == (h*w//config.spatial_merge_size**2, config.out_hidden_size)
    assert bool(torch.isfinite(features).all()), 'Nonfinite official features'
    features.half().cpu().contiguous().view(torch.uint8).numpy().tofile(args.output/'features.f16')
    result = dict(status='passed', implementation='Transformers Qwen3_5VisionModel',
                  transformers=__import__('transformers').__version__, dtype=args.dtype,
                  reduced_precision_gemm_reduction=False,
                  device=args.device, grid=[1,h,w], shape=list(features.shape),
                  elapsed_s=time.monotonic()-start, finite=bool(torch.isfinite(features).all()))
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
