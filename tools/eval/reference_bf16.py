"""Independent Transformers BF16 teacher-forced prefill with streamed weights.

Only one text component's weights reside on CUDA at a time. Checkpoint tensors
retain their stored dtypes; no W4 reconstruction or candidate kernel is used.
This freezes a full-sequence reference, not a BF16 autoregressive timing test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
from tools.eval.scoring_common import SEED, context_hash, image_fingerprints


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--batch-size',type=int,default=1,help='Right-pad independent histories to stream weights once per batch')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    requests = json.loads(a.requests.read_text())
    if requests['seed'] != SEED or not requests['cases']:
        raise ValueError('Reference requires fixed seed and nonempty cases')
    for case in requests['cases']:
        context_hash(case['prompt_ids'], case.get('images', ()))
        image_fingerprints(case.get('images', ()))
        if not case['target_ids']:
            raise ValueError('Reference requires explicit teacher-forced target IDs')
    if not 1 <= a.batch_size <= 128:
        raise ValueError('batch-size must be in 1..128')
    torch.set_num_threads(2)
    torch.manual_seed(requests['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # Use the official Torch equations even when optional FLA/conv packages are
    # installed. Their binary kernels may not support this Orin's SM87 target.
    modeling.FusedRMSNormGated = None
    modeling.causal_conv1d_fn = None
    modeling.causal_conv1d_update = None
    modeling.chunk_gated_delta_rule = None
    modeling.fused_recurrent_gated_delta_rule = None
    config_raw = (a.checkpoint/'config.json').read_bytes()
    config_json = json.loads(config_raw)
    config = Qwen3_5TextConfig(**config_json.get('text_config',config_json))
    config._attn_implementation = 'sdpa'
    multimodal = any(c.get('images') for c in requests['cases'])
    with torch.device('meta'):
        if multimodal:
            full_config = Qwen3_5Config(**config_json)
            full_config.text_config._attn_implementation = 'sdpa'
            full_config.vision_config._attn_implementation = 'sdpa'
            model = modeling.Qwen3_5Model(full_config).to(dtype=torch.bfloat16).eval()
            text_model = model.language_model
        else:
            model = modeling.Qwen3_5TextModel(config).to(dtype=torch.bfloat16).eval()
            text_model = model
    text_model.rotary_emb = modeling.Qwen3_5TextRotaryEmbedding(config,device='cuda')
    index_raw = (a.checkpoint/'model.safetensors.index.json').read_bytes()
    index = json.loads(index_raw)['weight_map']
    sources = {filename:safe_open(a.checkpoint/filename,framework='pt',device='cpu')
               for filename in set(index.values())}

    def tensor(name):
        return sources[index[name]].get_tensor(name).to('cuda')

    def attach(module,prefix):
        def load(current,inputs):
            state = {name:tensor(prefix+name) for name in current.state_dict()}
            current.load_state_dict(state,strict=True,assign=True)
            if prefix == 'model.visual.':
                # The nonpersistent buffer starts on meta and is unloaded with
                # the vision weights. Recreate the official formula per call.
                rope = current.rotary_pos_emb
                rope.inv_freq = 1.0 / (rope.theta ** (
                    torch.arange(0, rope.dim, 2, device='cuda').float() / rope.dim))
        def unload(current,inputs,result):
            current.to_empty(device='meta')
        module.register_forward_pre_hook(load)
        module.register_forward_hook(unload)
    attach(text_model.embed_tokens,'model.language_model.embed_tokens.')
    for i,layer in enumerate(text_model.layers):
        attach(layer,f'model.language_model.layers.{i}.')
    attach(text_model.norm,'model.language_model.norm.')
    if multimodal:
        attach(model.visual,'model.visual.')
    head = torch.nn.Linear(config.hidden_size,config.vocab_size,bias=False,device='meta',dtype=torch.bfloat16)
    attach(head,'lm_head.')
    started = time.monotonic()
    report = dict(seed=requests['seed'],reference_kind='BF16 source checkpoint',revision=a.revision,
                  transformers=__import__('transformers').__version__,execution='teacher-forced full-sequence prefill',
                  weights='stored checkpoint dtypes; one component streamed onto CUDA',
                  config_sha256=hashlib.sha256(config_raw).hexdigest(),
                  index_sha256=hashlib.sha256(index_raw).hexdigest(),
                  requests_sha256=hashlib.sha256(a.requests.read_bytes()).hexdigest(),probes=[],complete=False)
    report['batch_size'] = a.batch_size
    report['padding'] = 'right padding masked; only real history positions scored'
    report['modalities'] = 'text, images and multiple images' if multimodal else 'text'
    report['multimodal_scope'] = 'Identical normalized FP32 patch pixels; official BF16 encoder, embedding injection and MRoPE; no video or candidate features'
    with torch.inference_mode():
        for start in range(0,len(requests['cases']),a.batch_size):
            cases = requests['cases'][start:start+a.batch_size]
            histories = [c['prompt_ids']+c['target_ids'][:-1] for c in cases]
            width = max(map(len,histories))
            input_ids = torch.zeros((len(cases),width),device='cuda',dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for row,history in enumerate(histories):
                input_ids[row,:len(history)] = torch.tensor(history,device='cuda')
                attention_mask[row,:len(history)] = 1
            kwargs = {}
            if multimodal:
                images = [image for case in cases for image in case.get('images',())]
                if images:
                    kwargs['pixel_values'] = torch.cat([
                        torch.tensor(image['pixels'],device='cuda',dtype=torch.bfloat16).reshape(
                            image['grid_height']*image['grid_width'],1536)
                        for image in images])
                    kwargs['image_grid_thw'] = torch.tensor([
                        [1,image['grid_height'],image['grid_width']] for image in images],device='cuda')
                # New Transformers versions require explicit modality IDs.
                # Delimiters remain text; only expanded image pads are visual.
                kwargs['mm_token_type_ids'] = (input_ids == full_config.image_token_id).int()
            hidden = model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,**kwargs).last_hidden_state
            selected = torch.cat([hidden[row,len(c['prompt_ids'])-1:len(c['prompt_ids'])-1+len(c['target_ids'])]
                                  for row,c in enumerate(cases)],dim=0)
            # Match the official BF16 head GEMM, then normalize in FP64 over the
            # complete vocabulary. Top-3 ties prefer the smaller token ID.
            logits = head(selected).double()
            logprobs = logits - torch.logsumexp(logits,dim=-1,keepdim=True)
            assert bool(torch.isfinite(logprobs).all()), 'Nonfinite reference probabilities'
            top_ids = torch.argsort(logits,dim=-1,descending=True,stable=True)[:,:3].cpu().tolist()
            probabilities = logprobs.cpu()
            offset = 0
            for case in cases:
                prompt,targets = case['prompt_ids'],case['target_ids']
                for position,target in enumerate(targets):
                    ids = top_ids[offset+position]
                    queries = set(ids+[target]+(case.get('query_ids') or [[]]*len(targets))[position])
                    report['probes'].append(dict(case_id=case['id'],execution_mode='prefill',position=position,
                        seed=requests['seed'],context_sha256=context_hash(prompt+targets[:position],case.get('images',())),
                        reference_token_id=target,reference_logprob=float(probabilities[offset+position,target]),
                        top3=[dict(token_id=i,logprob=float(probabilities[offset+position,i])) for i in ids],
                        queried_logprobs={str(i):float(probabilities[offset+position,i]) for i in queries}))
                offset += len(targets)
            report['elapsed_s'] = time.monotonic()-started
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print('scored',[c['id'] for c in cases],'elapsed_s',round(report['elapsed_s'],2),flush=True)
            del hidden,selected,logits,logprobs,probabilities
    report['complete'] = True
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
