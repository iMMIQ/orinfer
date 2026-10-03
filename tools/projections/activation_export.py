"""Opt-in real activation capture. Import install after other runner hooks.

Set ORIN_PROJECTION_OUTPUT to a mounted directory. Request extra_args
orin_export='capture-512' (no orin_timing) on a 512-token prompt, >=8 output tokens.
Only this diagnostic request incurs CPU copies; no random data is generated.
"""
import hashlib
import json
import os
from pathlib import Path
import torch

_CONTEXT = None


def install():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    execute_original = GPUModelRunner.execute_model
    forward_original = GPUModelRunner._model_forward
    determine_original = GPUModelRunner._determine_batch_execution_and_padding

    def save(kind, tensor):
        if _CONTEXT is None:
            return
        meta = dict(_CONTEXT)
        x = tensor.detach().reshape(-1, tensor.shape[-1]).cpu().contiguous()
        path = Path(os.environ['ORIN_PROJECTION_OUTPUT'])
        path.mkdir(parents=True, exist_ok=True)
        name = f"{meta['run_id']}-{meta['computed_tokens_before']}-{kind.replace('.', '_')}.pt"
        target = path / name
        torch.save(x, target)
        meta.update(kind=kind, shape=list(x.shape), dtype=str(x.dtype),
                    file=name, file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    tensor_sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    activation_origin='actual experimental-vLLM forward',
                    diagnostic_only=True)
        target.with_suffix('.json').write_text(json.dumps(meta, indent=2))

    def execute(self, scheduler_output, *args, **kwargs):
        global _CONTEXT
        new = {r.req_id: r for r in scheduler_output.scheduled_new_reqs}
        cached = scheduler_output.scheduled_cached_reqs
        computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
        _CONTEXT = None
        for rid, count in scheduler_output.num_scheduled_tokens.items():
            req = new.get(rid) or self.requests.get(rid)
            if req is None or req.sampling_params is None:
                continue
            tag = (req.sampling_params.extra_args or {}).get('orin_export')
            if tag:
                if len(scheduler_output.num_scheduled_tokens) != 1:
                    raise ValueError('Projection export requires one diagnostic request')
                if not str(tag).replace('-', '').replace('_', '').isalnum():
                    raise ValueError('Unsafe projection export tag')
                start = req.num_computed_tokens if rid in new else computed.get(rid, req.num_computed_tokens)
                if start > len(req.prompt_token_ids) + 8:
                    continue
                _CONTEXT = dict(run_id=str(tag), request_id=rid, scheduled_rows=count,
                                computed_tokens_before=start, seed=req.sampling_params.seed,
                                prompt_token_ids=list(req.prompt_token_ids),
                                mode='prefill' if start < len(req.prompt_token_ids) else 'decode',
                                source_config=os.environ.get('ORIN_PROJECTION_SOURCE', 'see experiment lock'))
        try:
            return execute_original(self, scheduler_output, *args, **kwargs)
        finally:
            _CONTEXT = None

    def forward(self, *args, **kwargs):
        if _CONTEXT is not None:
            for key in ('input_ids', 'positions'):
                value = kwargs.get(key)
                if value is not None:
                    _CONTEXT[key] = value.detach().cpu().tolist()
        if _CONTEXT is not None and not getattr(self, '_orin_projection_hooks', False):
            names = []
            for name, module in self.model.named_modules():
                if any(f'layers.{i}.' in name for i in (0, 32)) and name.endswith(
                        ('gate_up_proj', 'down_proj', 'in_proj_qkvz', 'in_proj_qkv', 'in_proj_z', 'out_proj')):
                    module.register_forward_pre_hook(lambda mod, a, n=name: save(n, a[0]))
                    names.append(name)
            original_logits = self.model.compute_logits
            def logits(hidden_states, *a, **kw):
                save('lm_head', hidden_states)
                return original_logits(hidden_states, *a, **kw)
            self.model.compute_logits = logits
            self._orin_projection_hooks = True
            print('ORIN_PROJECTION_CAPTURE registered', names, flush=True)
        return forward_original(self, *args, **kwargs)

    def determine(self, *args, **kwargs):
        if _CONTEXT is not None:
            kwargs['force_eager'] = True
        return determine_original(self, *args, **kwargs)

    GPUModelRunner._determine_batch_execution_and_padding = determine
    GPUModelRunner.execute_model = execute
    GPUModelRunner._model_forward = forward
