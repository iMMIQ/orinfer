"""Causal GQA, Q/K norm+RoPE, final head, and changed-input Graph checks."""

from orinfer_build.common.paths import ROOT

import argparse
import json
from pathlib import Path
from kernels.cuda.model import reranking as r
from kernels.cuda.vision.encoder import attention
from orinfer_build.compiler.compile import primitive
from orinfer_build.models.nemotron_embed.check_kernels import check
from orinfer_build.tuning.row_tuning import launch
from orinfer_build.compiler.cuda.common import configure


def run(original):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb
    from orinfer_build.models.qwen_reranker.config import PREFIX, SUFFIX, INSTRUCTION

    configure()
    records = []
    for n in [1, 7, 33, 65, 129]:
        q = torch.randn(n, 2048, device="cuda", dtype=torch.float16) * 0.2
        k = torch.randn(n, 1024, device="cuda", dtype=torch.float16) * 0.2
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        length = torch.tensor([n], device="cuda", dtype=torch.int32)

        def reference():
            qs = q.view(n, 16, 128).transpose(0, 1)[None]
            ks, vs = [
                a.view(n, 8, 128).transpose(0, 1).repeat_interleave(2, 0)[None] for a in (k, v)
            ]
            return (
                F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)[0]
                .transpose(0, 1)
                .reshape(n, 2048)
            )

        kernel = primitive(attention, 2048, 16, kv_heads=8, causal=True)
        error = check(kernel, [q, k, v, length, out], out, reference, q.neg_)
        initial = out.clone()
        if n > 1:
            v[-1].add_(10)
            k[-1].neg_()
            launch(kernel, q, k, v, length, out)
            torch.testing.assert_close(out[:-1], initial[:-1], atol=0, rtol=0)
        records.append(
            dict(kind="causal_gqa", rows=n, max_abs=error, graph=True, future_isolated=True)
        )
    cfg = Qwen3Config.from_pretrained(original, local_files_only=True)
    rope = Qwen3RotaryEmbedding(cfg).cuda()
    for n in [7, 65]:
        x = torch.randn(n, 4096, device="cuda", dtype=torch.float16) * 0.2
        qnorm = torch.randn(128, device="cuda", dtype=torch.float16) * 0.1 + 1
        knorm = torch.randn_like(qnorm) * 0.1 + 1
        cos, sin = [a[0].contiguous() for a in rope(x, torch.arange(n, device="cuda")[None])]
        outputs = [
            torch.empty(n, w, device="cuda", dtype=torch.float16) for w in [2048, 1024, 1024]
        ]

        def reference_rope():
            q, k, v = x.split([2048, 1024, 1024], -1)
            normalized = []
            for a, heads, w in [(q, 16, qnorm), (k, 8, knorm)]:
                a = a.reshape(n, heads, 128)
                a = (
                    a.float() * torch.rsqrt(a.float().square().mean(-1, keepdim=True) + 1e-6)
                ).half() * w
                normalized.append(a.transpose(0, 1))
            q, k = apply_rotary_pos_emb(*normalized, cos, sin, unsqueeze_dim=0)
            return [q.transpose(0, 1).reshape(n, 2048), k.transpose(0, 1).reshape(n, 1024), v]

        ranges = torch.zeros(n, 2, device="cuda", dtype=torch.int32)
        for group in [1, 4, 16, 32]:
            kernel = primitive(r.qk_rope, 32768, dynamic_rows=True, head_group=group)
            launch(kernel, ranges, x, qnorm, knorm, cos, sin, *outputs, n)
            for a, b in zip(outputs, reference_rope()):
                torch.testing.assert_close(a, b, atol=0.002, rtol=0.002)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(kernel, ranges, x, qnorm, knorm, cos, sin, *outputs, n)
            x.neg_()
            graph.replay()
            for a, b in zip(outputs, reference_rope()):
                torch.testing.assert_close(a, b, atol=0.002, rtol=0.002)
            records.append(dict(kind="qk_rms_rope", rows=n, head_group=group, graph=True))
    x = torch.randn(33, 1024, device="cuda", dtype=torch.float16)
    w = torch.randn(2, 1024, device="cuda", dtype=torch.float16)
    out = torch.empty(2, device="cuda")
    kernel = primitive(r.classifier, 32768, dynamic_rows=True)
    error = check(
        kernel, [x, w, out, 33], out, lambda: x[-1].float() @ w.float().T, x.neg_, atol=0.0001
    )
    records.append(dict(kind="last_position_binary_head", max_abs=error, graph=True))
    tokenizer = AutoTokenizer.from_pretrained(original, local_files_only=True)
    body = f"<Instruct>: {INSTRUCTION}\n<Query>: What is the capital of China?\n<Document>: The capital of China is Beijing."
    tokens = sum(
        [tokenizer.encode(s, add_special_tokens=False) for s in [PREFIX, body, SUFFIX]], []
    )
    template = (ROOT / "tools/bench/templates/qwen3_reranker.jinja").read_text()
    rendered = tokenizer.apply_chat_template(
        [
            dict(role="query", content="What is the capital of China?"),
            dict(role="document", content="The capital of China is Beijing."),
        ],
        chat_template=template,
        tokenize=False,
    )
    assert tokenizer.encode(rendered, add_special_tokens=False) == tokens
    custom = "Check the complete document."
    query, document = "Question?\n", "First line.\nSecond line.\n"
    rendered = tokenizer.apply_chat_template(
        [dict(role="document", content=document), dict(role="query", content=query)],
        chat_template=template,
        tokenize=False,
        instruction=custom,
    )
    expected = PREFIX + f"<Instruct>: {custom}\n<Query>: {query}\n<Document>: {document}" + SUFFIX
    assert rendered == expected
    records.append(
        dict(
            kind="benchmark_template",
            exact_token_ids=True,
            custom_instruction=True,
            preserves_final_newlines=True,
        )
    )
    return dict(
        passed=True,
        checks=records,
        reference_token_ids=dict(
            tokens=tokens, ids=[tokenizer.convert_tokens_to_ids(t) for t in ["no", "yes"]]
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    a = p.parse_args()
    report = run(a.original)
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", len(report["checks"]), flush=True)


if __name__ == "__main__":
    main()
