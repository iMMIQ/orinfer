"""Compare complete native rerank requests against the original BF16 source."""

import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from orinfer_build.common.files import write_json
from orinfer_build.models.qwen_reranker.config import PREFIX, SUFFIX, INSTRUCTION


def native(a, name, request, graph="full"):
    path = a.output / f"{name}.request.json"
    write_json(path, request)
    started = time.perf_counter()
    result = subprocess.run(
        [str(a.engine), "rerank-model", str(a.model), str(path), "--cuda-graph", graph],
        text=True,
        capture_output=True,
    )
    (a.output / f"{name}-{graph}.stderr").write_text(result.stderr)
    result.check_returncode()
    report = json.loads(result.stdout)
    report["process_to_complete_s"] = time.perf_counter() - started
    write_json(a.output / f"{name}-{graph}.json", report)
    return np.array([s["logit_difference"] for s in report["scores"]]), report


def run(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from orinfer_build.compiler.cuda.common import configure

    configure()
    requests = [
        dict(
            query="What is the capital of China?",
            documents=[
                "The capital of China is Beijing.",
                "Gravity attracts objects with mass.",
                "Apples ripen in autumn.",
                "The capital of China is Beijing.",
            ],
        ),
        dict(
            query="Jetson AGX Orin 64GB 有多少内存？",
            documents=[
                "Jetson AGX Orin 64GB 配备 64GB 统一内存，GPU 与 CPU 共享。",
                "苹果树在春天开花，秋天结果。",
                "Rust 所有权避免悬空指针。",
            ],
        ),
        dict(
            query="How does Rust prevent dangling pointers?",
            documents=[
                "Rust checks ownership and borrowing to prevent invalid references and dangling pointers.",
                "The GPU launches asynchronous CUDA kernels.",
                "China has many historic cities.",
            ],
        ),
        dict(
            query="Python 如何把整数列表升序排序？",
            documents=[
                "调用 sorted(values) 返回升序的新列表；values.sort() 原地排序。",
                "Use a database transaction to commit changes.",
                "Text embeddings produce vectors for retrieval.",
            ],
        ),
        dict(
            query="Which document contains a French greeting?",
            instruction="Select the document containing a French greeting, regardless of the other topics.",
            documents=[
                "Bonjour, bienvenue dans notre magasin.",
                "Hello, the CUDA stream is ready.",
                "这是一个中文问候。",
            ],
        ),
        dict(
            query="How much memory does the Orin have?",
            documents=[
                "Orchard care and fruit harvest. " * 250
                + "Jetson AGX Orin 64GB provides 64GB unified memory.",
                "An apple orchard needs water and nutrients.",
            ],
        ),
    ]
    margins = []
    reports = []
    for i, request in enumerate(requests):
        margin, report = native(a, f"case{i}", request)
        margins.append(margin)
        reports.append(report)
    direct, _ = native(a, "case0", requests[0], "off")
    graph_error = float(np.max(np.abs(direct - margins[0])))
    assert graph_error < 1e-5, graph_error
    reversed_request = dict(requests[0], documents=requests[0]["documents"][::-1])
    reverse, _ = native(a, "reverse", reversed_request)
    assert np.max(np.abs(reverse[::-1] - direct)) < 1e-5
    assert direct[0] == direct[-1]
    tokenizer = AutoTokenizer.from_pretrained(a.original, local_files_only=True)
    ids = [tokenizer.convert_tokens_to_ids(t) for t in ["no", "yes"]]
    prefix, suffix = [tokenizer.encode(s, add_special_tokens=False) for s in [PREFIX, SUFFIX]]
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.original, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    records = []
    golden = None
    for i, request in enumerate(requests):
        reference = []
        lengths = []
        tokens_per_doc = []
        for document in request["documents"]:
            body = f"<Instruct>: {request.get('instruction', INSTRUCTION)}\n<Query>: {request['query']}\n<Document>: {document}"
            tokens = prefix + tokenizer.encode(body, add_special_tokens=False) + suffix
            tokens_per_doc.append(tokens)
            lengths.append(len(tokens))
            inputs = torch.tensor([tokens], device="cuda")
            with torch.inference_mode():
                logits = (
                    model(
                        input_ids=inputs,
                        attention_mask=torch.ones_like(inputs),
                        logits_to_keep=1,
                        use_cache=False,
                    )
                    .logits[0, -1, ids]
                    .float()
                )
            reference.append(float(logits[1] - logits[0]))
        if i == 0:
            golden = dict(tokens=tokens_per_doc[0], ids=ids)
        actual = margins[i]
        reference = np.array(reference)
        p = 1 / (1 + np.exp(-reference))
        q = 1 / (1 + np.exp(-actual))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        q = np.clip(q, 1e-12, 1 - 1e-12)
        kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        native_top = int(actual.argmax())
        reference_top = int(reference.argmax())
        records.append(
            dict(
                case=i,
                native_margin=actual.tolist(),
                bf16_margin=reference.tolist(),
                native_top=native_top,
                bf16_top=reference_top,
                top_matches=native_top == reference_top,
                probability_mae=float(np.abs(q - p).mean()),
                binary_kl=kl.tolist(),
                input_tokens=lengths,
            )
        )
        assert sum(lengths) == reports[i]["response"]["usage"]["prompt_tokens"]
    all_kl = np.concatenate([r["binary_kl"] for r in records])
    mae = np.mean([r["probability_mae"] for r in records])
    return dict(
        passed=all(r["top_matches"] for r in records) and float(all_kl.mean()) < 0.01,
        reference="Original BF16, causal Qwen3ForCausalLM and two-class last-position logits",
        comparisons=records,
        mean_binary_kl=float(all_kl.mean()),
        mean_case_probability_mae=float(mae),
        graph_max_abs=graph_error,
        reference_token_ids=golden,
        native_load=reports[0]["load"],
        request_inference_s=[r["inference_s"] for r in reports],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["output", "model", "original", "engine"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    report = run(a)
    write_json(a.output / "report.json", report)
    assert report["passed"], {
        k: report[k] for k in ["mean_binary_kl", "mean_case_probability_mae", "comparisons"]
    }
    print("PASS native reranking, causal BF16, Graph and request isolation", flush=True)


if __name__ == "__main__":
    main()
