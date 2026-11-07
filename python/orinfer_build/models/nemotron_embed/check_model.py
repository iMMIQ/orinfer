"""Compare native embeddings with original BF16 on real text and document inputs.

Run through tools/environments/cuda/run.sh, which owns the shared GPU lock. The
reference always reads the original checkpoint, never reconstructed Q4 weights.
"""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from orinfer_build.common.files import write_json


def native(args, name, request, graph):
    path = args.output / f"{name}.request.json"
    write_json(path, request)
    started = time.perf_counter()
    result = subprocess.run(
        [str(args.engine), "embed-model", str(args.model), str(path), "--cuda-graph", graph],
        capture_output=True,
        text=True,
    )
    (args.output / f"{name}-{graph}.stderr").write_text(result.stderr)
    result.check_returncode()
    report = json.loads(result.stdout)
    report["process_to_complete_s"] = time.perf_counter() - started
    write_json(args.output / f"{name}-{graph}.json", report)
    vectors = np.array([d["embedding"] for d in report["response"]["data"]], np.float32)
    return vectors, report


def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw
    from transformers import AutoModel, AutoProcessor
    from orinfer_build.compiler.cuda.common import configure

    configure()
    docs = [
        "Jetson AGX Orin 64GB supports CUDA SM87 and uses unified memory.",
        "苹果树在春季开花，秋季结果。果园需要定期灌溉和施肥。",
        "Rust 使用所有权和借用规则管理资源，避免悬空指针。",
        "Embedding models map text and images into vectors for semantic retrieval.",
        "机器翻译把源语言文本转换为目标语言文本。",
        "The GPU executes kernels asynchronously on a CUDA stream.",
        "[IMG] <s> are literal characters in this user document.",
        "图像编码器提取页面中的表格、图表和文字特征。" * 40,
    ]
    queries = ["Orin 有多少内存？", "How does Rust manage memory?", "语义检索如何表示文档？"]
    # A small actual document image with labeled values and a distractor page.
    images = []
    for name, title, text in [
        ("orin", "Jetson AGX Orin", "Memory: 64 GB\nCUDA architecture: SM87\nPower: configurable"),
        ("garden", "Orchard care", "Apple trees\nSpring: flowering\nAutumn: fruit harvest"),
    ]:
        image = Image.new("RGB", (280, 196), "white")
        draw = ImageDraw.Draw(image)
        draw.text((14, 14), title, fill="black")
        draw.multiline_text((14, 55), text, fill="black", spacing=12)
        path = args.output / f"{name}.png"
        image.save(path)
        images.append(path)
    multimodal = [
        dict(images=[str(images[0])]),
        dict(images=[str(images[1])]),
        dict(text=docs[0], images=[str(images[0])]),
        dict(text="Two document pages", images=list(map(str, images))),
    ]
    # Exercise actual bicubic resampling and round-up to the 28-pixel grid.
    resized = args.output / "orin-nonaligned.png"
    with Image.open(images[0]) as image:
        image.resize((301, 207), Image.Resampling.BICUBIC).save(resized)
    multimodal.append(dict(images=[str(resized)]))
    # Duplicate and reversed inputs exercise reused arenas and graph addresses.
    documents = docs + multimodal + [docs[0], multimodal[0]]
    requests = dict(
        documents=dict(input=documents, input_type="document"),
        queries=dict(input=queries, input_type="query"),
    )
    native_vectors, reports = {}, {}
    for name, request in requests.items():
        native_vectors[name], reports[name] = native(args, name, request, "full")
    direct, direct_report = native(args, "documents", requests["documents"], "off")
    graph_error = float(np.max(np.abs(direct - native_vectors["documents"])))
    assert graph_error < 1e-5, graph_error
    assert np.max(np.abs(direct[0] - direct[-2])) < 1e-6
    assert np.max(np.abs(direct[len(docs)] - direct[-1])) < 1e-6
    # Independent opposite admission order must not change a document embedding.
    reverse, _ = native(args, "reverse", dict(input=documents[::-1], input_type="document"), "full")
    assert np.max(np.abs(reverse[::-1] - direct)) < 1e-5
    processor = AutoProcessor.from_pretrained(args.original, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            args.original, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )

    def reference(item, role):
        text = item if isinstance(item, str) else item.get("text")
        source_images = [] if isinstance(item, str) else item.get("images", [])
        content = [dict(type="image", image=Image.open(p).convert("RGB")) for p in source_images]
        if text:
            content.append(dict(type="text", text=text))
        messages = [
            dict(role="system", content="query:" if role == "query" else "passage:"),
            dict(role="user", content=content),
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        with torch.inference_mode():
            # The checkpoint is bidirectional. Some Transformers releases keep
            # the decoder attention module's default is_causal=True even when
            # the config and mask builder are noncausal; pass the contract explicitly.
            assert model.config.text_config.is_causal is False
            hidden = model(**inputs, is_causal=False).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            vector = (hidden * mask).sum(1) / mask.sum(1)
        return F.normalize(vector.float(), dim=-1)[0].cpu().numpy(), inputs["input_ids"][0].tolist()

    references, records = {}, []
    for name, request in requests.items():
        pairs = [reference(item, request["input_type"]) for item in request["input"]]
        references[name] = np.stack([v for v, _ in pairs])
        actual = native_vectors[name]
        cosine = (actual * references[name]).sum(-1)
        records.append(
            dict(
                kind=name,
                cosine=cosine.tolist(),
                min_cosine=float(cosine.min()),
                input_tokens=[len(ids) for _, ids in pairs],
            )
        )
        write_json(
            args.output / f"{name}-bf16.json",
            dict(vectors=references[name].tolist(), tokens=[ids for _, ids in pairs]),
        )
        assert float(cosine.min()) > 0.90, records[-1]
        assert (
            sum(len(ids) for _, ids in pairs) == reports[name]["response"]["usage"]["prompt_tokens"]
        )
    q, d = native_vectors["queries"], native_vectors["documents"]
    refq, refd = references["queries"], references["documents"]
    ranking, reference_ranking = (
        (q @ d.T).argsort(-1)[:, ::-1],
        (refq @ refd.T).argsort(-1)[:, ::-1],
    )
    # Restrict duplicate-free text documents to make the expected answers unambiguous.
    top = (q @ d[: len(docs)].T).argmax(-1)
    ref_top = (refq @ refd[: len(docs)].T).argmax(-1)
    assert np.array_equal(top, ref_top), (top, ref_top)
    return dict(
        passed=True,
        reference="Original BF16 checkpoint; SDPA with explicit is_causal=False",
        comparisons=records,
        graph_max_abs=graph_error,
        text_top1=top.tolist(),
        bf16_text_top1=ref_top.tolist(),
        ranking=ranking.tolist(),
        bf16_ranking=reference_ranking.tolist(),
        native_load=reports["documents"]["load"],
        document_inference_s=reports["documents"]["inference_s"],
        query_inference_s=reports["queries"]["inference_s"],
        direct_inference_s=direct_report["inference_s"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["output", "model", "original", "engine"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    write_json(args.output / "report.json", run(args))
    print(
        "PASS native text, image, multi-image, request order, graph and BF16 comparison", flush=True
    )


if __name__ == "__main__":
    main()
