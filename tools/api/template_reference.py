#!/usr/bin/env python3
"""Generate offline Transformers token references for Rust chat codec tests."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    tools = [{"type": "function", "function": {"name": "lookup", "description": "Lookup a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    requests = [
        {"messages": [{"role": "user", "content": "你好，2+3等于几？"}]},
        {"messages": [{"role": "system", "content": "Answer briefly."},
                      {"role": "user", "content": [{"type": "text", "text": "天气"}]}], "tools": tools},
        {"messages": [{"role": "user", "content": "查一下北京"},
                      {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "lookup", "arguments": '{"city":"北京"}'}}]},
                      {"role": "tool", "tool_call_id": "a", "content": "晴朗"}], "tools": tools},
        {"messages": [{"role": "user", "content": "2+3?"}], "enable_thinking": True, "reasoning_effort": "low"},
    ]
    processor = None
    vision = None
    if args.vision_model:
        import base64
        import io
        from PIL import Image
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        if args.image is None:
            parser.error("--vision-model requires --image")
        vision_model = args.vision_model / 'cache/manifest.json' if args.vision_model.is_dir() else args.vision_model
        vision = json.loads(vision_model.read_text())["vision"]
        image_processor = Qwen2VLImageProcessor.from_pretrained(args.tokenizer_dir, local_files_only=True)
        image_processor.max_pixels = min(16777216, vision["max_patches"] * vision["patch_size"]**2)
        class ImagesProcessor(Qwen3VLProcessor):
            def check_argument_for_proper_class(self, argument_name, argument):
                # No videos are used. Allow an absent optional video backend on
                # CPU reference hosts; inherit renderer and image processing unchanged.
                if argument_name == "video_processor" and argument is None:
                    return type(None)
                return super().check_argument_for_proper_class(argument_name, argument)
        processor = ImagesProcessor(image_processor, tokenizer, video_processor=None,
                                    chat_template=(args.tokenizer_dir / "chat_template.jinja").read_text())
        image = {"type":"image_url", "image_url":{"url":"data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode()}}
        buf = io.BytesIO()
        Image.new("RGB",(256,256),"blue").save(buf,format="PNG")
        second = {"type":"image_url", "image_url":{"url":"data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}
        text = lambda s: {"type":"text","text":s}
        requests.extend([
            {"messages":[{"role":"user","content":[text("Read the large text in this image. Output only the text."),image]}]},
            {"messages":[{"role":"user","content":[text("Read the large text in this image. Output only the text."),image]}],"enable_thinking":True,"reasoning_effort":"xhigh"},
            {"messages":[{"role":"user","content":[image,text("Describe this image.")]}],"enable_thinking":True},
            {"messages":[{"role":"user","content":[text("Compare"),image,text("and"),second]}]},
            {"messages":[{"role":"user","content":[image,text("What is written?")]},
                         {"role":"assistant","content":"ORIN 2026"},
                         {"role":"user","content":[second,text("Compare with the previous image.")]}]},
            {"messages":[{"role":"user","content":[image,text("Use lookup for the city named in the picture.")]}],"tools":tools},
        ])
    cases = []
    for request in requests:
        request.update(model="qwen3.8-27b", max_tokens=16)
        # serde_json's object representation uses sorted keys. Canonicalize the
        # Python input before applying the unchanged checkpoint template.
        normalized = json.loads(json.dumps(request, sort_keys=True))
        for message in normalized["messages"]:
            for call in message.get("tool_calls", []):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        template_options = dict(tools=normalized.get("tools", []),
            enable_thinking=request.get("enable_thinking", False),
            reasoning_effort=request.get("reasoning_effort", "xhigh"),
            preserve_thinking=True, add_generation_prompt=True)
        images = []
        if processor:
            for message in normalized["messages"]:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        if part["type"] == "image_url":
                            images.append(Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",",1)[1]))).convert("RGB"))
        if images:
            rendered = processor.apply_chat_template(normalized["messages"],tokenize=False,**template_options)
            encoded = processor(text=[rendered],images=images,return_tensors="np")
            ids = encoded["input_ids"][0].tolist()
            grid = encoded["image_grid_thw"].tolist()
        else:
            ids = tokenizer.apply_chat_template(normalized["messages"],**template_options)
        if hasattr(ids, "get"):
            ids = ids["input_ids"]
        # The processor itself expands vision markers; the reference does not
        # copy the Rust marker expansion implementation.
        case = {"request":request,"ids":ids}
        if images:
            from types import SimpleNamespace
            import torch
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
            config = Qwen3_5Config.from_pretrained(args.tokenizer_dir, local_files_only=True)
            positions, delta = Qwen3_5Model.get_rope_index(
                SimpleNamespace(config=config), torch.tensor([ids]),
                image_grid_thw=torch.tensor(grid))
            # Runtime stores token-major [t,h,w], including generated positions.
            rope = positions[:,0].T.flatten().tolist()
            start = len(ids)+int(delta[0,0])
            for step in range(16):
                rope.extend([start+step]*3)
            case.update(vision=vision,image_grid_thw=grid,mrope_positions=rope)
        cases.append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(cases, output, ensure_ascii=False)
    print("Reference lengths:", [len(case["ids"]) for case in cases])
    print("ORIN_CHAT_REFERENCE=" + str(args.output.resolve()))


if __name__ == "__main__":
    main()
