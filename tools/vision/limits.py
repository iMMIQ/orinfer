"""Exercise the declared image capacity and recovery after rejected requests."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.manifest.read_text())["vision"]
    assert spec["max_patches"] == 32768, (
        "This capacity fixture targets the default 32768-patch adapter"
    )
    headers = {"Content-Type": "application/json"}
    if os.getenv("ORINFER_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["ORINFER_API_KEY"]
    checks = []

    def save():
        args.output.write_text(
            json.dumps({"status": "running", "checks": checks}, ensure_ascii=False, indent=2) + "\n"
        )

    def request(name, content, *, status=200, **options):
        body = dict(
            model=args.model,
            messages=[dict(role="user", content=content)],
            temperature=0,
            seed=20261002,
            max_tokens=16,
        )
        body.update(options)
        start = time.monotonic()
        req = urllib.request.Request(
            args.base_url + "/chat/completions", data=json.dumps(body).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                code, result = response.status, json.load(response)
        except urllib.error.HTTPError as error:
            code, result = error.code, json.load(error)
        assert code == status, (name, code, result)
        if status != 200:
            assert result["error"]["type"] == "invalid_request_error", result
        else:
            assert result["choices"][0]["finish_reason"] == "stop", result
        checks.append(
            dict(case=name, status=code, response=result, elapsed_s=time.monotonic() - start)
        )
        save()
        print(name, code, result.get("choices", result.get("error")), flush=True)
        return result

    def image(color, size):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        return {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            },
        }

    def text(value):
        return {"type": "text", "text": value}

    invalid = [
        ("invalid-base64", {"url": "data:image/png;base64,!!"}),
        ("invalid-image", {"url": "data:image/png;base64,AA=="}),
        ("missing-url", {}),
        ("unsupported-scheme", {"url": "file:///tmp/image.png"}),
        ("invalid-detail", {"url": "data:image/png;base64,AA==", "detail": "invalid"}),
    ]
    for name, value in invalid:
        request(name, [{"type": "image_url", "image_url": value}], status=400)
    green = image("green", (4096, 2048))
    request("image-context-overflow", [green, text("Name the color.")], status=400, max_tokens=1024)
    request("aggregate-image-capacity", [green, green, text("Name the colors.")], status=400)
    result = request(
        "maximum-image-capacity",
        [green, text("What is the background color? Reply with one English color word.")],
    )
    assert result["usage"]["prompt_tokens"] >= spec["max_patches"] // 4
    assert "green" in result["choices"][0]["message"]["content"].lower(), result
    request("unbound-image-marker", "<|vision_start|><|image_pad|><|vision_end|>", status=400)

    # Different image payloads arrive together and must remain request-private.
    def color_request(color):
        body = dict(
            model=args.model,
            messages=[
                dict(
                    role="user",
                    content=[
                        image(color, (256, 256)),
                        text("Name the background color. One English word."),
                    ],
                )
            ],
            temperature=0,
            seed=20261002,
            max_tokens=16,
        )
        req = urllib.request.Request(
            args.base_url + "/chat/completions", data=json.dumps(body).encode(), headers=headers
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [(color, pool.submit(color_request, color)) for color in ("red", "blue")]
        for color, job in jobs:
            result = job.result()
            assert color in result["choices"][0]["message"]["content"].lower(), result
            checks.append(dict(case="queued-distinct-" + color, response=result))
            save()
    result = request("text-after-maximum-image", "Reply with exactly OK.")
    assert result["choices"][0]["message"]["content"].strip() == "OK", result
    args.output.write_text(
        json.dumps({"status": "passed", "checks": checks}, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
