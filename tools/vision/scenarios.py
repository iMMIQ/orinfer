"""Image semantics, HTTP loading, chunk crossing, history and tool API checks."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    headers = {"Content-Type": "application/json"}
    if os.getenv("ORINFER_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["ORINFER_API_KEY"]

    def request(messages, **options):
        body = {
            "model": a.model,
            "messages": messages,
            "temperature": 0,
            "seed": 20261002,
            "max_tokens": 64,
            **options,
        }
        t = time.monotonic()
        r = urllib.request.Request(
            a.base_url + "/chat/completions", data=json.dumps(body).encode(), headers=headers
        )
        with urllib.request.urlopen(r, timeout=300) as response:
            out = json.load(response)
        return out, time.monotonic() - t

    def part(im):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            },
        }

    def text(s):
        return {"type": "text", "text": s}

    def check(name, out, elapsed, passed):
        record = {"case": name, "response": out, "elapsed_s": elapsed, "passed": passed}
        results.append(record)
        print(name, out["choices"][0]["message"], round(elapsed, 3), passed, flush=True)
        a.output.write_text(
            json.dumps({"status": "running", "checks": results}, ensure_ascii=False, indent=2)
            + "\n"
        )

    red = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(red).rectangle((96, 96, 416, 416), fill="red")
    blue = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(blue).rectangle((96, 96, 416, 416), fill="blue")
    out, t = request(
        [
            {
                "role": "user",
                "content": [
                    part(red),
                    part(blue),
                    text("Give the square colors in image order. Only two English color words."),
                ],
            }
        ]
    )
    answer = out["choices"][0]["message"]["content"].lower()
    check(
        "two-images-cross-prefill-block",
        out,
        t,
        out["usage"]["prompt_tokens"] >= 512
        and "red" in answer
        and "blue" in answer
        and answer.index("red") < answer.index("blue"),
    )
    ocr = Image.new("RGB", (544, 320), "white")
    draw = ImageDraw.Draw(ocr)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
    draw.text((40, 110), "ORIN 2026", fill="black", font=font)
    out, t = request(
        [
            {
                "role": "user",
                "content": [
                    text("Read the large text in this image. Output only the text."),
                    part(ocr),
                ],
            }
        ]
    )
    answer = out["choices"][0]["message"]["content"]
    check("non-square-ocr", out, t, "ORIN" in answer.upper() and "2026" in answer)
    count = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(count)
    for xy in [(25, 25, 85, 85), (155, 25, 215, 85), (90, 155, 150, 215)]:
        draw.ellipse(xy, fill="blue")
    out, t = request(
        [
            {
                "role": "user",
                "content": [
                    part(count),
                    text("How many blue circles are there? Output only a digit."),
                ],
            }
        ]
    )
    check("counting", out, t, "3" in out["choices"][0]["message"]["content"])
    smallblue = blue.resize((256, 256))
    smallred = red.resize((256, 256))
    history = [
        {"role": "user", "content": [part(smallred), text("Name the square color.")]},
        {"role": "assistant", "content": "Red."},
        {
            "role": "user",
            "content": [
                part(smallblue),
                text("What is the square color in the most recent image? One English word."),
            ],
        },
    ]
    out, t = request(history)
    check("image-history", out, t, "blue" in out["choices"][0]["message"]["content"].lower())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record_color",
                "description": "Record the color of the square in the supplied image.",
                "parameters": {
                    "type": "object",
                    "properties": {"color": {"type": "string", "enum": ["red", "blue"]}},
                    "required": ["color"],
                },
            },
        }
    ]
    out, t = request(
        [
            {
                "role": "user",
                "content": [
                    part(smallblue),
                    text("Look at this image and call record_color with its square color."),
                ],
            }
        ],
        tools=tools,
        tool_choice="required",
        max_tokens=96,
    )
    calls = out["choices"][0]["message"].get("tool_calls", [])
    check(
        "image-tool-call",
        out,
        t,
        len(calls) == 1
        and calls[0]["function"]["name"] == "record_color"
        and json.loads(calls[0]["function"]["arguments"])["color"] == "blue",
    )
    image_bytes = {}
    for color, image in [("red", smallred), ("blue", smallblue)]:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes["/" + color + ".png"] = buf.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            png = image_bytes[self.path]
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = []
            for color in ("red", "blue"):
                url = f"http://127.0.0.1:{server.server_port}/{color}.png"
                content = [
                    {"type": "image_url", "image_url": {"url": url}},
                    text("Name the square color. One English word."),
                ]
                jobs.append((color, pool.submit(request, [{"role": "user", "content": content}])))
            for color, job in jobs:
                out, t = job.result()
                check(
                    f"queued-http-image-{color}",
                    out,
                    t,
                    color in out["choices"][0]["message"]["content"].lower(),
                )
    finally:
        server.shutdown()
        server.server_close()
    a.output.write_text(
        json.dumps(
            {
                "status": "passed" if all(r["passed"] for r in results) else "failed",
                "checks": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    assert all(r["passed"] for r in results), "One or more visual scenarios failed; see result file"


if __name__ == "__main__":
    main()
