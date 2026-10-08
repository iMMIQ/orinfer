"""Generate frozen native Chat fixtures and run serialized Orin state/quality regressions.

Never stop a service or touch device power/clocks. GPU ownership is an explicit
nonblocking file lock; the caller must first release its own idle deployment.
"""

import argparse
import base64
import fcntl
import json
from pathlib import Path
import struct
import subprocess
import zlib

ROOT = Path(__file__).resolve().parents[2]


def solid_png(red, green, blue):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    width = height = 128
    raw = (b"\0" + bytes((red, green, blue)) * width) * height
    png = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    )
    return (
        "data:image/png;base64,"
        + base64.b64encode(png + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")).decode()
    )


def fixture_cases():
    def text(value):
        return {"type": "text", "text": value}

    def image(value):
        return {"type": "image_url", "image_url": {"url": value, "detail": "low"}}

    red, blue = solid_png(255, 0, 0), solid_png(0, 0, 255)
    contents = [
        "Reply with the single word OK.",
        "What is 7 plus 5? Reply with the number.",
        [text("Name the color. One word."), image(red)],
        [text("Name the two colors in order."), image(red), image(blue)],
    ]
    return [
        {
            "id": str(i),
            "request": {
                "model": "qwen3.8-27b",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 32,
                "temperature": 0,
                "seed": 20261002,
                "enable_thinking": False,
            },
        }
        for i, content in enumerate(contents)
    ]


def run(args):
    model = args.model.resolve(strict=True)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    lock = ROOT / "artifacts/gpu-experiment.lock"
    lock.parent.mkdir(exist_ok=True)
    with lock.open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("GPU lock is busy; leave foreign services running") from e
        summaries = []
        for mode in args.modes:
            spec = output / f"{mode}-spec.json"
            fixture = output / f"{mode}-fixture.json"
            result = output / f"{mode}-result.json"
            spec.write_text(
                json.dumps(
                    dict(
                        model=str(model),
                        tokenizer=str(model),
                        result=str(result),
                        fixture=str(fixture),
                        cuda_graph=mode,
                        repetitions=1,
                        cases=fixture_cases(),
                    ),
                    indent=2,
                )
            )
            env = dict(
                __import__("os").environ,
                ORINFER_MTP_FIXTURE_SPEC=str(spec),
                ORINFER_BATCH_FIXTURE=str(fixture),
            )
            with (output / f"{mode}.log").open("w") as log:
                subprocess.run(
                    [
                        "cargo",
                        "test",
                        "--locked",
                        "--offline",
                        "-p",
                        "orinfer-api",
                        "export_mtp_chat_fixture",
                        "--",
                        "--ignored",
                        "--test-threads=1",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                subprocess.run(
                    [
                        "cargo",
                        "test",
                        "--release",
                        "--locked",
                        "--offline",
                        "-p",
                        "orinfer-engine",
                        "validate_continuous_requests",
                        "--",
                        "--ignored",
                        "--nocapture",
                        "--test-threads=1",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            report = json.loads(result.read_text())
            summaries.append(dict(mode=mode, result=str(result), cases=len(report["cases"])))
        (output / "summary.json").write_text(
            json.dumps(dict(seed=20261002, validations=summaries), indent=2) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("off", "decode_only", "full"),
        default=["off", "decode_only", "full"],
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
