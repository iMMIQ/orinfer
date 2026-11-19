"""Compare changed-input and teacher-forced CUDA graph replay on a prepared model."""

import argparse
import json
from pathlib import Path
import subprocess

from orinfer_build.common.files import write_json
from orinfer_build.common.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=ROOT / "target/release/orinfer")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = args.model.resolve(strict=True)
    engine = args.engine.resolve(strict=True)
    subprocess.run([engine, "validate-model", model], check=True)
    requests = [
        dict(id="first", input_tokens=[0], max_new_tokens=5, logits_steps=list(range(5))),
        dict(id="changed", input_tokens=[2], max_new_tokens=5, logits_steps=list(range(5))),
        dict(
            id="forced",
            input_tokens=[1],
            max_new_tokens=5,
            forced_tokens=[3, 0, 2, 1],
            logits_steps=list(range(5)),
        ),
    ]
    report = dict(complete=False, scope=__doc__, modes=[])
    reference = None
    for mode in ("off", "decode_only", "full"):
        request_file = output / "requests.json"
        write_json(
            request_file, dict(requests=requests, logits_output=str(output / f"logits-{mode}"))
        )
        result = json.loads(
            subprocess.run(
                [engine, "run-model", model, request_file, "--cuda-graph", mode],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        write_json(output / f"{mode}.json", result)
        observed = []
        for request, actual in zip(requests, result["requests"], strict=True):
            assert actual["id"] == request["id"], (mode, request["id"])
            logits = [Path(path).read_bytes() for path in actual["logits_files"]]
            assert len(logits) >= 2, (mode, request["id"], "missing decode logits")
            observed.append((actual["output_tokens"], logits))
        if reference is None:
            assert observed[0][1][0] != observed[1][1][0], "Changed input produced identical logits"
            reference = observed
        else:
            assert observed == reference, (mode, "tokens or logits differ from eager execution")
            assert result["captured_programs"], (mode, "no programs captured")
        report["modes"].append(
            dict(mode=mode, requests=len(requests), captured_programs=result["captured_programs"])
        )
        write_json(output / "results.json", report)
    report["complete"] = True
    write_json(output / "results.json", report)


if __name__ == "__main__":
    main()
