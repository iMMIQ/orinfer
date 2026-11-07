"""Build reusable A2 DSpark cost priors from opt-in native timing traces.

Run the representative short workload with ORINFER_A2_DSPARK_POLICY_TRACE=1,
then consume this output with ORINFER_A2_DSPARK_COST_MODEL. No weights are read.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median

from tools.build.files import atomic_json


def profile(paths):
    identity = None
    groups = defaultdict(list)
    decode = False
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.startswith("DSPARK_COST_IDENTITY "):
                current = json.loads(line.split(" ", 1)[1])
                if current["schema_version"] not in (1, 2):
                    raise ValueError("Unsupported DSpark timing schema")
                current["schema_version"] = 2
                decode = False
                if identity is not None and identity != current:
                    raise ValueError("DSpark traces have different execution identities")
                identity = current
            elif line.startswith("DSPARK_BASELINE "):
                baseline = json.loads(line.split(" ", 1)[1])
                decode = baseline["batch"] > 0 and not baseline["mixed"]
            elif line.startswith("DSPARK_POLICY "):
                decode = False
            elif line.startswith("DSPARK_COST "):
                if identity is None:
                    raise ValueError("DSpark cost trace has no execution identity")
                point = json.loads(line.split(" ", 1)[1])
                if point.get("stage") == "target" and decode:
                    point["stage"] = "decode"
                if not (
                    math.isfinite(point["seconds"])
                    and point["seconds"] > 0
                    and math.isfinite(point["setup_s"])
                    and point["setup_s"] >= 0
                ):
                    raise ValueError("Invalid DSpark timing")
                shared_setup = point.get("shared_setup_s", 0.0)
                if not math.isfinite(shared_setup) or not 0 <= shared_setup <= point["setup_s"]:
                    raise ValueError("Invalid shared Graph preparation timing")
                if point.get("shared") and point["replay"]:
                    raise ValueError("Shared Graph is not a full replay")
                if not point.get("shared"):
                    point.pop("shared", None)
                shape = {
                    k: v
                    for k, v in point.items()
                    if k not in ("seconds", "setup_s", "shared_setup_s")
                }
                work = shape["work"] = dict(point["work"])
                work["context"] = 1 << (max(1, work["context"]) - 1).bit_length()
                # The runtime uses the same context/prompt-work neighborhoods.
                # Keep an actual representative prefill count in the output.
                work["prefill_rows"] = (work["prefill_rows"] + 127) // 128
                key = json.dumps(shape, sort_keys=True)
                groups[key].append(point)
    if identity is None or not groups:
        raise ValueError("No identified DSpark cost observations")
    points = []
    for key, observations in sorted(groups.items()):
        shared = [
            p.get("shared_setup_s", 0.0) for p in observations if p.get("shared_setup_s", 0.0) > 0
        ]
        local = [
            p["setup_s"] - p.get("shared_setup_s", 0.0)
            for p in observations
            if p["setup_s"] > p.get("shared_setup_s", 0.0)
        ]
        points.append(
            dict(
                {
                    k: v
                    for k, v in observations[0].items()
                    if k not in ("seconds", "setup_s", "shared_setup_s")
                },
                seconds=median(p["seconds"] for p in observations),
                setup_s=(median(local) if local else 0.0) + (median(shared) if shared else 0.0),
                **({"shared_setup_s": median(shared)} if shared else {}),
            )
        )
    if len(points) > 2048:
        raise ValueError("Profile exceeds 2048 points; select representative traces")
    return {"identity": identity, "points": points}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = profile(args.log)
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "points": len(result["points"])}))


if __name__ == "__main__":
    main()
