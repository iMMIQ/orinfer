"""Publish measured W4 tiling hints with reusable capacity descriptors.

Select the expected rows per expert from real-weight operator measurements and
confirm the resulting plan with full requests. Device routing counts continue to
set the live work; this transform preserves weights, arithmetic and row kernels.
"""

import argparse
import json
from pathlib import Path

from tools.model.deepseek_v41.assembly import publish_package, validate_rows


def transform(model, expected_expert_rows):
    if not 0 < expected_expert_rows < 2**63:
        raise ValueError("Expected a positive int64 expert row estimate")
    changes = 0
    for rank in model["ranks"]:
        definitions = {b["name"]: b for b in rank["buffers"]}
        for packet in rank["packets"]:
            steps = packet["body"] + packet["head"]
            host_routed = {name for s in steps if s["op"] == "route" for name in s["groups"]}
            for step in steps:
                if step["op"] != "w4":
                    continue
                if not step.get("groups") or step["name"] in host_routed:
                    raise ValueError("Capacity W4 execution requires device-owned routing counts")
                # Capacity descriptors may touch inactive scratch. Keep this
                # publication restricted to dedicated allocations, not aliases.
                operands = step["buffers"] + [step["groups"], step["workspace"]]
                if any(definitions[name].get("view") for name in operands):
                    raise ValueError("Capacity W4 operands require dedicated allocations")
                step["expected_expert_rows"] = expected_expert_rows
                changes += 1
    if not changes:
        raise ValueError("No device-routed W4 instructions found")
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--package-output", type=Path, required=True)
    parser.add_argument("--expected-expert-rows", type=int, required=True)
    args = parser.parse_args()
    model = json.loads(args.plan.read_text())
    changes = transform(model, args.expected_expert_rows)
    validate_rows(model, args.plan.parent)
    assets = publish_package(model, args.plan.parent, args.package_output)
    print(json.dumps(dict(w4_instructions=changes, assets=assets)))


if __name__ == "__main__":
    main()
