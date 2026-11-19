"""Generate a deterministic synthetic workload for the AgentX Mini runner."""

import argparse
from pathlib import Path

from orinfer_build.common.files import atomic_json, file_hash


BLOCK_TOKENS = 64
MODEL = "synthetic-mini-weka"
WORKLOADS = {
    "compact": ("orinfer-synthetic-mini-weka-compact-v1", (2, 3, 4, 6)),
    "long": ("orinfer-synthetic-mini-weka-v1", (8, 24, 48, 80)),
}
DEFAULT_TRAJECTORIES = {"compact": 512, "long": 16}
# In trace order after the cold request: fork, child 1, child 2, join, follow-up, final.
TOOL_INPUT_TOKENS = {
    "compact": (256, 512, 768, 768, 512, 256),
    "long": (512, 768, 1536, 2048, 1024, 1024),
}


def trace(trace_id, cold_input_tokens, first_block, tool_input_tokens):
    fork_tokens, child_one_tokens, child_two_tokens, join_tokens, follow_tokens, final_tokens = (
        tool_input_tokens
    )
    next_block = first_block

    def extend(
        prefix, suffix_tokens, timestamp, output_tokens, *, tool_result=False, stop="tool_use"
    ):
        nonlocal next_block
        end = next_block + suffix_tokens // BLOCK_TOKENS
        hashes = prefix + list(range(next_block, end))
        next_block = end
        return {
            "type": "s",
            "t": timestamp,
            "model": MODEL,
            "in": len(hashes) * BLOCK_TOKENS,
            "out": output_tokens,
            "hash_ids": hashes,
            "input_types": ["tool_result" if tool_result else "text"],
            "output_types": ["text"],
            "stop": stop,
            "api_time": 0.25,
        }

    cold = extend([], cold_input_tokens, 0.0, 32)
    fork = extend(cold["hash_ids"], fork_tokens, 0.5, 96, tool_result=True)
    branches = []
    for agent_id, spawn, suffix_tokens, output_tokens in (
        ("worker-1", 1.0, child_one_tokens, 48),
        ("worker-2", 1.125, child_two_tokens, 64),
    ):
        # Inner timestamps are relative to the subagent marker.
        child = extend(fork["hash_ids"], suffix_tokens, 0.0, output_tokens, stop="end_turn")
        branches.append(
            {
                "type": "subagent",
                "agent_id": agent_id,
                "subagent_type": "Explore",
                "t": spawn,
                "duration_ms": 250,
                "status": "completed",
                "models": [MODEL],
                "requests": [child],
            }
        )
    joined = extend(fork["hash_ids"], join_tokens, 1.5, 128, tool_result=True)
    follow_up = extend(joined["hash_ids"], follow_tokens, 2.0, 256, tool_result=True)
    final = extend(follow_up["hash_ids"], final_tokens, 2.5, 96, tool_result=True, stop="end_turn")
    return {
        "id": trace_id,
        "models": [MODEL],
        "block_size": BLOCK_TOKENS,
        "hash_id_scope": "local",
        "system_tokens": 512,
        "requests": [cold, fork, *branches, joined, follow_up, final],
    }


def generate(output, profile="compact", trajectories=None):
    workload, cold_lengths = WORKLOADS[profile]
    if trajectories is None:
        trajectories = DEFAULT_TRAJECTORIES[profile]
    if trajectories <= 0 or trajectories % len(cold_lengths):
        raise ValueError(f"Trajectories must be a positive multiple of {len(cold_lengths)}")
    rounds = trajectories // len(cold_lengths)
    width = max(2, len(str(trajectories - 1)))
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "traces"
    dataset.mkdir()
    windows = []
    for lane, cold_kib in enumerate(cold_lengths * rounds):
        trace_id = f"synthetic-mini-{lane:0{width}d}"
        path = dataset / f"{trace_id}.json"
        item = trace(trace_id, cold_kib * 1024, lane * 4096 + 1, TOOL_INPUT_TOKENS[profile])
        atomic_json(path, item, compact=True)
        windows.append({"trace": trace_id, "sha256": file_hash(path)})
    atomic_json(
        output / "manifest.json",
        {"workload": workload, "trajectories": len(windows), "windows": windows},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory")
    parser.add_argument(
        "--profile",
        choices=WORKLOADS,
        default="compact",
        help="Compact prompts for mixed inference diagnostics, or the original long-input stress test",
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        help="Positive multiple of 4 (default: 512 for compact, 16 for long)",
    )
    args = parser.parse_args()
    try:
        generate(args.output, args.profile, args.trajectories)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
