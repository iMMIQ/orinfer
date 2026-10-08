"""Freeze same-weight logits for Rust deployment migration, including prefill buckets.

This tests execution equivalence, not the independently measured BF16 quality.
"""

import argparse
from pathlib import Path
import shutil

from tools.model.flash_next.checkpoint import Checkpoint
from tools.model.flash_next.native import Model
from tools.model.flash_next.scenes import scenes
from tools.model.flash_next.prepare import PREFILL_PROFILES
from tools.operators.common import configure, write_json


def freeze(model, checkpoint, output):
    tokenizer, cases = scenes(checkpoint)
    cases = cases[:3]
    for width in (128, 512, 2048, 4096):
        prefix = tokenizer.encode("记录编号：甲乙丙丁。" * width, add_special_tokens=False)
        prompt = cases[2]["prompt_ids"]
        # Retain the exact chat-template suffix and exercise every fixed-width plan.
        cases.append(
            dict(
                id=f"prefill-{width}",
                prompt_ids=prompt[:3] + prefix[:width] + prompt[3:],
                target_ids=cases[2]["target_ids"],
            )
        )
    original_plan = model.plan
    for m in (*reversed(PREFILL_PROFILES), 1):
        model.position = model.capacity - m
        original_plan(m)
    model.position = 0
    model.plan = lambda m, **kwargs: model.plans[m, model.capacity, False]
    reference = []
    try:
        for case in cases:
            model.reset()
            prompt = case["prompt_ids"]
            cursor = 0
            while cursor < len(prompt):
                rows = next(
                    m for m in (*reversed(PREFILL_PROFILES), 1) if m <= len(prompt) - cursor
                )
                logits = model.execute(
                    prompt[cursor : cursor + rows],
                    output="logits" if cursor + rows == len(prompt) else "none",
                )
                cursor += rows
            history = list(prompt)
            for index, token in enumerate(case["target_ids"][:4]):
                if index:
                    logits = model.execute([case["target_ids"][index - 1]], output="logits")
                file = f"reference-{case['id']}-{index}.f32"
                logits.astype("<f4").tofile(output / file)
                reference.append(
                    dict(
                        case=case["id"],
                        position=index,
                        history=list(history),
                        target=token,
                        logits=file,
                    )
                )
                history.append(token)
            print("FROZEN", case["id"], len(prompt), flush=True)
        write_json(output / "reference.json", reference)
        write_json(
            output / "score-requests.json",
            dict(
                seed=20261002,
                cases=[
                    dict(id=c["id"], prompt_ids=c["prompt_ids"], target_ids=c["target_ids"][:4])
                    for c in cases
                ],
            ),
        )
    finally:
        model.plan = original_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile-cache", type=Path)
    args = p.parse_args()
    configure()
    if args.compile_cache:
        cache = args.output / "cache/0.1.15"
        if cache.exists():
            if any(cache.rglob("*.cubin")):
                raise ValueError("Refusing to replace populated compile cache")
            shutil.rmtree(cache)
        cache.symlink_to(args.compile_cache.resolve(), target_is_directory=True)
    model = Model(
        Checkpoint(args.checkpoint, verify_hashes=False), 262144, args.output, use_graph=False
    )
    freeze(model, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
