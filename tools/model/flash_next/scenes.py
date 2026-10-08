"""Small public, deterministic scenes for native model execution and scoring."""

from pathlib import Path
import argparse
import hashlib
import json

from tools.model.flash_next.validation import CASES
from tools.eval.scoring_common import SEED


def scenes(checkpoint):
    from transformers import AutoTokenizer

    checkpoint = Path(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    template = (checkpoint / "chat_template.jinja").read_text()
    result = []
    for name, question, answer in CASES:
        messages = [{"role": "user", "content": question}]
        ids = tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rule = {"kind": "exact", "expected": answer}
        if name == "json":
            rule = {"kind": "json_equal", "expected": {"name": "Alice", "age": 30}}
        if name == "code":
            rule = {"kind": "manual", "expected": "Valid Python function returning a + b"}
        result.append(
            {
                "id": name,
                "messages": messages,
                "thinking": False,
                "prompt_ids": ids,
                "target_ids": tokenizer.encode(answer, add_special_tokens=False),
                "rule": rule,
            }
        )
    messages = [{"role": "user", "content": CASES[2][1]}]
    answer = "17×23=17×(20+3)=340+51=391。\n</think>\n\n391"
    ids = tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="low",
    )
    result.append(
        {
            "id": "math-thinking",
            "messages": messages,
            "thinking": True,
            "prompt_ids": ids,
            "target_ids": tokenizer.encode(answer, add_special_tokens=False),
            "rule": {"kind": "manual", "expected": "Completed thinking followed by answer 391"},
        }
    )
    messages = [
        {"role": "user", "content": "记住编号：BLUE-73。"},
        {"role": "assistant", "content": "我记住了。"},
        {"role": "user", "content": "只输出刚才的编号。"},
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    result.append(
        {
            "id": "multi-turn",
            "messages": messages,
            "thinking": False,
            "prompt_ids": ids,
            "target_ids": tokenizer.encode("BLUE-73", add_special_tokens=False),
            "rule": {"kind": "exact", "expected": "BLUE-73"},
        }
    )
    return tokenizer, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite frozen scenes")
    _, cases = scenes(args.checkpoint)
    files = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    manifest = {
        "seed": SEED,
        "cases": cases,
        "scope": "Fixed authored answers; no model probabilities yet",
        "frontend_sha256": {
            name: hashlib.sha256((args.checkpoint / name).read_bytes()).hexdigest()
            for name in files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "prompt_tokens": sum(len(c["prompt_ids"]) for c in cases),
                "target_tokens": sum(len(c["target_ids"]) for c in cases),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
