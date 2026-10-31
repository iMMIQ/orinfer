"""Build owner-row decoder SWA replay with shared dynamic A2 templates.

The retained suffix and state policy are fixed by the execution package.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--keep",
        type=int,
        default=128,
        help="Retained decoder rows per owner (128..512 for owner-row plans)",
    )
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument(
        "--final-prompt-only",
        action="store_true",
        help="Run decoder replay only for the final prompt window; reserve that window on prefix restore",
    )
    parser.add_argument(
        "--source-kv-boundary",
        action="store_true",
        help="Select query rows immediately after the last shared KV writes",
    )
    args = parser.parse_args()
    if not json.loads(args.plan.read_text()).get("batch"):
        parser.error("Decoder replay requires a complete owner-row plan")
    from tools.model.deepseek_v41.decoder_rows import build

    print(
        json.dumps(
            build(
                args.plan,
                args.output,
                args.keep,
                args.compile_workers,
                final_prompt_only=args.final_prompt_only,
                source_kv_boundary=args.source_kv_boundary,
            )
        )
    )


if __name__ == "__main__":
    main()
