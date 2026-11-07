"""Compile and validate tuned fixed and runtime-row decode kernels on SM87."""

import argparse
from pathlib import Path

from orinfer_build.contracts.projection_policy import SPECIALIZATIONS
from orinfer_build.tuning.projection_tuning import ProjectionTuner


def main():
    from orinfer_build.compiler.cuda.common import configure, write_json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=list(SPECIALIZATIONS))
    args = p.parse_args()
    configure()
    tuner = ProjectionTuner(args.output)
    for kind in ("q4", "w8", "swiglu", "expert"):
        for rows, dynamic in [(row, False) for row in args.rows] + [(128, True)]:
            print("TUNE", kind, rows, dynamic, flush=True)
            tuner.compile(kind, rows, 128, 256, dynamic=dynamic, experts=16)
    write_json(args.output / "results.json", dict(complete=True, tuned=len(tuner.results)))


if __name__ == "__main__":
    main()
