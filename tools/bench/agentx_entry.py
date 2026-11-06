"""Run the installed AgentX CLI with a fixed logical benchmark identity.

AgentX's cache-bust prefix depends on this identity in addition to its random
seed. The adapter stays in the offline benchmark process; installed packages
and the service are untouched.
"""

import argparse
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--benchmark-id", required=True)
    args, remaining = parser.parse_known_args()
    import aiperf.cli_runner as runner

    original = runner._make_benchmark_run

    def fixed_run(config, **kwargs):
        return original(config, **(kwargs | {"benchmark_id": args.benchmark_id}))

    runner._make_benchmark_run = fixed_run
    sys.argv = [args.harness, *remaining]
    runpy.run_path(args.harness, run_name="__main__")


if __name__ == "__main__":
    main()
