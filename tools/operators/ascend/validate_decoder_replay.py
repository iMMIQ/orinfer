"""Compatibility command; implementation belongs to the build or validation package."""

from orinfer_build.common.cli import run

if __name__ == "__main__":
    run("tests.ascend.operators.validate_decoder_replay")
