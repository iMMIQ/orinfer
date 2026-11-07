"""Shared bootstrap for the supported command delegates."""

from orinfer_build.common.paths import ROOT


import runpy
import sys


def run(module):
    # Kernels and validation programs are source-checkout packages. The build
    # library itself is installed by uv, including direct script invocations.
    sys.path.insert(0, str(ROOT))
    runpy.run_module(module, run_name="__main__", alter_sys=True)
