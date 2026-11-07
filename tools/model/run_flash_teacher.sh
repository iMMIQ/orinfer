#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
mkdir -p artifacts
mkdir -p artifacts/quantization/flash-next/teacher-cache
exec 9>artifacts/gpu-experiment.lock
flock 9
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$repo_dir/artifacts/quantization/flash-next/teacher-cache"
export TILELANG_CACHE_DIR="$XDG_CACHE_HOME/tilelang"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec uv run --locked --project "$repo_dir/tools/environments/cuda" \
    python -m orinfer_build.models.flash_next.reference.teacher "$@"
