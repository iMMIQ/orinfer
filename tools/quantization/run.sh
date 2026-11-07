#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
module="${1:?usage: run.sh flash_next|qwen35|qwen_e8p [converter arguments]}"
shift
case "$module" in
    flash_next|qwen35|qwen_e8p) ;;
    *) echo 'Unknown quantization entrypoint' >&2; exit 2 ;;
esac
cd "$repo_dir"
mkdir -p artifacts/quantization/compiler-cache
if [[ "$module" == flash_next ]]; then
    exec 9>artifacts/gpu-experiment.lock
    flock 9
fi
# qwen35 acquires the same GPU lock per bounded chunk, including synchronization
# and allocator release, so CPU downloads can overlap unrelated GPU validation.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export TILELANG_CACHE_DIR="$repo_dir/artifacts/quantization/compiler-cache"
exec uv run --locked --project "$repo_dir/tools/environments/cuda" \
    python -m "orinfer_build.quantization.$module" "$@"
