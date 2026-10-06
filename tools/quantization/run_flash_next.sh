#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
mkdir -p artifacts/quantization/flash-next
exec 9>artifacts/gpu-experiment.lock
flock 9
# Online frontend remains Rust. This image is only an offline GPU converter.
exec docker run --rm --name "orinfer-flash-quantizer-$$" --runtime nvidia --network host \
    --user "$(id -u):$(id -g)" --entrypoint /usr/bin/python3 \
    -v "$repo_dir:$repo_dir" -w "$repo_dir" \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
    -e HTTPS_PROXY -e HTTP_PROXY -e https_proxy -e http_proxy -e NO_PROXY -e no_proxy \
    -e PYTHONPATH="$repo_dir:/opt/venv/lib/python3.10/site-packages" \
    -e TILELANG_CACHE_DIR="$repo_dir/artifacts/quantization/flash-next/compiler-cache" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
    orinfer-compiler:0.1.1 -m tools.quantization.flash_next "$@"
