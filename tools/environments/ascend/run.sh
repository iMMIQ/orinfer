#!/usr/bin/env bash
# Offline compiler/validation runner. Production execution uses the Rust loader.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
image=${ORINFER_ASCEND_IMAGE:-orinfer-a2-tilelang:0.1.4-d182ae8-cann9.2-arm64}
toolchain_root=${ORINFER_ASCEND_CANN_ROOT:-/usr/local/Ascend/cann-9.2.0}
driver_root=${ORINFER_ASCEND_DRIVER_ROOT:-/usr/local/Ascend/driver}
model_root=${ORINFER_ASCEND_MODEL_ROOT:-/data/models}
wheel_name=tilelang-0.1.4+ubuntu.24.3.cann910-cp312-cp312-linux_aarch64.whl
wheel_dir=$(python3 "$repo_dir/python/orinfer_build/common/paths.py" --root "$repo_dir" --kind cache ascend-runtime/wheels)
wheel_url='https://github.com/tile-ai/tilelang-ascend/releases/download/TileLang-ascend-v0.1.2.000-release/tilelang-0.1.4%2Bubuntu.24.3.cann910-cp312-cp312-linux_aarch64.whl'

if [[ ${1:-} == build ]]; then
    base_image=${ORINFER_ASCEND_BASE_IMAGE:?Set ORINFER_ASCEND_BASE_IMAGE to a Python 3.12 / torch_npu compiler frontend image}
    base_id=$(docker image inspect --format '{{.Id}}' "$base_image")
    base_tag="orinfer-ascend-base:build-$$"
    docker tag "$base_id" "$base_tag"
    trap 'docker image rm "$base_tag" >/dev/null 2>&1 || true' EXIT
    mkdir -p "$wheel_dir"
    if [[ ! -f "$wheel_dir/$wheel_name" ]]; then
        curl --fail --location --retry 2 "$wheel_url" -o "$wheel_dir/$wheel_name"
    fi
    docker build --pull=false --build-arg "BASE_IMAGE=$base_tag" -t "$image" \
        -f "$repo_dir/tools/environments/ascend/Dockerfile" "$wheel_dir"
    exit
fi

if [[ $# == 0 ]]; then
    echo 'Usage: run.sh build | run.sh python3 -m tests.ascend.operators.validate --output artifacts/ascend-runtime/aot' >&2
    exit 2
fi
without_devices=false
if [[ ${1:-} == --without-devices ]]; then
    without_devices=true
    shift
    if [[ $# == 0 ]]; then
        echo 'Pass the compiler command after --without-devices.' >&2
        exit 2
    fi
fi
docker image inspect "$image" >/dev/null
if [[ ! -f "$toolchain_root/compiler/version.info" ]] ||
    ! rg -q '^Version=9\.2\.' "$toolchain_root/compiler/version.info"; then
    echo 'Native A2 compilation and validation require a CANN 9.2 installation.' >&2
    exit 2
fi
devices=()
lock=()
if [[ $without_devices == false ]]; then
    for device in /dev/davinci{0..7} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
        devices+=(--device "$device")
    done
    gpu_lock=$(python3 "$repo_dir/python/orinfer_build/common/paths.py" --root "$repo_dir" --gpu-lock)
    mkdir -p "$(dirname -- "$gpu_lock")"
    lock=(flock -w 5 "$gpu_lock")
fi
work_root=$(python3 "$repo_dir/python/orinfer_build/common/paths.py" --root "$repo_dir" --work-root)
mounts=(--mount "type=bind,src=$work_root,dst=$work_root")
if [[ -d $model_root ]]; then
    mounts+=(--mount "type=bind,src=$model_root,dst=$model_root,readonly")
fi
mkdir -p "$repo_dir/artifacts"
"${lock[@]}" docker run --rm --init --runtime runc --shm-size=8g \
    "${devices[@]}" \
    "${mounts[@]}" \
    --mount "type=bind,src=$driver_root,dst=/usr/local/Ascend/driver,readonly" \
    --mount "type=bind,src=$toolchain_root,dst=$toolchain_root,readonly" \
    --env "ASCEND_HOME_PATH=$toolchain_root" \
    --env "PYTHONPATH=/workspace/orinfer/python:/workspace/orinfer" \
    --mount "type=bind,src=$repo_dir,dst=/workspace/orinfer" \
    --workdir /workspace/orinfer --entrypoint /bin/bash "$image" -c \
    'source "$ASCEND_HOME_PATH/set_env.sh"; exec "$@"' orinfer-ascend "$@"
