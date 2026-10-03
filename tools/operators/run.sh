#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -lt 2 ]]; then
    echo "usage: run.sh tools/operators/opXX_name.py artifacts/operators/opXX_name/runNN [runner args]" >&2
    exit 2
fi
runner="$1"
output_arg="$2"
shift 2
cd "$repo_dir"
if [[ "$output_arg" = /* ]]; then output_dir="$output_arg"; else output_dir="$repo_dir/$output_arg"; fi
if [[ -e "$output_dir" ]]; then echo "Refusing existing output: $output_dir" >&2; exit 2; fi
mkdir -p "$output_dir"
mkdir -p "$output_dir/measurement-source"
cp --parents "$runner" tools/operators/common.py tools/operators/run.sh "$output_dir/measurement-source/"
kernel_file="kernels/operators/$(basename -- "$runner")"
if [[ -f "$kernel_file" ]]; then cp --parents "$kernel_file" "$output_dir/measurement-source/"; fi
exec 9>"$repo_dir/artifacts/gpu-experiment.lock"
flock 9
python3 tools/bench/sample_machine.py --output "$output_dir/machine-before.json"
printf '%s\n' "compile-and-test" > "$output_dir/phase.txt"
sampler_pid=""
container_name="orin-operator-$$"
entrypoint="python3"
command_args=("$runner")
if [[ -n "${ORIN_OPERATOR_SANITIZER:-}" ]]; then
    entrypoint="/usr/local/cuda/bin/compute-sanitizer"
    command_args=(--tool "$ORIN_OPERATOR_SANITIZER" --error-exitcode 99)
    if [[ -n "${ORIN_OPERATOR_SANITIZER_KERNEL_NAME:-}" ]]; then
        # Compute Sanitizer accepts an explicit kernel filter as one argument;
        # never interpret it as shell flags. Preserve the inspection scope.
        command_args+=(--kernel-name "$ORIN_OPERATOR_SANITIZER_KERNEL_NAME")
    fi
    command_args+=(python3 "$runner")
    printf '%q ' "${command_args[@]}" > "$output_dir/sanitizer-command.txt"
    printf '\n' >> "$output_dir/sanitizer-command.txt"
fi
if [[ -n "${ORIN_OPERATOR_NCU:-}" ]]; then
    if [[ -n "${ORIN_OPERATOR_SANITIZER:-}" ]]; then
        echo 'Cannot combine Nsight Compute and Compute Sanitizer' >&2
        exit 2
    fi
    entrypoint="/usr/local/cuda/bin/ncu"
    # Keep clocks and caches untouched. Explicit candidate symbols prevent
    # accidentally profiling a same-run reference instead of the candidate.
    command_args=(--clock-control none --cache-control none --set detailed
        --kernel-name-base function --kernel-name "${ORIN_OPERATOR_NCU_KERNEL_NAME:-regex:^kernel_kernel$}"
        --launch-count 1 --export "$output_dir/ncu" python3 "$runner")
    printf '%q ' "${command_args[@]}" > "$output_dir/ncu-command.txt"
    printf '\n' >> "$output_dir/ncu-command.txt"
fi
cleanup() {
    docker rm -f "$container_name" >/dev/null 2>&1 || true
    touch "$output_dir/sampler.stop"
    if [[ -n "$sampler_pid" ]]; then wait "$sampler_pid" || true; fi
    python3 tools/bench/sample_machine.py --output "$output_dir/machine-after.json" || true
}
trap cleanup EXIT
python3 tools/bench/sample_continuous.py --output "$output_dir/machine.jsonl" \
    --phase "$output_dir/phase.txt" --stop "$output_dir/sampler.stop" &
sampler_pid=$!
docker run --rm --name "$container_name" --runtime nvidia --network none \
    --entrypoint "$entrypoint" --shm-size 2g -v /home/nvidia/model:/home/nvidia/model \
    -w "$repo_dir" -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
    -e PYTHONPATH="$repo_dir:$repo_dir/tools/operators" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e ORIN_OPERATOR_OUTPUT="$output_dir" -e TILELANG_CACHE_DIR="$output_dir/cache" \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
    "${ORIN_OPERATOR_IMAGE:-lada-orin-tilelang:0.11.0-exp7}" "${command_args[@]}" \
    --output "$output_dir" "$@" 2>&1 | tee "$output_dir/run.log"
