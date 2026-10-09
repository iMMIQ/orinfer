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
container_name="orinfer-operator-$$"
extra_mounts=()
group_args=()
for group_id in $(id -G); do group_args+=(--group-add "$group_id"); done
reference_env=()
declare -A mounted_inputs=()
for variable in ORINFER_CHECKPOINT_DIR ORINFER_REFERENCE_CHECKPOINT ORINFER_REFERENCE_SOURCE ORINFER_REFERENCE_ACTIVATIONS ORINFER_EXECUTION_CACHE; do
    if [[ -n "${!variable:-}" ]]; then
        input_path="$(realpath -- "${!variable}")"
        if [[ -z "${mounted_inputs[$input_path]+present}" ]]; then
            extra_mounts+=(-v "$input_path:$input_path:ro")
            mounted_inputs["$input_path"]=1
        fi
        reference_env+=(-e "$variable=$input_path")
    fi
done
entrypoint="${ORINFER_COMPILER_PYTHON:-/usr/bin/python3}"
command_args=("$runner")
if [[ -n "${ORINFER_OPERATOR_SANITIZER:-}" ]]; then
    entrypoint="/usr/local/cuda/bin/compute-sanitizer"
    command_args=(--tool "$ORINFER_OPERATOR_SANITIZER" --error-exitcode 99)
    if [[ -n "${ORINFER_OPERATOR_SANITIZER_KERNEL_NAME:-}" ]]; then
        # Compute Sanitizer accepts an explicit kernel filter as one argument;
        # never interpret it as shell flags. Preserve the inspection scope.
        command_args+=(--kernel-name "$ORINFER_OPERATOR_SANITIZER_KERNEL_NAME")
    fi
    command_args+=(python3 "$runner")
    printf '%q ' "${command_args[@]}" > "$output_dir/sanitizer-command.txt"
    printf '\n' >> "$output_dir/sanitizer-command.txt"
fi
if [[ -n "${ORINFER_OPERATOR_NCU:-}" ]]; then
    if [[ -n "${ORINFER_OPERATOR_SANITIZER:-}" ]]; then
        echo 'Cannot combine Nsight Compute and Compute Sanitizer' >&2
        exit 2
    fi
    entrypoint="/usr/local/cuda/bin/ncu"
    # Keep clocks and caches untouched. Explicit candidate symbols prevent
    # accidentally profiling a same-run reference instead of the candidate.
    command_args=(--clock-control none --cache-control none --set detailed
        --kernel-name-base function --kernel-name "${ORINFER_OPERATOR_NCU_KERNEL_NAME:-regex:^kernel_kernel$}"
        --launch-count 1 --export "$output_dir/ncu" python3 "$runner")
    printf '%q ' "${command_args[@]}" > "$output_dir/ncu-command.txt"
    printf '\n' >> "$output_dir/ncu-command.txt"
fi
if [[ -n "${ORINFER_OPERATOR_NSYS:-}" ]]; then
    if [[ -n "${ORINFER_OPERATOR_NCU:-}${ORINFER_OPERATOR_SANITIZER:-}" ]]; then
        echo 'Cannot combine Nsight Systems with another GPU profiler' >&2
        exit 2
    fi
    entrypoint="$(realpath -- "$(command -v nsys)")"
    nsys_directory="$(dirname -- "$(dirname -- "$entrypoint")")"
    if [[ ! -x "$nsys_directory/host-linux-armv8/QdstrmImporter" ]]; then
        echo 'Jetson Nsight Systems installation must include QdstrmImporter' >&2
        exit 2
    fi
    extra_mounts+=(-v "$nsys_directory:$nsys_directory:ro")
    command_args=(profile --trace cuda,nvtx --sample none --cpuctxsw none
        --capture-range cudaProfilerApi --capture-range-end stop
        --cuda-graph-trace node --output "$output_dir/nsys" python3 "$runner")
    printf '%q ' "${command_args[@]}" > "$output_dir/nsys-command.txt"
    printf '\n' >> "$output_dir/nsys-command.txt"
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
    --user "$(id -u):$(id -g)" "${group_args[@]}" \
    --entrypoint "$entrypoint" --shm-size 2g -v "$repo_dir:$repo_dir" -v "$output_dir:$output_dir" \
    "${extra_mounts[@]}" "${reference_env[@]}" -w "$repo_dir" -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
    -e PYTHONPATH="$repo_dir:$repo_dir/tools/operators:${ORINFER_COMPILER_SITE_PACKAGES:-/opt/venv/lib/python3.10/site-packages}" \
    -e TORCH_EXTENSIONS_DIR="$output_dir/cache/torch" -e XDG_CACHE_HOME="$output_dir/cache" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e ORINFER_OPERATOR_OUTPUT="$output_dir" -e TILELANG_CACHE_DIR="$output_dir/cache" \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
    "${ORINFER_OPERATOR_IMAGE:-orinfer-compiler:0.1.1}" "${command_args[@]}" \
    --output "$output_dir" "$@" 2>&1 | tee "$output_dir/run.log"
if [[ -n "${ORINFER_OPERATOR_NSYS:-}" && ! -f "$output_dir/nsys.nsys-rep" ]]; then
    echo 'Nsight Systems did not export a report; any raw stream remains in the output directory' >&2
    exit 1
fi
