#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
resume=false
if [[ "${1:-}" == --resume ]]; then resume=true; shift; fi
if [[ $# -lt 2 ]]; then
    echo "usage: run.sh [--resume] runner.py output-directory [runner args]" >&2
    exit 2
fi
cd "$repo_dir"
runner="$(realpath --relative-to="$repo_dir" -- "$1")"
output_arg="$2"
shift 2
if [[ "$runner" == ../* || "$runner" != *.py || ! -f "$runner" ]]; then
    echo 'Runner must be a Python module within the repository.' >&2
    exit 2
fi
module="${runner#python/}"
module="${module%.py}"
module="${module//\//.}"
if [[ "$output_arg" = /* ]]; then output_dir="$output_arg"; else output_dir="$repo_dir/$output_arg"; fi
if [[ -e "$output_dir" && "$resume" != true ]]; then echo "Refusing existing output: $output_dir (use --resume to reuse an offline build)" >&2; exit 2; fi
uv sync --locked --project "$repo_dir"
uv sync --locked --project "$repo_dir/tools/environments/cuda"
cpu=(uv run --locked --no-sync --project "$repo_dir" python)
cuda=(uv run --locked --no-sync --project "$repo_dir/tools/environments/cuda")
mkdir -p "$output_dir/measurement-source"
cp --parents "$runner" python/orinfer_build/compiler/cuda/common.py tools/environments/cuda/run.sh \
    pyproject.toml uv.lock tools/environments/cuda/pyproject.toml tools/environments/cuda/uv.lock \
    "$output_dir/measurement-source/"
kernel_file="kernels/cuda/operators/$(basename -- "$runner")"
if [[ -f "$kernel_file" ]]; then cp --parents "$kernel_file" "$output_dir/measurement-source/"; fi
gpu_lock=$(python3 "$repo_dir/python/orinfer_build/common/paths.py" --root "$repo_dir" --gpu-lock)
mkdir -p "$(dirname -- "$gpu_lock")"
exec 9>"$gpu_lock"
flock 9
"${cpu[@]}" -m tools.bench.sample_machine --output "$output_dir/machine-before.json"
printf '%s\n' "compile-and-test" > "$output_dir/phase.txt"
command_args=(python -m "$module")
if [[ -n "${ORINFER_OPERATOR_SANITIZER:-}" ]]; then
    command_args=(compute-sanitizer --tool "$ORINFER_OPERATOR_SANITIZER" --error-exitcode 99)
    if [[ -n "${ORINFER_OPERATOR_SANITIZER_KERNEL_NAME:-}" ]]; then
        command_args+=(--kernel-name "$ORINFER_OPERATOR_SANITIZER_KERNEL_NAME")
    fi
    command_args+=(python -m "$module")
    printf '%q ' "${command_args[@]}" > "$output_dir/sanitizer-command.txt"
    printf '\n' >> "$output_dir/sanitizer-command.txt"
fi
if [[ -n "${ORINFER_OPERATOR_NCU:-}" ]]; then
    if [[ -n "${ORINFER_OPERATOR_SANITIZER:-}" ]]; then
        echo 'Cannot combine Nsight Compute and Compute Sanitizer' >&2
        exit 2
    fi
    command_args=(ncu --clock-control none --cache-control none --set detailed
        --kernel-name-base function --kernel-name "${ORINFER_OPERATOR_NCU_KERNEL_NAME:-regex:^kernel_kernel$}"
        --launch-count 1 --export "$output_dir/ncu" python -m "$module")
    printf '%q ' "${command_args[@]}" > "$output_dir/ncu-command.txt"
    printf '\n' >> "$output_dir/ncu-command.txt"
fi
if [[ -n "${ORINFER_OPERATOR_NSYS:-}" ]]; then
    if [[ -n "${ORINFER_OPERATOR_NCU:-}${ORINFER_OPERATOR_SANITIZER:-}" ]]; then
        echo 'Cannot combine Nsight Systems with another GPU profiler' >&2
        exit 2
    fi
    command_args=(nsys profile --trace cuda,nvtx --sample none --cpuctxsw none
        --capture-range cudaProfilerApi --capture-range-end stop
        --cuda-graph-trace node --output "$output_dir/nsys" python -m "$module")
    printf '%q ' "${command_args[@]}" > "$output_dir/nsys-command.txt"
    printf '\n' >> "$output_dir/nsys-command.txt"
fi
sampler_pid=""
worker_pid=""
log_pid=""
cleanup() {
    if [[ -n "$worker_pid" ]]; then
        kill -TERM -- "-$worker_pid" 2>/dev/null || true
        for _attempt in {1..20}; do
            if ! kill -0 -- "-$worker_pid" 2>/dev/null; then break; fi
            sleep 0.1
        done
        kill -KILL -- "-$worker_pid" 2>/dev/null || true
        wait "$worker_pid" 2>/dev/null || true
    fi
    touch "$output_dir/sampler.stop"
    if [[ -n "$sampler_pid" ]]; then wait "$sampler_pid" || true; fi
    if [[ -n "$log_pid" ]]; then wait "$log_pid" || true; fi
    "${cpu[@]}" -m tools.bench.sample_machine --output "$output_dir/machine-after.json" || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
rm -f "$output_dir/sampler.stop"
"${cpu[@]}" -m tools.bench.sample_continuous --output "$output_dir/machine.jsonl" \
    --phase "$output_dir/phase.txt" --stop "$output_dir/sampler.stop" &
sampler_pid=$!
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="$output_dir/cache/torch" XDG_CACHE_HOME="$output_dir/cache"
export ORINFER_OPERATOR_OUTPUT="$output_dir"
export TILELANG_CACHE_DIR="${ORINFER_TILELANG_CACHE_DIR:-$output_dir/cache}"
exec {log_fd}> >(tee "$output_dir/run.log")
log_pid=$!
setsid "${cuda[@]}" "${command_args[@]}" --output "$output_dir" "$@" >&"$log_fd" 2>&1 &
worker_pid=$!
exec {log_fd}>&-
status=0
wait "$worker_pid" || status=$?
wait "$log_pid" || true
log_pid=""
if [[ $status != 0 ]]; then exit "$status"; fi
if [[ -n "${ORINFER_OPERATOR_NSYS:-}" && ! -f "$output_dir/nsys.nsys-rep" ]]; then
    echo 'Nsight Systems did not export a report; any raw stream remains in the output directory' >&2
    exit 1
fi
