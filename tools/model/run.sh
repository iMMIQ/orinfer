#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# != 3 ]]; then echo 'usage: run.sh MODEL_DIR REQUESTS.json NEW_OUTPUT_DIR' >&2; exit 2; fi
model_manifest="$1"
request_manifest="$2"
output_arg="$3"
cd "$repo_dir"
if [[ -d "$model_manifest" ]]; then model_manifest="$model_manifest/cache/model.json"; fi
if [[ "$output_arg" = /* ]]; then output_dir="$output_arg"; else output_dir="$repo_dir/$output_arg"; fi
if [[ -e "$output_dir" ]]; then echo "Refusing existing output $output_dir" >&2; exit 2; fi
mkdir -p "$output_dir/measurement-source"
cp --parents "$model_manifest" "$(dirname -- "$model_manifest")/weights/model.safetensors.index.json" "$request_manifest" tools/model/run.sh crates/orinfer-engine/src/{model,artifact,weights,loader,operators}.rs crates/orinfer-cli/src/main.rs "$output_dir/measurement-source/"
rustc -Vv > "$output_dir/rustc.txt"
cp --parents Cargo.lock Cargo.toml crates/orinfer-engine/Cargo.toml crates/orinfer-engine/src/lib.rs "$output_dir/measurement-source/"
cp -r --parents crates/orinfer-engine/src/{cuda,runtime,architecture} "$output_dir/measurement-source/"
cp target/release/orinfer "$output_dir/orinfer"
"$output_dir/orinfer" plan-model "$model_manifest" > "$output_dir/plan.json"
sha256sum "$output_dir/orinfer" > "$output_dir/binary.sha256"
exec 9>"$repo_dir/artifacts/gpu-experiment.lock"
flock 9
python3 tools/bench/sample_machine.py --output "$output_dir/machine-before.json"
printf '%s\n' 'rust-model-load-and-inference' > "$output_dir/phase.txt"
python3 tools/bench/sample_continuous.py --output "$output_dir/machine.jsonl" --phase "$output_dir/phase.txt" --stop "$output_dir/sampler.stop" &
sampler_pid=$!
cleanup() {
    touch "$output_dir/sampler.stop"
    wait "$sampler_pid" || true
    python3 tools/bench/sample_machine.py --output "$output_dir/machine-after.json" || true
}
trap cleanup EXIT
ORINFER_MODEL_PROGRESS="$output_dir/progress.json" "$output_dir/orinfer" run-model "$model_manifest" "$request_manifest" > "$output_dir/report.json" 2> "$output_dir/run.log"
