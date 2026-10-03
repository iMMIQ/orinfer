#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# != 3 ]]; then echo 'usage: profile.sh MODEL_DIR REQUESTS.json NEW_OUTPUT_DIR' >&2; exit 2; fi
model_manifest="$1"
request_manifest="$2"
output_arg="$3"
cd "$repo_dir"
if [[ -d "$model_manifest" ]]; then model_manifest="$model_manifest/cache/manifest.json"; fi
if [[ "$output_arg" = /* ]]; then output_dir="$output_arg"; else output_dir="$repo_dir/$output_arg"; fi
if [[ -e "$output_dir" ]]; then echo "Refusing existing output $output_dir" >&2; exit 2; fi
mkdir -p "$output_dir/measurement-source"
cp --parents "$model_manifest" "$(dirname -- "$model_manifest")/weights/model.safetensors.index.json" "$request_manifest" tools/model/{profile.sh,profile_summary.py} \
    crates/orin-engine/src/{cuda,model,artifact,weights}.rs crates/orin-cli/src/main.rs \
    "$output_dir/measurement-source/"
cp "$request_manifest" "$output_dir/requests.json"
cp target/release/orin-llm "$output_dir/orin-llm"
sha256sum "$model_manifest" "$request_manifest" "$output_dir/orin-llm" > "$output_dir/identities.sha256"
exec 9>"$repo_dir/artifacts/gpu-experiment.lock"
flock 9
python3 tools/bench/sample_machine.py --output "$output_dir/machine-before.json"
cleanup() {
    python3 tools/bench/sample_machine.py --output "$output_dir/machine-after.json" || true
}
trap cleanup EXIT
# Preserve target JSON/stderr separately from Nsight's own console output.
# Positional shell parameters keep filenames as data, not executable text.
ORIN_MODEL_PROGRESS="$output_dir/progress.json" nsys profile \
    --output "$output_dir/trace" --force-overwrite=false --trace=cuda \
    --cuda-graph-trace=node --sample=none --cpuctxsw=none \
    bash -c 'exec "$1" run-model "$2" "$3" >"$4" 2>"$5"' profile-target \
    "$output_dir/orin-llm" "$model_manifest" "$request_manifest" \
    "$output_dir/report.json" "$output_dir/run.log" > "$output_dir/nsys.log" 2>&1
nsys export --type=sqlite --output "$output_dir/trace.sqlite" \
    "$output_dir/trace.nsys-rep" > "$output_dir/export.log" 2>&1
python3 tools/model/profile_summary.py "$output_dir" "$model_manifest"
