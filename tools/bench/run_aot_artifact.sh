#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo 'usage: run_aot_artifact.sh manifest.json new-output-directory' >&2
    exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
manifest_path="$(realpath -- "$1")"
if [[ "$2" = /* ]]; then output_dir="$2"; else output_dir="$repo_dir/$2"; fi
if [[ -e "$output_dir" ]]; then
    echo "Refusing existing output: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir/source"
# Freeze the CPU build before joining the shared GPU experiment queue.
cargo build --release --offline -p orinfer-cli
cp --parents Cargo.toml Cargo.lock crates/orinfer-engine/Cargo.toml \
    crates/orinfer-engine/src/*.rs crates/orinfer-cli/Cargo.toml crates/orinfer-cli/src/*.rs \
    tools/bench/run_aot_artifact.sh "$output_dir/source/"
cp target/release/orinfer "$output_dir/source/orinfer"
cp "$manifest_path" "$output_dir/manifest.json"
sha256sum "$output_dir/source/orinfer" "$manifest_path" > "$output_dir/identities.txt"
rustc --version --verbose > "$output_dir/rustc-version.txt"
printf '%s\n' "$manifest_path" > "$output_dir/fixture-path.txt"
exec 9>"$repo_dir/artifacts/gpu-experiment.lock"
flock 9
python3 tools/bench/sample_machine.py --output "$output_dir/machine-before.json"
printf '%s\n' 'aot-load-validate-capture-replay' > "$output_dir/phase.txt"
sampler_pid=''
cleanup() {
    touch "$output_dir/sampler.stop"
    if [[ -n "$sampler_pid" ]]; then wait "$sampler_pid" || true; fi
    python3 tools/bench/sample_machine.py --output "$output_dir/machine-after.json" || true
}
trap cleanup EXIT
python3 tools/bench/sample_continuous.py --output "$output_dir/machine.jsonl" \
    --phase "$output_dir/phase.txt" --stop "$output_dir/sampler.stop" &
sampler_pid=$!
"$output_dir/source/orinfer" run-artifact "$manifest_path" \
    2>"$output_dir/stderr.log" | tee "$output_dir/result.json"
