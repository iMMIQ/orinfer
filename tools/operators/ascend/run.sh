#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
exec bash "$repo_dir/tools/environments/ascend/run.sh" "$@"
