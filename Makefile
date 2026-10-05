.PHONY: check build info python-env compiler-image check-offline check-gpu

PYTHON ?= .venv/bin/python

python-env:
	uv venv --allow-existing --python python3.10 .venv
	uv pip sync --python $(PYTHON) tools/build/cpu-requirements.txt

build:
	cargo build --release --locked --offline

check:
	cargo fmt --all -- --check
	cargo check --workspace --offline
	cargo clippy --workspace --all-targets --offline -- -D warnings
	cargo test --workspace --offline
	$(PYTHON) -m unittest discover -s tools/eval -p 'test_*.py' -v
	$(PYTHON) -m unittest discover -s tools/bench -p 'test_*.py' -v
	$(PYTHON) -m unittest discover -s tools/operators -p 'test_abi.py' -v
	$(PYTHON) -m unittest discover -s tools/vision -p 'test_*.py' -v
	$(PYTHON) -m unittest tools.model.test_prepare tools.model.test_package tools.model.test_publication tools.model.test_resize_context tools.model.test_optimize_kv tools.model.test_stage_kv_prefill tools.model.test_upgrade_batching tools.model.test_upgrade_dynamic_batch tools.model.test_checkpoint tools.release.test_package -v
	$(PYTHON) tools/bench/validate_plan.py

info:
	cargo run --offline -p orin-cli -- info

compiler-image:
	docker build -t orin-llm-compiler:0.1.0 -f tools/build/compiler.Dockerfile .

check-offline:
	docker run --rm --runtime runc -v "$(CURDIR):$(CURDIR):ro" -w "$(CURDIR)" -e PYTHONPATH="$(CURDIR)" --entrypoint python3 orin-llm-compiler:0.1.0 -m unittest tools.model.test_mtp_weights -v

check-gpu:
	$(PYTHON) tools/bench/check_gpu.py --model "$(MODEL)" --output "$(OUTPUT)"
