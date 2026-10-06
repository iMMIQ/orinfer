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
	$(PYTHON) -m unittest discover -s tools/quantization -p 'test_*.py' -v
	$(PYTHON) -m unittest tools.model.test_safetensors_source -v
	$(PYTHON) -m unittest tools.model.test_flash_weights -v
	$(PYTHON) -m unittest tools.model.test_flash_ple tools.model.test_flash_chunks tools.model.test_flash_validation tools.model.test_flash_original tools.model.test_flash_teacher_inputs tools.model.test_flash_speculation -v
	$(PYTHON) -m unittest tools.model.test_prepare tools.model.test_package tools.model.test_publication tools.model.test_resize_context tools.model.test_optimize_kv tools.model.test_stage_kv_prefill tools.model.test_upgrade_batching tools.model.test_upgrade_dynamic_batch tools.model.test_checkpoint tools.model.test_gguf tools.release.test_package tools.test_reference -v
	$(PYTHON) tools/bench/validate_plan.py

info:
	cargo run --offline -p orinfer-cli -- info

compiler-image:
	docker build -t orinfer-compiler:0.1.1 -f tools/build/compiler.Dockerfile .

check-offline:
	docker run --rm --runtime runc --network none -v "$(CURDIR):$(CURDIR):ro" -w "$(CURDIR)" -e PYTHONPATH="$(CURDIR)" --entrypoint python3 orinfer-compiler:0.1.1 -m unittest tools.model.test_mtp_weights tools.model.test_flash_scenes tools.model.test_flash_reference_math tools.model.test_flash_teacher tools.model.test_flash_qsa_reference -v

check-gpu:
	$(PYTHON) tools/bench/check_gpu.py --model "$(MODEL)" --output "$(OUTPUT)"
