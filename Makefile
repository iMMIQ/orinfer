.PHONY: check fmt lint build info python-env compiler-image check-offline check-gpu

PYTHON ?= .venv/bin/python

python-env:
	uv venv --allow-existing --python python3.10 .venv
	uv pip sync --python $(PYTHON) tools/build/cpu-requirements.txt

build:
	cargo build --release --locked --offline

fmt:
	cargo fmt --all
	$(PYTHON) -m ruff format tools kernels

lint:
	cargo fmt --all -- --check
	$(PYTHON) -m ruff format --check tools kernels
	$(PYTHON) -m ruff check tools kernels
	cargo clippy --workspace --all-targets --locked --offline -- -D warnings

check: lint
	cargo check --workspace --locked --offline
	cargo test --workspace --locked --offline
	$(PYTHON) -m unittest discover -s tools/eval -p 'test_*.py' -v
	$(PYTHON) -m unittest discover -s tools/bench -p 'test_*.py' -v
	$(PYTHON) -m unittest discover -s tools/operators -p 'test_abi.py' -v
	$(PYTHON) -m unittest discover -s tools/vision -p 'test_*.py' -v
	$(PYTHON) -m unittest discover -s tools/quantization -p 'test_*.py' -v
	$(PYTHON) -m unittest tools.model.test_safetensors_source -v
	$(PYTHON) -m unittest tools.model.test_compact -v
	$(PYTHON) -m unittest tools.model.flash_next.tests.test_decode_package -v
	$(PYTHON) -m unittest tools.model.flash_next.tests.test_weights -v
	$(PYTHON) -m unittest tools.model.flash_next.tests.test_ple tools.model.flash_next.tests.test_chunks tools.model.flash_next.tests.test_validation tools.model.flash_next.tests.test_original tools.model.flash_next.tests.test_teacher_inputs tools.model.flash_next.tests.test_speculation tools.model.flash_next.tests.test_config -v
	$(PYTHON) -m unittest tools.model.test_prepare tools.model.test_attach_execution tools.model.test_package tools.model.test_publication tools.model.test_resize_context tools.model.test_optimize_kv tools.model.test_stage_kv_prefill tools.model.test_upgrade_batching tools.model.test_upgrade_dynamic_batch tools.model.test_checkpoint tools.release.test_package tools.test_reference -v
	$(PYTHON) tools/bench/validate_plan.py

info:
	cargo run --offline -p orinfer-cli -- info

compiler-image:
	docker build -t orinfer-compiler:0.1.1 -f tools/build/compiler.Dockerfile .

check-offline:
	docker run --rm --runtime runc --network none -v "$(CURDIR):$(CURDIR):ro" -w "$(CURDIR)" -e PYTHONPATH="$(CURDIR):$(CURDIR)/tools/operators" --entrypoint python3 orinfer-compiler:0.1.1 -m tools.build.check_imports
	docker run --rm --runtime runc --network none -v "$(CURDIR):$(CURDIR):ro" -w "$(CURDIR)" -e PYTHONPATH="$(CURDIR)" --entrypoint python3 orinfer-compiler:0.1.1 -m unittest tools.model.test_mtp_weights tools.model.flash_next.tests.test_scenes tools.model.flash_next.tests.test_reference_math tools.model.flash_next.tests.test_teacher tools.model.flash_next.tests.test_qsa_reference tools.model.flash_next.tests.test_native_contract tools.model.flash_next.tests.test_dynamic_batch -v

check-gpu:
	$(PYTHON) tools/bench/check_gpu.py --model "$(MODEL)" --output "$(OUTPUT)"
