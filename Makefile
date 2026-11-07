.PHONY: check check-locks fmt lint build info python-env cuda-env check-offline check-ascend-offline check-gpu

UV ?= uv
PYTHON = $(UV) run --locked python
CUDA_PYTHON = $(UV) run --locked --project tools/environments/cuda python

python-env:
	$(UV) sync --locked

cuda-env:
	$(UV) sync --locked --project tools/environments/cuda

check-locks:
	$(UV) lock --check --offline
	$(UV) lock --check --offline --project tools/environments/cuda

build:
	cargo build --release --locked --offline

fmt:
	cargo fmt --all
	$(PYTHON) -m ruff format tools kernels python tests setup.py

lint:
	cargo fmt --all -- --check
	$(PYTHON) -m ruff format --check tools kernels python tests setup.py
	$(PYTHON) -m ruff check tools kernels python tests setup.py
	cargo clippy --workspace --all-targets --locked --offline -- -D warnings

check: check-locks lint
	cargo check --workspace --locked --offline
	cargo test --workspace --locked --offline
	$(PYTHON) -m orinfer_build.common.testing --environment cpu
	$(PYTHON) tools/bench/validate_plan.py

info:
	cargo run --offline -p orinfer-cli -- info

check-offline:
	$(CUDA_PYTHON) -m orinfer_build.common.check_imports --backend cuda
	$(CUDA_PYTHON) -m orinfer_build.common.testing --environment cuda

check-ascend-offline:
	bash tools/environments/ascend/run.sh --without-devices python3 -m orinfer_build.common.check_imports --backend ascend

check-gpu:
	$(PYTHON) tools/bench/check_gpu.py --model "$(MODEL)" --output "$(OUTPUT)"
