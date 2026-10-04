.PHONY: check build info

build:
	cargo build --release --offline

check:
	cargo fmt --all -- --check
	cargo check --workspace --offline
	cargo clippy --workspace --all-targets --offline -- -D warnings
	cargo test --workspace --offline
	python3 -m unittest discover -s tools/eval -p 'test_*.py' -v
	python3 -m unittest discover -s tools/bench -p 'test_*.py' -v
	python3 -m unittest discover -s tools/operators -p 'test_abi.py' -v
	python3 -m unittest discover -s tools/vision -p 'test_*.py' -v
	python3 -m unittest tools.model.test_prepare tools.model.test_package tools.model.test_resize_context tools.model.test_optimize_kv tools.model.test_stage_kv_prefill tools.model.test_upgrade_batching -v
	python3 tools/bench/validate_plan.py

info:
	cargo run --offline -p orin-cli -- info
