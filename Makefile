# Makefile for aivo CLI
# Quick commands for development

.PHONY: build build-debug build-release test test-release npm-test check clippy clean install fmt dev release

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "aivo CLI - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build debug binary (~1s incremental)
	cargo build

build-release: ## Build optimized release binary
	cargo build --release

# Tests use a separate CARGO_TARGET_DIR so `__internal_test_fast_crypto` (reduced PBKDF2
# iterations for fast tests) never contaminates `target/debug/aivo`. Without
# this, `make test && ./target/debug/aivo keys cat <name>` would fail because
# keys encrypted by the normal build can't be decrypted by the test binary.
test: ## Run all tests (isolated target dir; won't clobber target/debug/aivo)
	CARGO_TARGET_DIR=target/test cargo test --features __internal_test_fast_crypto

npm-test: ## Run npm wrapper tests
	npm test --prefix npm

test-release: ## Run tests on release build (isolated target dir)
	CARGO_TARGET_DIR=target/test cargo test --release --features __internal_test_fast_crypto

check: ## Quick type check
	cargo check

clippy: ## Run clippy linter
	cargo clippy -- -D warnings

fmt: ## Format code
	cargo fmt

clean: ## Clean build artifacts
	cargo clean

install: build ## Install debug binary to /usr/local/bin (re-signs for macOS arm64)
	cp target/debug/aivo /usr/local/bin/aivo
	codesign --force -s - /usr/local/bin/aivo 2>/dev/null || true

dev: check test clippy npm-test ## Run all checks (check, test, clippy, npm-test)

release: test clippy npm-test build ## Full release workflow (test, lint, npm-test, build)
	@echo "Release binary ready at: target/release/aivo"
	@ls -lh target/release/aivo | awk '{print "Size:", $$5}'
