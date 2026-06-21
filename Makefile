# Makefile — single source of truth for all validation commands.
# Used by CI and local development. Do NOT bypass with raw cargo commands
# unless doing microscopic troubleshooting on a specific issue.
#
# Usage:
#   make check      — fast local validation (lint + compile + test)
#   make ci         — core CI validation shared with GitHub Actions
#   make bench      — criterion benchmarks
#   make bench-vs   — E2E comparison vs libvips (representative matrix)
#   make bench-vs BENCH_ITER=10  — quick smoke test
#   make cross-up        — start persistent x86 container (once)
#   make cross CMD=clippy — run any target in opposite arch (instant via exec)
#   make check-cross     — shorthand for make cross CMD=check
#   make cross-setup     — one-time: Colima with Rosetta (FAST x86)
#   make cross-shell     — interactive shell in cross container
#   make cross-clean     — nuke cross environment

CARGO := cargo
RUSTFLAGS_CI := -Dwarnings

# --all-features is the default. The build.rs in viprs-codecs emits compile_error!
# if a required system library is missing, so this is safe on any machine.
# Override: make check FEATURES="--features jpeg,png"
FEATURES ?= --all-features
COVERAGE_PACKAGES := -p viprs-codecs -p viprs-ops-pixel -p viprs-ops-colour -p viprs-ops-spatial -p viprs-ops-composite

.PHONY: all check ci cross cross-up cross-down cross-clean cross-setup cross-shell check-cross fmt clippy test test-xtask doc security deny audit require-cargo-audit require-cargo-deny fixtures fixtures-functional fixtures-bench fixtures-clean fixtures-pack fixtures-pack-functional fixtures-pack-bench bench bench-ci-full bench-baseline-full bench-compare-full bench-vs coverage xtask

# ─── Developer targets ─────────────────────────────────────────────────────────

## Fast local validation: format + lint + tests
check: fmt clippy test

## Core CI validation: format + lint + compile + test + docs + audits
ci: fmt clippy test doc security

# ─── Individual targets ────────────────────────────────────────────────────────

## Check formatting (auto-fix with `make fmt FIX=1`)
fmt:
ifdef FIX
	$(CARGO) fmt --all
else
	$(CARGO) fmt --all -- --check
endif

## Clippy: enforce perf + unwrap/expect ban + pedantic + nursery (via Cargo.toml [lints.clippy]).
## -Dwarnings ensures no warning passes silently. Also compiles all lib targets.
clippy:
	RUSTFLAGS="-Dwarnings" $(CARGO) clippy --workspace --lib $(FEATURES) -- \
		-D clippy::perf -D clippy::unwrap_used -D clippy::expect_used
	$(CARGO) check -p xtask

## Download or refresh functional fixtures if the manifest changed.
fixtures: fixtures-functional

## Fixtures required by local tests and coverage.
fixtures-functional:
	./tools/fixtures.sh functional

## Large fixtures required only by benchmarks and viprs/libvips comparisons.
fixtures-bench: fixtures-functional
	./tools/fixtures.sh bench

## Remove downloaded fixture archives and manifest markers. Extracted files stay in place.
fixtures-clean:
	rm -rf .artifacts/fixtures tests/fixtures/.fixtures-*.manifest-sha256

## Build release fixture archives without macOS xattrs. Does not publish assets.
fixtures-pack:
	./tools/package-fixtures.sh all

## Build only the functional fixture archive without macOS xattrs.
fixtures-pack-functional:
	./tools/package-fixtures.sh functional

## Build only the benchmark fixture archive without macOS xattrs.
fixtures-pack-bench:
	./tools/package-fixtures.sh bench

## All tests: unit + integration + doctests
test: fixtures-functional
	$(CARGO) test --workspace --lib $(FEATURES)
	$(CARGO) test --workspace --tests --exclude xtask $(FEATURES)
	# xtask installs a process-global counting allocator, so exact-counter tests
	# must not overlap with unrelated allocations from other xtask tests.
	$(CARGO) test -p xtask --tests -- --test-threads=1
	$(CARGO) test --workspace --doc $(FEATURES)

## xtask tests only. Coverage runs the library workspace tests separately.
test-xtask: fixtures-bench
	# xtask installs a process-global counting allocator, so exact-counter tests
	# must not overlap with unrelated allocations from other xtask tests.
	$(CARGO) test -p xtask --tests -- --test-threads=1

## Documentation (deny warnings)
doc:
	RUSTDOCFLAGS="-Dwarnings" $(CARGO) doc --workspace --no-deps $(FEATURES) --exclude xtask

## Security and policy checks run by CI.
security: audit deny

require-cargo-deny:
	@command -v cargo-deny >/dev/null 2>&1 || { \
		echo "cargo-deny is required. Install it outside Makefile before running this target."; \
		exit 127; \
	}

require-cargo-audit:
	@command -v cargo-audit >/dev/null 2>&1 || { \
		echo "cargo-audit is required. Install it outside Makefile before running this target."; \
		exit 127; \
	}

## License/advisory audit
deny: require-cargo-deny
	$(CARGO) deny check

## Security vulnerability audit
audit: require-cargo-audit
	$(CARGO) audit

## Threshold: ≥90% line coverage for ops/codecs, using all workspace tests except xtask.
## Some codec tests embed benchmark fixtures at compile time with include_bytes!.
coverage: fixtures-bench
	$(CARGO) llvm-cov --workspace --exclude xtask $(FEATURES) --ignore-filename-regex '(benches|tests|crates/viprs-core/|crates/viprs-ports/|crates/viprs-runtime/|/viprs/src/)' --fail-under-lines 90

## Build xtask release (for benchmark runner — native CPU for fair comparison)
xtask:
	RUSTFLAGS="-Ctarget-cpu=native" $(CARGO) build --release -p xtask --features count-alloc

# ─── Cross-architecture local CI (Docker) ───────────────────────────────────────
# Persistent container with native filesystem for source code.
# Architecture: container runs indefinitely, source is rsynced (tar pipe) on each
# `make cross` invocation. This gives native ext4 I/O speed instead of virtiofs
# overhead (~7x faster than bind-mount for many small files like .rs sources).
#
# Layout inside container:
#   /src/          ← synced source (native ext4, fast)
#   /src/target/   ← volume mount (persists builds across runs)
#   /workspace/    ← bind mount from host (only used for sync source)
#
# Usage:
#   make cross-up                       — start persistent container (once)
#   make cross CMD=check                — run command inside container
#   make cross CMD=bench                — benchmarks in x86
#   make check-cross                    — shorthand for make cross CMD=check
#   make cross-down                     — stop container (keeps state)
#   make cross-clean                    — nuke container + volumes (full reset)
#   make cross-shell                    — interactive shell in container
#   make cross-setup                    — one-time Colima setup with Rosetta

# Cross targets use the same feature coverage as CI. The CI image provides every
# native library required by `--all-features`.
CROSS_FEATURES := --all-features

CI_IMAGE := ghcr.io/mbertogliati/viprs-ci:latest

# Auto-detect: if host is arm64, test x86_64 and vice versa
HOST_ARCH := $(shell uname -m)
ifeq ($(HOST_ARCH),arm64)
  ARCH ?= x86_64
else ifeq ($(HOST_ARCH),aarch64)
  ARCH ?= x86_64
else
  ARCH ?= arm64
endif

# Map arch names to Docker platform strings
ifeq ($(ARCH),x86_64)
  DOCKER_PLATFORM := linux/amd64
else ifeq ($(ARCH),amd64)
  DOCKER_PLATFORM := linux/amd64
else
  DOCKER_PLATFORM := linux/arm64
endif

CROSS_CONTAINER := viprs-cross-$(ARCH)
CROSS_TARGET_VOL := viprs-cross-target-$(ARCH)
CROSS_REGISTRY_VOL := viprs-cross-registry
CMD ?= check

## Start persistent cross-compilation container (idempotent).
cross-up:
	@if docker inspect $(CROSS_CONTAINER) >/dev/null 2>&1; then \
		if [ "$$(docker inspect -f '{{.State.Running}}' $(CROSS_CONTAINER))" = "true" ]; then \
			echo "✓ $(CROSS_CONTAINER) already running"; \
		else \
			echo "↻ restarting $(CROSS_CONTAINER)..."; \
			docker start $(CROSS_CONTAINER); \
		fi; \
	else \
		echo "🚀 creating $(CROSS_CONTAINER) ($(DOCKER_PLATFORM))..."; \
		docker run -d --platform $(DOCKER_PLATFORM) \
			--name $(CROSS_CONTAINER) \
			--entrypoint "" \
			-v $(CROSS_TARGET_VOL):/src/target \
			-v $(CROSS_REGISTRY_VOL):/root/.cargo/registry \
			-v "$$(pwd)/tests/fixtures:/src/tests/fixtures:ro" \
			$(CI_IMAGE) \
			sleep infinity; \
		docker exec $(CROSS_CONTAINER) mkdir -p /src; \
		echo "✅ $(CROSS_CONTAINER) running"; \
	fi

## Sync source into container (tar pipe — fast, excludes heavy fixtures).
## Only code, configs, and build files are synced (~30MB vs 775MB full repo).
## COPYFILE_DISABLE=1 prevents macOS AppleDouble (._*) resource fork files.
## We exclude the real tests/fixtures/ dir (mounted via volume) but recreate
## the crate-level symlinks inside the container after sync.
cross-sync: cross-up
	@COPYFILE_DISABLE=1 tar -cf - \
		--exclude='target' \
		--exclude='.git' \
		--exclude='.libvips_repo' \
		--exclude='.worktrees' \
		--exclude='tests/fixtures' \
		. 2>/dev/null | docker exec -i $(CROSS_CONTAINER) tar -xf - -C /src 2>/dev/null
	@docker exec -w /src $(CROSS_CONTAINER) sh -c '\
		for d in crates/*/; do \
			if [ -d "$$d" ]; then \
				mkdir -p "$${d}tests"; \
				ln -sfn ../../../tests/fixtures "$${d}tests/fixtures"; \
			fi; \
		done'

## Run any make target inside the persistent container.
## Syncs source first, then executes in /src (native filesystem).
## Passes CROSS_FEATURES so cross checks match CI feature coverage.
cross: cross-sync
	@echo "── cross: make $(CMD) [$(ARCH)] ──"
	docker exec -w /src $(CROSS_CONTAINER) make $(CMD) FEATURES="$(CROSS_FEATURES)"

## Shorthand: make check-cross = make cross CMD=check
check-cross: CMD = check
check-cross: cross

## Interactive shell inside the cross container (working dir: /src)
cross-shell: cross-sync
	docker exec -it -w /src $(CROSS_CONTAINER) bash

## Stop the persistent container (keeps it for quick restart)
cross-down:
	@docker stop $(CROSS_CONTAINER) 2>/dev/null && echo "stopped $(CROSS_CONTAINER)" || echo "not running"

## Full reset: remove container + volumes (next cross-up starts fresh)
cross-clean:
	docker rm -f $(CROSS_CONTAINER) 2>/dev/null || true
	docker volume rm -f $(CROSS_TARGET_VOL) $(CROSS_REGISTRY_VOL) 2>/dev/null || true
	@echo "cross environment destroyed"

## Setup Colima with Rosetta (x86 emulation at near-native speed) + adequate resources.
## Run once — destroys and recreates the default Colima profile.
cross-setup:
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  Recreating Colima with Rosetta + more resources"
	@echo "  (this will stop the current instance)"
	@echo "══════════════════════════════════════════════════════════════"
	colima stop -f 2>/dev/null || true
	colima delete -f 2>/dev/null || true
	colima start \
		--vm-type vz \
		--vz-rosetta \
		--cpu 10 \
		--memory 16 \
		--disk 100 \
		--arch aarch64
	@echo ""
	@echo "✅ Colima ready with Rosetta + 10 CPUs + 16GB RAM"
	@echo "   Run: make cross-up && make cross CMD=check"1

# ─── Benchmarks ────────────────────────────────────────────────────────────────

## Criterion micro-benchmarks — full suite (native CPU for fair comparison)
bench: fixtures-bench
	RUSTFLAGS="-Ctarget-cpu=native" $(CARGO) bench $(FEATURES) --bench '*'

## Fast CI benchmark gate (target ≤5 min per arch): representative base/head comparison.
## The full suite is available through bench-ci-full / bench-baseline-full / bench-compare-full.
##
## CI overrides: disable fat LTO + use 16 codegen-units + no debuginfo → ~5x faster compile.
## Benchmark results are still valid (same opt-level 3, same target-cpu=native).
BENCH_CI_ENV := RUSTFLAGS="-Ctarget-cpu=native" \
	CARGO_PROFILE_BENCH_LTO=off \
	CARGO_PROFILE_BENCH_CODEGEN_UNITS=16 \
	CARGO_PROFILE_BENCH_DEBUG=0

CRITERION_BENCHES := $(shell awk 'BEGIN { in_bench = 0; name = ""; path = "" } \
	/^\[\[bench\]\]/ { \
		if (in_bench && path !~ /^benches\/iai\// && name != "") print name; \
		in_bench = 1; name = ""; path = ""; next \
	} \
	in_bench && /^name = "/ { name = $$3; gsub(/"/, "", name); next } \
	in_bench && /^path = "/ { path = $$3; gsub(/"/, "", path); next } \
	END { if (in_bench && path !~ /^benches\/iai\// && name != "") print name }' Cargo.toml)

## Representative PR gate. Keep this intentionally small: CI runs it for base and head on both arches.
BENCH_CI_TARGETS ?= \
	pipeline \
	fusion \
	invert \
	add \
	sin \
	linear \
	gauss_blur \
	conv \
	resize \
	thumbnail \
	srgb_to_lab \
	hist_find

BENCH_CI_ARGS := --sample-size 10 --warm-up-time 0.5 --measurement-time 0.5 --nresamples 100
BENCH_FULL_CI_ARGS := --sample-size 10 --warm-up-time 1 --measurement-time 1 --nresamples 100
BENCH_CI_FILTER := '/512'

bench-ci: fixtures-bench
	@set -e; \
	for bench in $(BENCH_CI_TARGETS); do \
		echo "▶ Running $$bench"; \
		$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench "$$bench" -- \
			$(BENCH_CI_ARGS) $(BENCH_CI_FILTER); \
	done

bench-ci-full: fixtures-bench
	@set -e; \
	for bench in $(CRITERION_BENCHES); do \
		echo "▶ Running $$bench"; \
		$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench "$$bench" -- \
			$(BENCH_FULL_CI_ARGS) $(BENCH_CI_FILTER); \
	done

## Save baseline on main (CI calls this after merge to main)
bench-baseline: fixtures-bench
	@set -e; \
	for bench in $(BENCH_CI_TARGETS); do \
		echo "▶ Saving baseline for $$bench"; \
		$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench "$$bench" -- \
			$(BENCH_CI_ARGS) --save-baseline main $(BENCH_CI_FILTER); \
	done

bench-baseline-full: fixtures-bench
	@set -e; \
	for bench in $(CRITERION_BENCHES); do \
		echo "▶ Saving baseline for $$bench"; \
		$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench "$$bench" -- \
			$(BENCH_FULL_CI_ARGS) --save-baseline main $(BENCH_CI_FILTER); \
	done

## Compare PR against main baseline — fails if any benchmark regresses >5%
bench-compare: fixtures-bench
	@set -e; \
	for bench in $(BENCH_CI_TARGETS); do \
		echo "▶ Comparing $$bench"; \
		$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench "$$bench" -- \
			$(BENCH_CI_ARGS) --baseline main $(BENCH_CI_FILTER); \
	done

bench-compare-full: fixtures-bench
	@set -e; \
	for bench in $(CRITERION_BENCHES); do \
		echo "▶ Comparing $$bench"; \
		$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench "$$bench" -- \
			$(BENCH_FULL_CI_ARGS) --baseline main $(BENCH_CI_FILTER); \
	done

## E2E comparison vs libvips (requires xtask + libvips installed).
## Runs the representative scenario matrix from PERFORMANCE.md:
##   - 3 standard sizes (512, 2048, 8192)
##   - Real-world ops (thumbnail, sharpen, convolution pipeline)
##   - Enough iterations for stable p50/p95 (30+)
##
## Override: make bench-vs BENCH_ITER=10  (quick smoke test)
BENCH_ITER ?= 30
BENCH_IMG_512  := tests/fixtures/images/bench_512x512.jpg
BENCH_IMG_2048 := tests/fixtures/images/bench_2048x2048.jpg
BENCH_IMG_8192 := tests/fixtures/images/bench_8192x8192.jpg

bench-vs: fixtures-bench
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  bench-vs: viprs/libvips ratio (target ≤1.00 = viprs wins)"
	@echo "  iterations=$(BENCH_ITER) — increase for tighter confidence"
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "── thumbnail 400 (THE web use case: decode + resize + encode) ──"
	$(CARGO) xtask bench $(BENCH_IMG_512)  thumbnail 400 --iterations $(BENCH_ITER)
	$(CARGO) xtask bench $(BENCH_IMG_2048) thumbnail 400 --iterations $(BENCH_ITER)
	$(CARGO) xtask bench $(BENCH_IMG_8192) thumbnail 400 --iterations $(BENCH_ITER)
	@echo ""
	@echo "── sharpen (compute-bound, tests SIMD path) ──"
	$(CARGO) xtask bench $(BENCH_IMG_2048) sharpen 1.5 1.0 --iterations $(BENCH_ITER)
	$(CARGO) xtask bench $(BENCH_IMG_8192) sharpen 1.5 1.0 --iterations $(BENCH_ITER)
	@echo ""
	@echo "── gauss_blur σ=3 (convolution kernel, memory bandwidth) ──"
	$(CARGO) xtask bench $(BENCH_IMG_2048) gauss_blur 3.0 --iterations $(BENCH_ITER)
	$(CARGO) xtask bench $(BENCH_IMG_8192) gauss_blur 3.0 --iterations $(BENCH_ITER)
	@echo ""
	@echo "── thumbnail_sharpen (multi-op pipeline: resize→sharpen) ──"
	$(CARGO) xtask bench $(BENCH_IMG_2048) thumbnail_sharpen --iterations $(BENCH_ITER)
	$(CARGO) xtask bench $(BENCH_IMG_8192) thumbnail_sharpen --iterations $(BENCH_ITER)
	@echo ""
	@echo "── perceptual_enhance webp (full pipeline: thumbnail→colour→sharpen→encode) ──"
	$(CARGO) xtask bench $(BENCH_IMG_2048) perceptual_enhance webp --iterations $(BENCH_ITER)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  Done. Any ratio > 1.00 = performance gap → file issue before merge"
	@echo "══════════════════════════════════════════════════════════════════════"
