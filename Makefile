# Makefile — single source of truth for all validation commands.
# Used by CI and local development. Do NOT bypass with raw cargo commands
# unless doing microscopic troubleshooting on a specific issue.
#
# Usage:
#   make check      — fast local validation (lint + compile + test)
#   make ci         — full CI pipeline (everything CI runs)
#   make bench      — criterion benchmarks
#   make bench-vs   — E2E comparison vs libvips (representative matrix)
#   make bench-vs BENCH_ITER=10  — quick smoke test
#   make check-cross — cross-arch validation via Docker (auto-detects opposite arch)

CARGO := cargo
RUSTFLAGS_CI := -Dwarnings

# Features that require system libraries (libjxl, libheif, libjpeg-turbo, etc.)
# CI runs inside a container with all deps; local devs enable what they have installed.
# Default features match CI (CONTAINER_FEATURES). Override locally if needed:
#   make check FEATURES="--all-features"
FEATURES ?= --features default,simd-pulp,rayon,jpeg,png,webp,tiff,heif,avif,gif,jp2k,fft,exr,lock_instrumentation

.PHONY: all check ci check-cross fmt clippy build test test-all doc deny audit bench bench-vs coverage xtask

# ─── Developer targets ─────────────────────────────────────────────────────────

## Fast local validation: format + lint + compile + unit tests
check: fmt clippy build test

## Full CI pipeline (mirrors .github/workflows/ci.yml exactly)
ci: fmt clippy build test doc deny audit

# ─── Individual targets ────────────────────────────────────────────────────────

## Check formatting (auto-fix with `make fmt FIX=1`)
fmt:
ifdef FIX
	$(CARGO) fmt --all
else
	$(CARGO) fmt --all -- --check
endif

## Clippy: enforce perf + unwrap/expect ban + pedantic + nursery (via Cargo.toml [lints.clippy]).
## -Dwarnings ensures no warning passes silently.
clippy:
	RUSTFLAGS="-Dwarnings" $(CARGO) clippy --lib $(FEATURES) -- \
		-D clippy::perf -D clippy::unwrap_used -D clippy::expect_used

## Compile check (lib mandatory; xtask checked here and in container CI job)
build:
	RUSTFLAGS="$(RUSTFLAGS_CI)" $(CARGO) check --lib $(FEATURES)
	$(CARGO) check -p xtask

## Unit tests only (containerless CI — no system libs)
test:
	$(CARGO) test --lib $(FEATURES)

## Full test suite: unit + doctests (requires system libs for all codec features)
## Uses the same feature set as xtask (all codecs that compile together without conflicts).
## Full test suite with all codec features (container).
CONTAINER_FEATURES := --features default,simd-pulp,rayon,jpeg,png,webp,tiff,heif,avif,gif,jp2k,fft,exr,lock_instrumentation

test-all:
	$(CARGO) test --lib $(CONTAINER_FEATURES)
	$(CARGO) test --tests $(CONTAINER_FEATURES)
	$(CARGO) test --doc $(CONTAINER_FEATURES)

## Documentation (deny warnings)
doc:
	RUSTDOCFLAGS="-Dwarnings" $(CARGO) doc --no-deps $(FEATURES)

## License/advisory audit
deny:
	$(CARGO) deny check

## Security vulnerability audit
audit:
	$(CARGO) audit

## Threshold: ≥87% line coverage (enforced).
coverage:
	$(CARGO) llvm-cov --lib $(CONTAINER_FEATURES) --ignore-filename-regex '(benches|tests)' --fail-under-lines 87

## Build xtask release (for benchmark runner — native CPU for fair comparison)
xtask:
	RUSTFLAGS="-Ctarget-cpu=native" $(CARGO) build --release -p xtask --features count-alloc

# ─── Cross-architecture local CI (Docker) ───────────────────────────────────────
# Verify lint/build/test in x86_64 or arm64 containers — catches arch-specific
# issues before pushing. Uses the CI container image + rustup for Rust.
#
# Usage:
#   make check-cross                    — run check in opposite arch (auto-detect)
#   make check-cross ARCH=x86_64       — explicit x86_64
#   make check-cross ARCH=arm64        — explicit arm64

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

check-cross:
	@echo "══════════════════════════════════════════════════════════════"
	@echo "  check-cross: running 'make check' on $(ARCH) ($(DOCKER_PLATFORM))"
	@echo "══════════════════════════════════════════════════════════════"
	docker run --rm --platform $(DOCKER_PLATFORM) \
		-v "$$(pwd):/workspace" -w /workspace \
		$(CI_IMAGE) \
		make check

# ─── Benchmarks ────────────────────────────────────────────────────────────────

## Criterion micro-benchmarks — full suite (native CPU for fair comparison)
bench:
	RUSTFLAGS="-Ctarget-cpu=native" $(CARGO) bench $(FEATURES) --bench '*'

## Fast CI benchmark gate (target ≤10 min: compile all, run only 512px with minimal stats)
## Verifies all 246 bench targets compile AND execute without panics.
## Full statistical runs (all sizes, 100 samples) are for local dev or scheduled jobs.
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

bench-ci:
	@set -e; \
	for bench in $(CRITERION_BENCHES); do \
		echo "▶ Running $$bench"; \
		$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench "$$bench" -- \
			--sample-size 10 --warm-up-time 1 --measurement-time 1 --nresamples 100 '/512'; \
	done

## Save baseline on main (CI calls this after merge to main)
bench-baseline:
	@set -e; \
	for bench in $(CRITERION_BENCHES); do \
		echo "▶ Saving baseline for $$bench"; \
		$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench "$$bench" -- \
			--sample-size 10 --warm-up-time 1 --measurement-time 1 --nresamples 100 \
			--save-baseline main '/512'; \
	done

## Compare PR against main baseline — fails if any benchmark regresses >5%
bench-compare:
	@set -e; \
	for bench in $(CRITERION_BENCHES); do \
		echo "▶ Comparing $$bench"; \
		$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench "$$bench" -- \
			--sample-size 10 --warm-up-time 1 --measurement-time 1 --nresamples 100 \
			--baseline main '/512'; \
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

bench-vs:
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
