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

CARGO := cargo
RUSTFLAGS_CI := -Dwarnings -Adead_code

# Features that require system libraries (libjxl, libheif, libjpeg-turbo, etc.)
# CI runs inside a container with all deps; local devs enable what they have installed.
# Override: make check FEATURES="--features jpeg,png,webp"
FEATURES ?= --all-features

.PHONY: all check ci fmt clippy build test test-all doc deny audit bench bench-vs coverage xtask

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

## Clippy: enforce perf + unwrap/expect ban. Pedantic/nursery are in Cargo.toml [lints.clippy]
## but produce cross-platform false positives (x86 SIMD code not visible on ARM dev machines).
## Allow nursery entirely and specific pedantic lints that are architecture-dependent.
clippy:
	RUSTFLAGS="-A dead_code" $(CARGO) clippy --lib $(FEATURES) -- \
		-D clippy::perf -D clippy::unwrap_used -D clippy::expect_used \
		-A clippy::nursery -A clippy::cast_ptr_alignment

## Compile check (lib mandatory; xtask best-effort — requires system codec libs)
build:
	RUSTFLAGS="$(RUSTFLAGS_CI)" $(CARGO) check --lib $(FEATURES)
	$(CARGO) check -p xtask || echo "xtask check skipped (missing system libs)"

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

## Full test suite with coverage instrumentation (≥90% on ops/ and codecs/).
## Runs all tests (unit + integration + functional) with coverage instrumentation.
## Requires system libs for all codec features.
coverage:
	$(CARGO) llvm-cov $(CONTAINER_FEATURES) --ignore-filename-regex '(benches|tests)' --fail-under-lines 90

## Build xtask release (for benchmark runner — native CPU for fair comparison)
xtask:
	RUSTFLAGS="-Ctarget-cpu=native" $(CARGO) build --release -p xtask --features count-alloc

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

bench-ci:
	$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench '*' \
		-- --sample-size 10 --warm-up-time 1 --measurement-time 1 --nresamples 100 '/512'

## Save baseline on main (CI calls this after merge to main)
bench-baseline:
	$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench '*' \
		-- --sample-size 10 --warm-up-time 1 --measurement-time 1 --nresamples 100 \
		--save-baseline main '/512'

## Compare PR against main baseline — fails if any benchmark regresses >5%
bench-compare:
	$(BENCH_CI_ENV) $(CARGO) bench $(FEATURES) --bench '*' \
		-- --sample-size 10 --warm-up-time 1 --measurement-time 1 --nresamples 100 \
		--baseline main '/512'

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
