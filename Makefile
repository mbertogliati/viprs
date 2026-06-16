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
RUSTFLAGS_CI := -Dwarnings

# Features that require system libraries (libjxl, libheif, libjpeg-turbo, etc.)
# CI runs inside a container with all deps; local devs enable what they have installed.
# Override: make check FEATURES="--features jpeg,png,webp"
FEATURES ?= --all-features

.PHONY: all check ci fmt clippy build test doc deny audit bench bench-vs coverage xtask

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

## Clippy with all project lints (pedantic + nursery + perf + no unwrap/expect)
clippy:
	RUSTFLAGS="$(RUSTFLAGS_CI)" $(CARGO) clippy --lib $(FEATURES) -- \
		-D clippy::perf -D clippy::unwrap_used -D clippy::expect_used

## Compile check (lib + xtask)
build:
	RUSTFLAGS="$(RUSTFLAGS_CI)" $(CARGO) check --lib $(FEATURES)
	$(CARGO) check -p xtask

## Unit tests (warnings allowed in test code — dead code is expected with partial features)
test:
	$(CARGO) test --lib $(FEATURES)

## Documentation (deny warnings)
doc:
	RUSTDOCFLAGS="-Dwarnings" $(CARGO) doc --no-deps $(FEATURES)

## License/advisory audit
deny:
	$(CARGO) deny check

## Security vulnerability audit
audit:
	$(CARGO) audit

## Coverage (≥90% on ops/ and codecs/)
coverage:
	$(CARGO) llvm-cov --lib $(FEATURES) --ignore-filename-regex '(benches|tests)' --fail-under-lines 90

## Build xtask release (for benchmark runner)
xtask:
	$(CARGO) build --release -p xtask --features count-alloc

# ─── Benchmarks ────────────────────────────────────────────────────────────────

## Criterion micro-benchmarks
bench:
	$(CARGO) bench --bench '*'

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
