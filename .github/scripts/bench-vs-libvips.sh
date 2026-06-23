#!/usr/bin/env bash
set -euo pipefail

# E2E benchmark: run all supported operations across standard sizes,
# compare viprs vs libvips, and fail if any ratio exceeds 1.05.
#
# Individual benchmark failures are collected and reported at the end
# rather than aborting the entire suite — we want maximum data.

# Use pre-built binary if available (CI), otherwise compile via cargo.
XTASK="${XTASK_BIN:-cargo xtask}"

mkdir -p /tmp/bench-results

# Core ops tested with JPEG (U8) fixtures.
OPS="load invert bandmean add multiply and equal linear colourspace resize zoom shrink shrinkh shrinkv gauss_blur abs sign"

# Float-only ops tested with EXR (F32) fixtures.
FLOAT_OPS="round floor ceil"

# Thumbnail uses explicit target width smaller than all fixtures (smallest is 512px)
THUMBNAIL_TARGET=256

FAILED_OPS=()

# Standard sizes (JPEG / U8 pipeline)
for size in 512 2048 8192; do
  FIXTURE="tests/fixtures/images/bench_${size}x${size}.jpg"
  if [ ! -f "$FIXTURE" ]; then
    echo "::warning::Missing fixture: $FIXTURE — skipping ${size}px benchmarks"
    continue
  fi
  for op in $OPS; do
    echo "=== $op @ ${size}px ==="
    if ! $XTASK bench "$FIXTURE" "$op" --iterations 20 --json       > "/tmp/bench-results/${op}_${size}.json" 2>/dev/null; then
      echo "::warning::Benchmark failed: $op @ ${size}px"
      FAILED_OPS+=("$op@${size}px")
    fi
  done

  # Thumbnail with explicit downscale target
  echo "=== thumbnail @ ${size}px ==="
  if ! $XTASK bench "$FIXTURE" "thumbnail" "$THUMBNAIL_TARGET" --iterations 20 --json     > "/tmp/bench-results/thumbnail_${size}.json" 2>/dev/null; then
    echo "::warning::Benchmark failed: thumbnail @ ${size}px"
    FAILED_OPS+=("thumbnail@${size}px")
  fi
done

# Workflow/perceptual_enhance (composite pipeline)
FIXTURE="tests/fixtures/images/bench_2048x2048.jpg"
if [ -f "$FIXTURE" ]; then
  echo "=== perceptual_enhance @ 2048px ==="
  if ! $XTASK bench "$FIXTURE" "perceptual_enhance" webp --iterations 20 --json     > "/tmp/bench-results/perceptual_enhance_2048.json" 2>/dev/null; then
    echo "::warning::Benchmark failed: perceptual_enhance @ 2048px"
    FAILED_OPS+=("perceptual_enhance@2048px")
  fi
fi

# EXR format (float pipeline) — includes float-only ops
FIXTURE="tests/fixtures/images/bench_512x512.exr"
if [ -f "$FIXTURE" ]; then
  for op in invert linear add multiply $FLOAT_OPS; do
    echo "=== $op @ 512px EXR ==="
    if ! $XTASK bench "$FIXTURE" "$op" --iterations 20 --json       > "/tmp/bench-results/${op}_512_exr.json" 2>/dev/null; then
      echo "::warning::Benchmark failed: $op @ 512px EXR"
      FAILED_OPS+=("$op@512px-EXR")
    fi
  done
fi

if [ ${#FAILED_OPS[@]} -gt 0 ]; then
  echo ""
  echo "::error::${#FAILED_OPS[@]} benchmark(s) failed: ${FAILED_OPS[*]}"
  exit 1
fi

echo "✅ E2E benchmark suite completed — all operations succeeded."
