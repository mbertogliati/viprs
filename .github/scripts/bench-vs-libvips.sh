#!/usr/bin/env bash
set -euo pipefail

# E2E benchmark: run all supported operations across standard sizes,
# compare viprs vs libvips, and fail if any ratio exceeds 1.05.
#
# Individual benchmark failures are collected and reported at the end
# rather than aborting the entire suite — we want maximum data.

mkdir -p /tmp/bench-results

# All supported operations
OPS="load invert bandmean add multiply and equal linear colourspace resize zoom shrink shrinkh shrinkv thumbnail gauss_blur abs sign round floor ceil"

FAILED_OPS=()

# Standard sizes (JPEG)
for size in 512 2048 8192; do
  FIXTURE="tests/fixtures/images/bench_${size}x${size}.jpg"
  if [ ! -f "$FIXTURE" ]; then
    echo "::warning::Missing fixture: $FIXTURE — skipping ${size}px benchmarks"
    continue
  fi
  for op in $OPS; do
    echo "=== $op @ ${size}px ==="
    if ! cargo xtask bench "$FIXTURE" "$op" --iterations 20 --json \
      > "/tmp/bench-results/${op}_${size}.json" 2>&1; then
      echo "::warning::Benchmark failed: $op @ ${size}px"
      FAILED_OPS+=("$op@${size}px")
    fi
  done
done

# Workflow/perceptual_enhance (composite pipeline)
FIXTURE="tests/fixtures/images/bench_2048x2048.jpg"
if [ -f "$FIXTURE" ]; then
  echo "=== perceptual_enhance @ 2048px ==="
  if ! cargo xtask bench "$FIXTURE" "perceptual_enhance" webp --iterations 20 --json \
    > "/tmp/bench-results/perceptual_enhance_2048.json" 2>&1; then
    echo "::warning::Benchmark failed: perceptual_enhance @ 2048px"
    FAILED_OPS+=("perceptual_enhance@2048px")
  fi
fi

# EXR format (float pipeline)
FIXTURE="tests/fixtures/images/bench_512x512.exr"
if [ -f "$FIXTURE" ]; then
  for op in invert linear add multiply; do
    echo "=== $op @ 512px EXR ==="
    if ! cargo xtask bench "$FIXTURE" "$op" --iterations 20 --json \
      > "/tmp/bench-results/${op}_512_exr.json" 2>&1; then
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
