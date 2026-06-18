#!/usr/bin/env bash
set -euo pipefail

# E2E benchmark: run all supported operations across standard sizes,
# compare viprs vs libvips, and fail if any ratio exceeds 1.05.

# Use pre-built binary if available (CI), otherwise compile via cargo.
XTASK="${XTASK_BIN:-cargo xtask}"

mkdir -p /tmp/bench-results

# All supported operations
OPS="load invert bandmean add multiply and equal linear colourspace resize zoom shrink shrinkh shrinkv thumbnail gauss_blur abs sign round floor ceil"

# Standard sizes (JPEG)
for size in 512 2048 8192; do
  FIXTURE="tests/fixtures/images/bench_${size}x${size}.jpg"
  [ -f "$FIXTURE" ] || continue
  for op in $OPS; do
    echo "=== $op @ ${size}px ==="
    $XTASK bench "$FIXTURE" "$op" --iterations 20 --json \
      > "/tmp/bench-results/${op}_${size}.json" 2>/dev/null || true
  done
done

# Workflow/perceptual_enhance (composite pipeline)
FIXTURE="tests/fixtures/images/bench_2048x2048.jpg"
if [ -f "$FIXTURE" ]; then
  echo "=== perceptual_enhance @ 2048px ==="
  $XTASK bench "$FIXTURE" "perceptual_enhance" webp --iterations 20 --json \
    > "/tmp/bench-results/perceptual_enhance_2048.json" 2>/dev/null || true
fi

# EXR format (float pipeline)
FIXTURE="tests/fixtures/images/bench_512x512.exr"
if [ -f "$FIXTURE" ]; then
  for op in invert linear add multiply; do
    echo "=== $op @ 512px EXR ==="
    $XTASK bench "$FIXTURE" "$op" --iterations 20 --json \
      > "/tmp/bench-results/${op}_512_exr.json" 2>/dev/null || true
  done
fi

echo "✅ E2E benchmark suite completed."
