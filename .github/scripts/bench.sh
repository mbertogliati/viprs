#!/usr/bin/env bash
set -euo pipefail

# Benchmark CI script — compares against main baseline if available,
# otherwise runs a smoke test to verify all benchmarks compile and execute.

FEATURES="${BENCH_FEATURES:---features default,simd-pulp,fft,heif,webp,tiff,png}"

if find target/criterion -path '*/main/estimates.json' -print -quit 2>/dev/null | grep -q .; then
  echo "▶ Baseline found — comparing against main"
  make bench-compare FEATURES="$FEATURES"
else
  echo "▶ No baseline — running smoke test (first run)"
  make bench-ci FEATURES="$FEATURES"
fi

echo "✅ Benchmarks completed."
