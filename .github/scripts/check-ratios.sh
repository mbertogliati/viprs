#!/usr/bin/env bash
set -euo pipefail

# Check viprs/libvips ratios.
# Warn on known gaps above 1.05 and fail only on catastrophic regressions above 2.0.

WARNING_COUNT=0
FAILED=0
for f in /tmp/bench-results/*.json; do
  [ -f "$f" ] || continue
  ratio=$(jq -r '.ratio_p50 // .ratio // empty' "$f" 2>/dev/null || true)
  [ -z "$ratio" ] && continue
  name=$(basename "$f" .json)
  # Skip non-numeric ratios (corrupt JSON or null values)
  if ! echo "$ratio" | grep -qE '^[0-9]+\.?[0-9]*$'; then
    echo "::warning::$name: skipped — non-numeric ratio '$ratio'"
    continue
  fi
  echo "$name: ratio=$ratio"
  if [ "$(echo "$ratio > 2.0" | bc -l)" = "1" ]; then
    echo "::error::$name: viprs/libvips ratio $ratio > 2.0"
    FAILED=$((FAILED + 1))
  elif [ "$(echo "$ratio > 1.05" | bc -l)" = "1" ]; then
    echo "::warning::$name: viprs/libvips ratio $ratio > 1.05"
    WARNING_COUNT=$((WARNING_COUNT + 1))
  fi
done

echo "Summary: $WARNING_COUNT operation(s) in warning zone (1.05-2.0], $FAILED operation(s) in failure zone (>2.0)"

if [ "$FAILED" -gt 0 ]; then
  echo "::error::$FAILED operation(s) exceeded the catastrophic regression threshold (ratio > 2.0)"
  exit 1
fi
if [ "$WARNING_COUNT" -gt 0 ]; then
  echo "::warning::$WARNING_COUNT operation(s) slower than target ratio (> 1.05) but below the failure threshold (≤ 2.0)"
else
  echo "✓ All operations within target ratio (≤ 1.05)"
fi
